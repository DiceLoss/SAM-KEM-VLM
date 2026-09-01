"""Two-case fragment writer and equivalence gate before a formal sharded sweep."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from .artifacts import (
  atomic_csv_writer,
  atomic_write_json,
  file_sha256,
  finalize_csv,
  verify_csv_artifact,
)
from .config import load_contract, require_development_execution
from .experiments import load_case_manifest, validate_case_contract
from .operator_factory import build_discrete_radon
from .prior_registration import validate_registered_validation_prior_configuration
from .protocols import load_protocol
from .reconstruction import operator_support_mask
from .sweep_evaluation import (
  SWEEP_FIELDS,
  evaluate_validation_case,
  parse_sweep_grid,
)
from .sweep_preflight import _commit_allowing_untracked


PRIMARY_KEY = [
  "run_id", "dataset", "subject_id", "slice_id", "realization_id", "count_level",
  "method", "sigma", "radius", "neighbours", "iterations", "alpha",
]


def _metric_hashes(records: list[dict]) -> dict[str, str]:
  methods = {}
  for method in dict.fromkeys(str(row["method"]) for row in records):
    values = np.asarray([
      (float(row["nrmse"]), float(row["ssim"]))
      for row in records if row["method"] == method
    ], dtype="<f8")
    methods[method] = hashlib.sha256(values.tobytes()).hexdigest()
  return methods


def _write_fragment_process(payload: tuple) -> dict:
  root, config_path, protocol_path, case, output_dir, commit = payload
  cfg, contract = load_contract(config_path)
  protocol = load_protocol(protocol_path)
  grid = parse_sweep_grid(protocol)
  operator = build_discrete_radon(
    repo_root=root, size=int(cfg["forward_model"]["image_size"]),
    angles=contract.angles, operator_id=contract.operator_id,
    expected_implementation_sha256=cfg["forward_model"]["implementation_sha256"],
  )
  support = operator_support_mask(operator)
  started = time.perf_counter()
  records = evaluate_validation_case(
    case, operator, support, cfg, contract.run_id, protocol.methods, grid,
  )
  if len(records) != 580:
    raise ValueError(f"case fragment has {len(records)} rows, expected 580")
  fragment_dir = output_dir / case.dataset / case.subject_id / case.slice_id
  fragment = fragment_dir / f"c{case.count_level}_r{case.realization_id}.csv"
  with atomic_csv_writer(fragment, SWEEP_FIELDS) as writer:
    writer.writerows(records)
  sidecar = finalize_csv(
    fragment, schema_name="validation_sweep_case_fragment_v1",
    primary_key=PRIMARY_KEY, split="validation",
  )
  marker = fragment.with_suffix(".SUCCESS.json")
  atomic_write_json(marker, {
    "artifact_kind": "validation_sweep_case_fragment_success", "schema_version": 1,
    "status": "complete", "git_commit": commit,
    "dataset": case.dataset, "subject_id": case.subject_id, "slice_id": case.slice_id,
    "count_level": case.count_level, "realization_id": case.realization_id,
    "row_count": len(records), "fragment_sha256": file_sha256(fragment),
    "fragment_manifest_sha256": file_sha256(sidecar),
    "metric_hashes": _metric_hashes(records),
  })
  return {
    "dataset": case.dataset, "subject_id": case.subject_id, "slice_id": case.slice_id,
    "count_level": case.count_level, "realization_id": case.realization_id,
    "fragment_path": fragment.relative_to(root).as_posix(),
    "fragment_sha256": file_sha256(fragment),
    "fragment_manifest_sha256": file_sha256(sidecar),
    "success_marker_sha256": file_sha256(marker), "row_count": len(records),
    "metric_hashes": _metric_hashes(records),
    "elapsed_seconds": time.perf_counter() - started, "process_id": os.getpid(),
  }


def _verify_fragment(root: Path, item: dict, commit: str) -> None:
  fragment = root / item["fragment_path"]
  record = verify_csv_artifact(fragment)
  marker_path = fragment.with_suffix(".SUCCESS.json")
  marker = json.loads(marker_path.read_text(encoding="utf-8"))
  with fragment.open("r", encoding="utf-8", newline="") as handle:
    rows = list(csv.DictReader(handle))
  metric_hashes = _metric_hashes(rows)
  if (
    record.get("schema_name") != "validation_sweep_case_fragment_v1"
    or record.get("split") != "validation"
    or int(record.get("row_count", -1)) != 580
    or file_sha256(fragment) != item["fragment_sha256"]
    or marker.get("status") != "complete"
    or marker.get("git_commit") != commit
    or marker.get("fragment_sha256") != item["fragment_sha256"]
    or marker.get("row_count") != 580
    or metric_hashes != marker.get("metric_hashes")
    or metric_hashes != item.get("metric_hashes")
  ):
    raise ValueError(f"unverified sweep fragment: {fragment}")


def run_sweep_shard_smoke(
  root: Path, config_path: Path, protocol_path: Path, runtime_preflight_path: Path,
  parallel_preflight_path: Path, output_dir: Path, audit_path: Path,
) -> dict:
  root = root.resolve()
  config_path = config_path.resolve()
  protocol_path = protocol_path.resolve()
  runtime_preflight_path = runtime_preflight_path.resolve()
  parallel_preflight_path = parallel_preflight_path.resolve()
  output_dir = output_dir.resolve()
  audit_path = audit_path.resolve()
  if output_dir.exists() or audit_path.exists():
    raise FileExistsError("refusing to overwrite sweep shard smoke artifacts")
  commit = _commit_allowing_untracked(
    root, {runtime_preflight_path, parallel_preflight_path},
  )
  serial = json.loads(runtime_preflight_path.read_text(encoding="utf-8"))
  parallel = json.loads(parallel_preflight_path.read_text(encoding="utf-8"))
  if (
    serial.get("status") != "PASS"
    or parallel.get("status") != "PASS"
    or parallel.get("worker_processes") != 8
    or parallel.get("result_evaluations_per_sample") != 580
    or parallel.get("formal_output_created") is not False
  ):
    raise ValueError("runtime/scaling evidence is not ready for shard smoke")
  cfg, contract = load_contract(config_path)
  require_development_execution(cfg)
  registration = validate_registered_validation_prior_configuration(root, config_path, cfg)
  protocol = load_protocol(protocol_path)
  if protocol.kind != "hyperparameter_sweep" or protocol.split != "validation":
    raise ValueError("shard smoke requires the validation sweep protocol")
  manifest_path = root / str(cfg["outputs"]["validation_manifest"])
  cases = load_case_manifest(root, manifest_path, split="validation")
  validate_case_contract(cases, contract)
  eligible = [
    case for case in cases if case.count_level == 200000 and case.realization_id == 0
  ]
  samples = [
    min((case for case in eligible if case.dataset == dataset),
        key=lambda case: (case.subject_id, case.slice_id))
    for dataset in ("brainweb", "hecktor")
  ]
  payloads = [
    (root, config_path, protocol_path, case, output_dir, commit) for case in samples
  ]
  with ProcessPoolExecutor(max_workers=2) as executor:
    fragments = list(executor.map(_write_fragment_process, payloads))
  for item in fragments:
    _verify_fragment(root, item, commit)
  serial_by_dataset = {item["dataset"]: item for item in serial["measurements"]}
  equivalence = {}
  for fragment in fragments:
    expected = {
      method: details["metric_array_sha256"]
      for method, details in serial_by_dataset[fragment["dataset"]]["metrics"].items()
    }
    actual = fragment["metric_hashes"]
    equivalence[fragment["dataset"]] = {
      "expected_metric_hashes": expected, "actual_metric_hashes": actual,
      "all_methods_exact": expected == actual,
    }
  if not all(item["all_methods_exact"] for item in equivalence.values()):
    raise ValueError(f"refactored shard evaluator differs from serial gate: {equivalence}")
  result = {
    "artifact_kind": "validation_sweep_shard_writer_smoke", "schema_version": 1,
    "status": "PASS", "git_commit": commit,
    "config_sha256": file_sha256(config_path),
    "protocol_sha256": file_sha256(protocol_path),
    "manifest_sha256": file_sha256(manifest_path),
    "runtime_preflight_sha256": file_sha256(runtime_preflight_path),
    "parallel_preflight_sha256": file_sha256(parallel_preflight_path),
    "prior_registration": registration,
    "execution_backend": "scipy_sparse_csr_cpu_float64_multiprocess",
    "worker_processes": 2, "fragments_written": 2,
    "fragments_resume_verified": 2, "total_rows": 1160,
    "fragments": fragments, "metric_equivalence": equivalence,
    "output_namespace": output_dir.relative_to(root).as_posix(),
    "output_role": "staging_smoke_not_formal_sweep",
    "formal_output_created": False, "frozen_hyperparameters_created": False,
    "test_locked_access": "prohibited",
  }
  atomic_write_json(audit_path, result)
  return result
