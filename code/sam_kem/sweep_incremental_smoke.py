"""Two-slice numerical smoke gate for the incremental validation sweep."""

from __future__ import annotations

import csv
import json
import math
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from .artifacts import atomic_write_json, file_sha256
from .config import load_contract
from .experiments import Case, load_case_manifest, validate_case_contract
from .sweep_incremental import (
  EXPECTED_OPERATOR_UPDATES,
  EXPECTED_ROWS,
  OUTPUT_NAMESPACE,
  REVIEWED_EXPANSION_PLAN_SHA256,
  _write_shard_process,
  build_incremental_shards,
  incremental_candidate_map,
  run_incremental_validation_sweep_shards,
  verify_incremental_shard,
)
from .sweep_sharded import _paths


FIXED_SMOKE_KEYS = {
  ("brainweb", "subject04", "z038"),
  ("hecktor", "10", "z072"),
}


def fixed_incremental_smoke_shards(
  cases: list[Case], candidates: dict,
) -> tuple[list[dict], dict[tuple[str, str, str], list[Case]]]:
  """Select the two predeclared shards and their complete 12-case membership."""
  all_shards = build_incremental_shards(cases, candidates)
  selected = [
    shard for shard in all_shards
    if (shard["dataset"], shard["subject_id"], shard["slice_id"]) in FIXED_SMOKE_KEYS
  ]
  if len(selected) != 2 or {
    (item["dataset"], item["subject_id"], item["slice_id"]) for item in selected
  } != FIXED_SMOKE_KEYS:
    raise ValueError("fixed incremental smoke shards are absent or ambiguous")
  grouped: dict[tuple[str, str, str], list[Case]] = defaultdict(list)
  for case in cases:
    key = (case.dataset, case.subject_id, case.slice_id)
    if key in FIXED_SMOKE_KEYS:
      grouped[key].append(case)
  for members in grouped.values():
    members.sort(key=lambda case: (case.count_level, case.realization_id))
  if any(len(members) != 12 for members in grouped.values()):
    raise ValueError("each fixed incremental smoke shard must contain 12 cases")
  return selected, dict(grouped)


def _fragment_summary(output_dir: Path, shard: dict) -> dict:
  fragment, sidecar, marker = _paths(output_dir, shard)
  marker_record = json.loads(marker.read_text(encoding="utf-8"))
  with fragment.open("r", encoding="utf-8", newline="") as handle:
    rows = list(csv.DictReader(handle))
  metrics = [(float(row["nrmse"]), float(row["ssim"])) for row in rows]
  if not all(math.isfinite(value) for pair in metrics for value in pair):
    raise ValueError(f"non-finite metric in smoke shard {shard['shard_id']}")
  return {
    "shard_id": shard["shard_id"], "dataset": shard["dataset"],
    "subject_id": shard["subject_id"], "slice_id": shard["slice_id"],
    "row_count": len(rows),
    "method_row_counts": dict(sorted(Counter(row["method"] for row in rows).items())),
    "nrmse_min": min(value[0] for value in metrics),
    "nrmse_max": max(value[0] for value in metrics),
    "ssim_min": min(value[1] for value in metrics),
    "ssim_max": max(value[1] for value in metrics),
    "elapsed_seconds": float(marker_record["elapsed_seconds"]),
    "process_id": int(marker_record["process_id"]),
    "fragment_sha256": file_sha256(fragment),
    "fragment_manifest_sha256": file_sha256(sidecar),
    "success_marker_sha256": file_sha256(marker),
  }


def run_incremental_validation_sweep_smoke(
  root: Path, config_path: Path, protocol_path: Path, selection_audit_path: Path,
  plan_path: Path, output_dir: Path, audit_path: Path,
) -> dict:
  """Execute only the fixed BW/HE smoke shards and verify strict resume reads."""
  root = root.resolve()
  config_path = config_path.resolve()
  protocol_path = protocol_path.resolve()
  selection_audit_path = selection_audit_path.resolve()
  plan_path = plan_path.resolve()
  output_dir = output_dir.resolve()
  audit_path = audit_path.resolve()
  if output_dir.exists() or audit_path.exists():
    raise FileExistsError("refusing to overwrite incremental smoke artifacts")
  release = run_incremental_validation_sweep_shards(
    root, config_path, protocol_path, selection_audit_path, plan_path,
    root / "logs/validation_hyperparameter_expansion_shard_smoke.json",
    root / OUTPUT_NAMESPACE, workers=8, resume_verified=False,
    release_check_only=True,
  )
  plan = json.loads(plan_path.read_text(encoding="utf-8"))
  candidates = incremental_candidate_map(plan)
  cfg, contract = load_contract(config_path)
  manifest_path = root / str(cfg["outputs"]["validation_manifest"])
  cases = load_case_manifest(root, manifest_path, split="validation")
  validate_case_contract(cases, contract)
  shards, grouped = fixed_incremental_smoke_shards(cases, candidates)
  output_dir.mkdir(parents=True)
  execution_record_path = output_dir / "smoke_execution_record.json"
  atomic_write_json(execution_record_path, {
    "artifact_kind": "validation_hyperparameter_expansion_smoke_execution",
    "schema_version": 1, "status": "running",
    "execution_git_commit": release["execution_git_commit"],
    "plan_sha256": REVIEWED_EXPANSION_PLAN_SHA256,
    "selection_audit_sha256": release["selection_audit_sha256"],
    "config_sha256": release["config_sha256"],
    "protocol_sha256": release["protocol_sha256"],
    "manifest_sha256": release["manifest_sha256"],
    "output_role": "staging_smoke_not_formal_expansion",
    "test_locked_access": "prohibited",
  })
  execution_record_sha256 = file_sha256(execution_record_path)
  payloads = []
  for shard in shards:
    key = (shard["dataset"], shard["subject_id"], shard["slice_id"])
    payloads.append((
      root, config_path, output_dir, shard, grouped[key], candidates,
      release["execution_git_commit"], execution_record_sha256,
      REVIEWED_EXPANSION_PLAN_SHA256,
    ))
  with ProcessPoolExecutor(max_workers=2) as executor:
    written = list(executor.map(_write_shard_process, payloads))
  verified = []
  for shard in shards:
    item = verify_incremental_shard(
      output_dir, shard, candidates,
      execution_commit=release["execution_git_commit"],
      execution_record_sha256=execution_record_sha256,
      plan_sha256=REVIEWED_EXPANSION_PLAN_SHA256, run_id=contract.run_id,
    )
    if item is None:
      raise RuntimeError(f"smoke shard disappeared: {shard['shard_id']}")
    verified.append(item)
  summaries = [_fragment_summary(output_dir, shard) for shard in shards]
  expected_methods = {
    "brainweb": {
      "Bowsher-KEM": 48, "MR-KEM": 18,
      "SAM-KEM-MedSAM2-Qwen3-full": 12, "SelfKEM": 30,
    },
    "hecktor": {
      "Bowsher-KEM": 6, "CT-KEM": 30,
      "SAM-KEM-MedSAM2-Qwen3-full": 36, "SelfKEM": 48,
    },
  }
  if (
    sum(item["row_count"] for item in summaries) != 228
    or any(item["method_row_counts"] != expected_methods[item["dataset"]] for item in summaries)
    or len({item["process_id"] for item in summaries}) != 2
    or {item["shard_id"] for item in written} != {item["shard_id"] for item in verified}
  ):
    raise ValueError("incremental smoke fragments differ from the reviewed plan")
  result = {
    "artifact_kind": "validation_hyperparameter_expansion_shard_smoke",
    "schema_version": 1, "status": "PASS",
    "execution_git_commit": release["execution_git_commit"],
    "plan_sha256": REVIEWED_EXPANSION_PLAN_SHA256,
    "selection_audit_sha256": release["selection_audit_sha256"],
    "merged_csv_sha256": release["merged_csv_sha256"],
    "execution_backend": "scipy_sparse_csr_cpu_float64_multiprocess",
    "worker_processes": 2, "gpu_used": False,
    "fixed_shard_count": 2, "fragments_written": len(written),
    "fragments_resume_verified": len(verified), "total_rows": 228,
    "process_ids_unique": True,
    "formal_declared_result_rows": EXPECTED_ROWS,
    "formal_declared_operator_updates": EXPECTED_OPERATOR_UPDATES,
    "fragments": summaries,
    "output_namespace": output_dir.relative_to(root).as_posix(),
    "output_role": "staging_smoke_not_formal_expansion",
    "formal_output_created": False, "merge_created": False,
    "frozen_hyperparameters_created": False, "test_locked_access": "prohibited",
  }
  atomic_write_json(audit_path, result)
  return result
