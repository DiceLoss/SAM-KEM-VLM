"""SHA-bound, resumable execution of incremental validation-grid shards."""

from __future__ import annotations

import csv
import json
import os
import platform
import sys
import tempfile
import time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import scipy

from .artifacts import (
  atomic_csv_writer,
  atomic_write_json,
  file_sha256,
  finalize_csv,
  verify_csv_artifact,
)
from .config import load_contract, require_development_execution
from .experiments import Case, load_case_manifest, validate_case_contract
from .operator_factory import build_discrete_radon
from .prior_registration import validate_registered_validation_prior_configuration
from .protocols import load_protocol
from .reconstruction import operator_support_mask
from .sweep_evaluation import SWEEP_FIELDS, evaluate_incremental_validation_case
from .sweep_sharded import (
  PRIMARY_KEY,
  THREAD_VARIABLES,
  _clean_commit,
  _paths,
  _shard_set_sha256,
  line_ending_equivalent_sha256,
)
from .sweep_sharding import group_validation_cases


REVIEWED_EXPANSION_PLAN_SHA256 = (
  "5a2d7d1013ec0d11fa190db60dc8802248771a3a378ccc04e48116877069b0d8"
)
EXPECTED_SELECTION_AUDIT_SHA256 = (
  "d2ea068f0dbbb3643512f84637ea450a053a2c2b0b8d233897ea39c2d166af28"
)
EXPECTED_ROWS = 80_568
EXPECTED_OPERATOR_UPDATES = 6_124_905
OUTPUT_NAMESPACE = "runs/dev_trainmax_v1/hyperparameter_expansion_shards"
_WORKER_STATE: dict = {}


def _json(path: Path) -> dict:
  value = json.loads(path.read_text(encoding="utf-8"))
  if not isinstance(value, dict):
    raise ValueError(f"expected a JSON object: {path}")
  return value


def incremental_candidate_map(plan: dict) -> dict[tuple[str, int], list[dict]]:
  """Flatten the reviewed groups while retaining dataset/count applicability."""
  result: dict[tuple[str, int], list[dict]] = defaultdict(list)
  seen = set()
  for group in plan.get("groups", []):
    dataset = str(group["dataset"])
    count = int(group["count_level"])
    method = str(group["method"])
    for item in group["incremental_candidates"]:
      candidate = {
        "method": method,
        "sigma": float(item["sigma"]),
        "radius": int(item["radius"]),
        "neighbours": int(item["neighbours"]),
        "iterations": int(item["iterations"]),
        "alpha": float(item["alpha"]),
      }
      identity = (
        dataset, count, candidate["method"], candidate["sigma"],
        candidate["radius"], candidate["neighbours"],
        candidate["iterations"], candidate["alpha"],
      )
      if identity in seen:
        raise ValueError(f"duplicate incremental candidate: {identity}")
      seen.add(identity)
      result[(dataset, count)].append(candidate)
  counts = Counter()
  for (dataset, _count), items in result.items():
    counts[dataset] += len(items)
  if len(seen) != 76 or dict(counts) != {"brainweb": 36, "hecktor": 40}:
    raise ValueError("incremental candidate inventory differs from the reviewed plan")
  return dict(result)


def build_incremental_shards(cases: list[Case], candidates: dict) -> list[dict]:
  """Attach the dataset-specific incremental row count to all 741 slice shards."""
  shards = group_validation_cases(cases)
  for shard in shards:
    row_count = sum(
      len(candidates[(shard["dataset"], int(key["count_level"]))])
      for key in shard["case_keys"]
    )
    shard["expected_result_rows"] = row_count
  counts = Counter(shard["dataset"] for shard in shards)
  row_counts = Counter(
    (shard["dataset"], int(shard["expected_result_rows"])) for shard in shards
  )
  if (
    len(shards) != 741
    or dict(counts) != {"brainweb": 696, "hecktor": 45}
    or dict(row_counts) != {("brainweb", 108): 696, ("hecktor", 120): 45}
  ):
    raise ValueError("incremental shard inventory differs from the reviewed validation set")
  return shards


def _expected_case_rows(shard: dict, candidates: dict) -> dict[tuple[int, int], int]:
  return {
    (int(key["count_level"]), int(key["realization_id"])): len(
      candidates[(shard["dataset"], int(key["count_level"]))]
    )
    for key in shard["case_keys"]
  }


def _expected_method_rows(shard: dict, candidates: dict) -> dict[str, int]:
  counts: Counter = Counter()
  for key in shard["case_keys"]:
    counts.update(
      item["method"]
      for item in candidates[(shard["dataset"], int(key["count_level"]))]
    )
  return dict(counts)


def _row_candidate_identity(row: dict) -> tuple:
  return (
    row["method"], float(row["sigma"]), int(row["radius"]),
    int(row["neighbours"]), int(row["iterations"]), float(row["alpha"]),
  )


def verify_incremental_shard(
  output_dir: Path, shard: dict, candidates: dict, *, execution_commit: str,
  execution_record_sha256: str, plan_sha256: str, run_id: str,
) -> dict | None:
  """Strictly verify a complete expansion shard or return None when absent."""
  fragment, sidecar, marker_path = _paths(output_dir, shard)
  present = [path.is_file() for path in (fragment, sidecar, marker_path)]
  if not fragment.parent.exists():
    return None
  if not all(present):
    raise ValueError(f"incomplete expansion shard cannot be resumed: {shard['shard_id']}")
  record = verify_csv_artifact(fragment)
  marker = _json(marker_path)
  with fragment.open("r", encoding="utf-8", newline="") as handle:
    rows = list(csv.DictReader(handle))
  case_rows = Counter(
    (int(row["count_level"]), int(row["realization_id"])) for row in rows
  )
  method_rows = Counter(row["method"] for row in rows)
  expected_candidates = {
    (count, *(
      item["method"], item["sigma"], item["radius"], item["neighbours"],
      item["iterations"], item["alpha"],
    ))
    for count in {int(key["count_level"]) for key in shard["case_keys"]}
    for item in candidates[(shard["dataset"], count)]
  }
  valid_rows = all(
    row["run_id"] == run_id
    and row["dataset"] == shard["dataset"]
    and row["subject_id"] == shard["subject_id"]
    and row["slice_id"] == shard["slice_id"]
    and row["split"] == "validation"
    and (int(row["count_level"]), *_row_candidate_identity(row)) in expected_candidates
    for row in rows
  )
  if (
    record.get("schema_name") != "validation_hyperparameter_expansion_slice_shard_v1"
    or record.get("split") != "validation"
    or record.get("primary_key") != PRIMARY_KEY
    or int(record.get("row_count", -1)) != int(shard["expected_result_rows"])
    or len(rows) != int(shard["expected_result_rows"])
    or dict(case_rows) != _expected_case_rows(shard, candidates)
    or dict(method_rows) != _expected_method_rows(shard, candidates)
    or not valid_rows
    or marker.get("artifact_kind") != "validation_hyperparameter_expansion_shard_success"
    or marker.get("status") != "complete"
    or marker.get("shard_id") != shard["shard_id"]
    or marker.get("execution_git_commit") != execution_commit
    or marker.get("execution_record_sha256") != execution_record_sha256
    or marker.get("plan_sha256") != plan_sha256
    or int(marker.get("row_count", -1)) != int(shard["expected_result_rows"])
    or marker.get("fragment_sha256") != file_sha256(fragment)
    or marker.get("fragment_manifest_sha256") != file_sha256(sidecar)
  ):
    raise ValueError(f"expansion shard failed strict verification: {shard['shard_id']}")
  return {
    "shard_id": shard["shard_id"], "dataset": shard["dataset"],
    "subject_id": shard["subject_id"], "slice_id": shard["slice_id"],
    "case_count": int(shard["case_count"]), "row_count": len(rows),
    "method_row_counts": dict(method_rows),
    "fragment_sha256": file_sha256(fragment),
    "fragment_manifest_sha256": file_sha256(sidecar),
    "success_marker_sha256": file_sha256(marker_path),
    "elapsed_seconds": float(marker.get("elapsed_seconds", 0.0)),
    "process_id": int(marker.get("process_id", -1)),
  }


def _worker_runtime(root: Path, config_path: Path) -> dict:
  key = (str(root), str(config_path))
  if _WORKER_STATE.get("key") != key:
    cfg, contract = load_contract(config_path)
    operator = build_discrete_radon(
      repo_root=root, size=int(cfg["forward_model"]["image_size"]),
      angles=contract.angles, operator_id=contract.operator_id,
      expected_implementation_sha256=cfg["forward_model"]["implementation_sha256"],
    )
    _WORKER_STATE.clear()
    _WORKER_STATE.update({
      "key": key, "cfg": cfg, "contract": contract, "operator": operator,
      "support": operator_support_mask(operator),
    })
  return _WORKER_STATE


def _write_shard_process(payload: tuple) -> dict:
  (
    root, config_path, output_dir, shard, cases, candidates,
    execution_commit, execution_record_sha256, plan_sha256,
  ) = payload
  runtime = _worker_runtime(root, config_path)
  fragment, _, marker_path = _paths(output_dir, shard)
  final_directory = fragment.parent
  if final_directory.exists():
    raise FileExistsError(f"refusing to overwrite expansion shard: {shard['shard_id']}")
  final_directory.parent.mkdir(parents=True, exist_ok=True)
  temporary_directory = Path(tempfile.mkdtemp(
    prefix=f'.{shard["shard_id"]}.', dir=final_directory.parent,
  ))
  temporary_fragment = temporary_directory / fragment.name
  temporary_sidecar = temporary_fragment.with_suffix(
    temporary_fragment.suffix + ".manifest.json"
  )
  temporary_marker = temporary_directory / marker_path.name
  started = time.perf_counter()
  row_count = 0
  with atomic_csv_writer(temporary_fragment, SWEEP_FIELDS) as writer:
    for case in cases:
      case_candidates = candidates[(case.dataset, case.count_level)]
      records = evaluate_incremental_validation_case(
        case, runtime["operator"], runtime["support"], runtime["cfg"],
        runtime["contract"].run_id, case_candidates,
      )
      if len(records) != len(case_candidates):
        raise ValueError(f"case in {shard['shard_id']} produced an unexpected row count")
      writer.writerows(records)
      row_count += len(records)
  if row_count != int(shard["expected_result_rows"]):
    raise ValueError(f"expansion shard {shard['shard_id']} produced {row_count} rows")
  finalize_csv(
    temporary_fragment,
    schema_name="validation_hyperparameter_expansion_slice_shard_v1",
    primary_key=PRIMARY_KEY, split="validation",
  )
  atomic_write_json(temporary_marker, {
    "artifact_kind": "validation_hyperparameter_expansion_shard_success",
    "schema_version": 1, "status": "complete",
    "execution_git_commit": execution_commit,
    "execution_record_sha256": execution_record_sha256,
    "plan_sha256": plan_sha256, "shard_id": shard["shard_id"],
    "dataset": shard["dataset"], "subject_id": shard["subject_id"],
    "slice_id": shard["slice_id"], "case_count": int(shard["case_count"]),
    "row_count": row_count, "fragment_sha256": file_sha256(temporary_fragment),
    "fragment_manifest_sha256": file_sha256(temporary_sidecar),
    "elapsed_seconds": time.perf_counter() - started, "process_id": os.getpid(),
  })
  os.replace(temporary_directory, final_directory)
  result = verify_incremental_shard(
    output_dir, shard, candidates, execution_commit=execution_commit,
    execution_record_sha256=execution_record_sha256,
    plan_sha256=plan_sha256, run_id=runtime["contract"].run_id,
  )
  if result is None:
    raise RuntimeError(f"new expansion shard disappeared: {shard['shard_id']}")
  result["elapsed_seconds"] = time.perf_counter() - started
  result["process_id"] = os.getpid()
  return result


def _write_progress(
  path: Path, *, status: str, total: int, reused: int, written: int,
  failed_shard: str | None = None,
) -> None:
  atomic_write_json(path, {
    "artifact_kind": "validation_hyperparameter_expansion_progress",
    "schema_version": 1, "status": status, "shards_total": total,
    "shards_reused_verified": reused, "shards_written": written,
    "shards_completed": reused + written, "failed_shard": failed_shard,
    "updated_utc": datetime.now(timezone.utc).isoformat(),
  }, replace=True)


def run_incremental_validation_sweep_shards(
  root: Path, config_path: Path, protocol_path: Path, selection_audit_path: Path,
  plan_path: Path, output_dir: Path, *, workers: int, resume_verified: bool,
  release_check_only: bool = False,
) -> dict:
  """Run or strictly resume the reviewed 76-candidate incremental shell."""
  root = root.resolve()
  config_path = config_path.resolve()
  protocol_path = protocol_path.resolve()
  selection_audit_path = selection_audit_path.resolve()
  plan_path = plan_path.resolve()
  output_dir = output_dir.resolve()
  if workers != 8:
    raise ValueError("the expansion release requires exactly 8 worker processes")
  thread_environment = {name: os.environ.get(name) for name in THREAD_VARIABLES}
  if any(value is not None for value in thread_environment.values()):
    raise RuntimeError(f"expansion requires unset BLAS thread overrides: {thread_environment}")
  execution_commit = _clean_commit(root)
  if file_sha256(plan_path) != REVIEWED_EXPANSION_PLAN_SHA256:
    raise ValueError("expansion plan is not the researcher-reviewed artifact")
  if file_sha256(selection_audit_path) != EXPECTED_SELECTION_AUDIT_SHA256:
    raise ValueError("selection audit differs from the expansion plan input")
  plan = _json(plan_path)
  selection_audit = _json(selection_audit_path)
  if (
    plan.get("status") != "ready_for_researcher_review_no_execution"
    or plan.get("selection_audit_sha256") != EXPECTED_SELECTION_AUDIT_SHA256
    or plan.get("incremental_candidate_count") != 76
    or plan.get("expected_incremental_result_rows") != EXPECTED_ROWS
    or plan.get("expected_incremental_operator_updates") != EXPECTED_OPERATOR_UPDATES
    or plan.get("formal_expansion_output_created") is not False
    or plan.get("frozen_hyperparameters_created") is not False
    or plan.get("test_locked_access") != "prohibited"
    or selection_audit.get("status") != "BLOCKED_grid_boundary_expansion_required"
  ):
    raise ValueError("expansion plan is not ready for released execution")
  cfg, contract = load_contract(config_path)
  require_development_execution(cfg)
  registration = validate_registered_validation_prior_configuration(root, config_path, cfg)
  protocol = load_protocol(protocol_path)
  manifest_path = root / str(cfg["outputs"]["validation_manifest"])
  merged_path = root / str(cfg["outputs"]["hyperparameter_sweep"])
  frozen_path = root / str(cfg["outputs"]["frozen_hyperparameters"])
  if (
    protocol.kind != "hyperparameter_sweep" or protocol.split != "validation"
    or file_sha256(config_path) != selection_audit.get("config_sha256")
    or not line_ending_equivalent_sha256(
      protocol_path, str(selection_audit.get("protocol_sha256")),
    )
    or file_sha256(merged_path) != plan.get("merged_csv_sha256")
    or registration != selection_audit.get("prior_registration")
    or output_dir.relative_to(root).as_posix() != OUTPUT_NAMESPACE
    or frozen_path.exists()
  ):
    raise ValueError("expansion inputs differ from the reviewed selection chain")
  candidates = incremental_candidate_map(plan)
  cases = load_case_manifest(root, manifest_path, split="validation")
  validate_case_contract(cases, contract)
  shards = build_incremental_shards(cases, candidates)
  grouped_cases: dict[tuple[str, str, str], list[Case]] = defaultdict(list)
  for case in cases:
    grouped_cases[(case.dataset, case.subject_id, case.slice_id)].append(case)
  for members in grouped_cases.values():
    members.sort(key=lambda case: (case.count_level, case.realization_id))

  if release_check_only:
    if resume_verified:
      raise ValueError("release check cannot be combined with --resume-verified")
    if output_dir.exists():
      raise FileExistsError("release check requires an absent expansion namespace")
    return {
      "artifact_kind": "validation_hyperparameter_expansion_release_check",
      "schema_version": 1, "status": "PASS_ready_for_incremental_shard_execution",
      "execution_git_commit": execution_commit,
      "plan_sha256": REVIEWED_EXPANSION_PLAN_SHA256,
      "selection_audit_sha256": EXPECTED_SELECTION_AUDIT_SHA256,
      "config_sha256": file_sha256(config_path),
      "protocol_sha256": file_sha256(protocol_path),
      "protocol_binding": "exact_or_lf_crlf_equivalent_bytes",
      "manifest_sha256": file_sha256(manifest_path),
      "merged_csv_sha256": file_sha256(merged_path),
      "prior_registration": registration,
      "execution_backend": "scipy_sparse_csr_cpu_float64_multiprocess",
      "worker_processes": workers, "gpu_used": False,
      "thread_environment": thread_environment,
      "shard_count": len(shards),
      "dataset_shard_counts": dict(Counter(item["dataset"] for item in shards)),
      "dataset_rows_per_shard": {"brainweb": 108, "hecktor": 120},
      "total_cases": len(cases), "incremental_candidate_count": 76,
      "total_result_rows": EXPECTED_ROWS,
      "total_operator_updates": EXPECTED_OPERATOR_UPDATES,
      "formal_output_namespace": OUTPUT_NAMESPACE,
      "formal_output_created": False, "merge_created": False,
      "frozen_hyperparameters_created": False, "test_locked_access": "prohibited",
    }

  execution_record_path = output_dir / "execution_record.json"
  progress_path = output_dir / "progress.json"
  completion_path = output_dir / "shard_batch_SUCCESS.json"
  if output_dir.exists():
    if not resume_verified:
      raise FileExistsError("expansion namespace exists; use --resume-verified")
    if not execution_record_path.is_file():
      raise ValueError("expansion namespace lacks its immutable execution record")
    execution_record = _json(execution_record_path)
    if (
      execution_record.get("execution_git_commit") != execution_commit
      or execution_record.get("plan_sha256") != REVIEWED_EXPANSION_PLAN_SHA256
      or execution_record.get("selection_audit_sha256") != EXPECTED_SELECTION_AUDIT_SHA256
      or execution_record.get("config_sha256") != file_sha256(config_path)
      or execution_record.get("protocol_sha256") != file_sha256(protocol_path)
      or execution_record.get("manifest_sha256") != file_sha256(manifest_path)
      or execution_record.get("worker_processes") != workers
    ):
      raise ValueError("resume identity differs from the existing expansion run")
  else:
    output_dir.mkdir(parents=True)
    execution_record = {
      "artifact_kind": "validation_hyperparameter_expansion_execution",
      "schema_version": 1, "status": "running",
      "execution_git_commit": execution_commit,
      "plan_sha256": REVIEWED_EXPANSION_PLAN_SHA256,
      "selection_audit_sha256": EXPECTED_SELECTION_AUDIT_SHA256,
      "config_sha256": file_sha256(config_path),
      "protocol_sha256": file_sha256(protocol_path),
      "protocol_binding": "exact_or_lf_crlf_equivalent_bytes",
      "manifest_sha256": file_sha256(manifest_path),
      "merged_csv_sha256": file_sha256(merged_path),
      "execution_backend": "scipy_sparse_csr_cpu_float64_multiprocess",
      "worker_processes": workers, "gpu_used": False,
      "thread_environment": thread_environment,
      "python": sys.version, "platform": platform.platform(),
      "numpy_version": np.__version__, "scipy_version": scipy.__version__,
      "shard_count": len(shards), "total_cases": len(cases),
      "total_result_rows": EXPECTED_ROWS,
      "total_operator_updates": EXPECTED_OPERATOR_UPDATES,
      "test_locked_access": "prohibited",
      "created_utc": datetime.now(timezone.utc).isoformat(),
    }
    atomic_write_json(execution_record_path, execution_record)
  execution_record_sha256 = file_sha256(execution_record_path)

  verified = []
  pending = []
  for shard in shards:
    item = verify_incremental_shard(
      output_dir, shard, candidates, execution_commit=execution_commit,
      execution_record_sha256=execution_record_sha256,
      plan_sha256=REVIEWED_EXPANSION_PLAN_SHA256, run_id=contract.run_id,
    )
    (pending if item is None else verified).append(shard if item is None else item)
  if verified and not resume_verified:
    raise FileExistsError("verified expansion shards exist; use --resume-verified")
  if completion_path.exists():
    completion = _json(completion_path)
    if (
      pending or completion.get("status") != "complete_pending_incremental_integrity_merge"
      or completion.get("execution_git_commit") != execution_commit
      or completion.get("execution_record_sha256") != execution_record_sha256
      or completion.get("plan_sha256") != REVIEWED_EXPANSION_PLAN_SHA256
      or completion.get("shards_completed") != len(shards)
      or completion.get("total_result_rows") != EXPECTED_ROWS
      or completion.get("shard_set_sha256") != _shard_set_sha256(verified)
      or completion.get("merge_created") is not False
      or completion.get("frozen_hyperparameters_created") is not False
    ):
      raise ValueError("existing expansion completion marker failed verification")
    _write_progress(
      progress_path, status="complete", total=len(shards),
      reused=len(verified), written=0,
    )
    return completion
  _write_progress(
    progress_path, status="running", total=len(shards), reused=len(verified), written=0,
  )
  written = []
  failed_shard = None
  executor = ProcessPoolExecutor(max_workers=workers)
  futures = {}
  try:
    for shard in pending:
      key = (shard["dataset"], shard["subject_id"], shard["slice_id"])
      payload = (
        root, config_path, output_dir, shard, grouped_cases[key], candidates,
        execution_commit, execution_record_sha256, REVIEWED_EXPANSION_PLAN_SHA256,
      )
      futures[executor.submit(_write_shard_process, payload)] = shard["shard_id"]
    for future in as_completed(futures):
      failed_shard = futures[future]
      written.append(future.result())
      failed_shard = None
      print(json.dumps({
        "status": "running", "shards_total": len(shards),
        "shards_reused_verified": len(verified), "shards_written": len(written),
        "shards_completed": len(verified) + len(written),
        "latest_shard": written[-1]["shard_id"],
      }, sort_keys=True), flush=True)
      _write_progress(
        progress_path, status="running", total=len(shards),
        reused=len(verified), written=len(written),
      )
  except BaseException:
    for future in futures:
      future.cancel()
    _write_progress(
      progress_path, status="interrupted", total=len(shards),
      reused=len(verified), written=len(written), failed_shard=failed_shard,
    )
    executor.shutdown(wait=True, cancel_futures=True)
    raise
  else:
    executor.shutdown(wait=True)

  completed = []
  for shard in shards:
    item = verify_incremental_shard(
      output_dir, shard, candidates, execution_commit=execution_commit,
      execution_record_sha256=execution_record_sha256,
      plan_sha256=REVIEWED_EXPANSION_PLAN_SHA256, run_id=contract.run_id,
    )
    if item is None:
      raise RuntimeError(f"expansion shard missing after run: {shard['shard_id']}")
    completed.append(item)
  result = {
    "artifact_kind": "validation_hyperparameter_expansion_shard_batch_completion",
    "schema_version": 1, "status": "complete_pending_incremental_integrity_merge",
    "execution_git_commit": execution_commit,
    "execution_record_sha256": execution_record_sha256,
    "plan_sha256": REVIEWED_EXPANSION_PLAN_SHA256,
    "worker_processes": workers, "shards_total": len(shards),
    "shards_reused_verified": len(verified), "shards_written": len(written),
    "shards_completed": len(completed), "total_cases": len(cases),
    "total_result_rows": sum(int(item["row_count"]) for item in completed),
    "total_operator_updates": EXPECTED_OPERATOR_UPDATES,
    "shard_set_sha256": _shard_set_sha256(completed),
    "merge_created": False, "frozen_hyperparameters_created": False,
    "test_locked_access": "prohibited",
  }
  atomic_write_json(completion_path, result)
  _write_progress(
    progress_path, status="complete", total=len(shards),
    reused=len(verified), written=len(written),
  )
  return result
