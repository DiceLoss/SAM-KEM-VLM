"""SHA-bound release identity for the reviewed third incremental shell."""

from __future__ import annotations

import json
import math
import os
import platform
import sys
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import scipy

from .artifacts import atomic_write_json, file_sha256
from .config import load_contract, require_development_execution
from .experiments import Case, load_case_manifest, validate_case_contract
from .prior_registration import validate_registered_validation_prior_configuration
from .protocols import load_protocol
from .sweep_expansion_round3 import (
  EXPECTED_CANDIDATE_COUNTS,
  EXPECTED_CANDIDATES,
  EXPECTED_OPERATOR_UPDATES,
  EXPECTED_ROWS,
  REVIEWED_ROUND2_UNION_SELECTION_AUDIT_SHA256,
)
from .sweep_incremental_round2 import (
  EXPECTED_ORIGINAL_MERGED_CSV_SHA256,
  EXPECTED_ROUND1_MERGED_CSV_SHA256,
  REVIEWED_ROUND2_PLAN_SHA256,
)
from .sweep_incremental_round2_merge import ROUND2_MERGED_NAME
from .sweep_incremental import _write_shard_process, verify_incremental_shard
from .sweep_sharded import (
  THREAD_VARIABLES,
  _clean_commit,
  _shard_set_sha256,
  line_ending_equivalent_sha256,
)
from .sweep_sharding import group_validation_cases
from .sweep_union_selection_round2 import REVIEWED_ROUND2_MERGE_AUDIT_SHA256


REVIEWED_ROUND3_PLAN_SHA256 = (
  "1559ead5c7ba844d78feb1b17ebbb481b3830d4bf7c6f45215fd21bd8036a192"
)
REVIEWED_ROUND3_SMOKE_AUDIT_SHA256 = (
  "a0bf660f2b1452c68e312e5519ac102c9ec937e748172299e4f434ece53efb19"
)
REVIEWED_ROUND3_SMOKE_EXECUTION_COMMIT = (
  "452e4ccefbc9b78dc607e626eb47707dca68c9af"
)
EXPECTED_ROUND2_MERGED_CSV_SHA256 = (
  "6de1e06fd0fd31e38b10a83230a4421b4173b831541e1b93445947bb62a0f192"
)
OUTPUT_NAMESPACE = "runs/dev_trainmax_v1/hyperparameter_expansion_round3_shards"
FIXED_SMOKE_KEYS = {
  ("brainweb", "subject04", "z038"),
  ("hecktor", "10", "z072"),
}
EXPECTED_ROWS_PER_SHARD = {"brainweb": 45, "hecktor": 15}
EXPECTED_METHOD_ROWS = {
  "brainweb": {"Bowsher-KEM": 27, "MR-KEM": 18},
  "hecktor": {
    "Bowsher-KEM": 3,
    "CT-KEM": 9,
    "SAM-KEM-MedSAM2-Qwen3-full": 3,
  },
}


def _json(path: Path) -> dict:
  value = json.loads(path.read_text(encoding="utf-8"))
  if not isinstance(value, dict):
    raise TypeError(f"expected a JSON object: {path}")
  return value


def round3_candidate_map(plan: dict) -> dict[tuple[str, int], list[dict]]:
  """Flatten the reviewed round-3 groups by dataset and count level."""
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
        raise ValueError(f"duplicate round-3 candidate: {identity}")
      seen.add(identity)
      result[(dataset, count)].append(candidate)
  # The generic shard writer indexes every validation case directly.  Retain
  # uncovered cells explicitly so a reviewed zero-candidate cell is skipped
  # instead of being confused with a malformed plan.
  for dataset in ("brainweb", "hecktor"):
    for count in (100_000, 200_000, 500_000, 1_000_000):
      result.setdefault((dataset, count), [])
  counts = Counter()
  for (dataset, _count), items in result.items():
    counts[dataset] += len(items)
  if len(seen) != EXPECTED_CANDIDATES or dict(counts) != EXPECTED_CANDIDATE_COUNTS:
    raise ValueError("round-3 candidate inventory differs from the reviewed plan")
  return dict(result)


def build_round3_shards(cases: list[Case], candidates: dict) -> list[dict]:
  """Attach exact round-3 row counts to all validation slice shards."""
  shards = group_validation_cases(cases)
  for shard in shards:
    shard["expected_result_rows"] = sum(
      len(candidates.get((shard["dataset"], int(key["count_level"])), ()))
      for key in shard["case_keys"]
    )
  counts = Counter(shard["dataset"] for shard in shards)
  row_counts = Counter(
    (shard["dataset"], int(shard["expected_result_rows"])) for shard in shards
  )
  if (
    len(shards) != 741
    or dict(counts) != {"brainweb": 696, "hecktor": 45}
    or dict(row_counts) != {("brainweb", 45): 696, ("hecktor", 15): 45}
    or sum(int(shard["expected_result_rows"]) for shard in shards) != EXPECTED_ROWS
  ):
    raise ValueError("round-3 shard inventory differs from the reviewed validation set")
  return shards


def release_check_round3_incremental_validation_sweep(
  root: Path, config_path: Path, protocol_path: Path,
  round2_plan_path: Path, union_audit_path: Path,
  round2_merge_audit_path: Path, round3_plan_path: Path,
  output_dir: Path, *, workers: int, release_check_only: bool,
  allow_existing_output: bool = False,
) -> dict:
  """Validate the round-3 execution identity without releasing formal execution."""
  root = root.resolve()
  config_path = config_path.resolve()
  protocol_path = protocol_path.resolve()
  round2_plan_path = round2_plan_path.resolve()
  union_audit_path = union_audit_path.resolve()
  round2_merge_audit_path = round2_merge_audit_path.resolve()
  round3_plan_path = round3_plan_path.resolve()
  output_dir = output_dir.resolve()
  if not release_check_only:
    raise RuntimeError("formal round-3 execution awaits a reviewed smoke audit")
  if workers != 8:
    raise ValueError("the round-3 release requires exactly 8 worker processes")
  thread_environment = {name: os.environ.get(name) for name in THREAD_VARIABLES}
  if any(value is not None for value in thread_environment.values()):
    raise RuntimeError(
      f"round-3 execution requires unset BLAS thread overrides: {thread_environment}"
    )
  execution_commit = _clean_commit(root)
  if (
    file_sha256(round2_plan_path) != REVIEWED_ROUND2_PLAN_SHA256
    or file_sha256(union_audit_path)
    != REVIEWED_ROUND2_UNION_SELECTION_AUDIT_SHA256
    or file_sha256(round2_merge_audit_path) != REVIEWED_ROUND2_MERGE_AUDIT_SHA256
    or file_sha256(round3_plan_path) != REVIEWED_ROUND3_PLAN_SHA256
  ):
    raise ValueError("round-3 inputs are not the researcher-reviewed artifacts")

  round2_plan = _json(round2_plan_path)
  union_audit = _json(union_audit_path)
  round2_merge_audit = _json(round2_merge_audit_path)
  round3_plan = _json(round3_plan_path)
  if (
    round3_plan.get("artifact_kind")
    != "validation_hyperparameter_incremental_expansion_round3_plan"
    or round3_plan.get("status") != "ready_for_researcher_review_no_execution"
    or round3_plan.get("expansion_round") != 3
    or round3_plan.get("source_union_selection_audit_sha256")
    != REVIEWED_ROUND2_UNION_SELECTION_AUDIT_SHA256
    or round3_plan.get("previous_round2_expansion_plan_sha256")
    != REVIEWED_ROUND2_PLAN_SHA256
    or round3_plan.get("round2_incremental_merge_audit_sha256")
    != REVIEWED_ROUND2_MERGE_AUDIT_SHA256
    or round3_plan.get("original_merged_csv_sha256")
    != EXPECTED_ORIGINAL_MERGED_CSV_SHA256
    or round3_plan.get("round1_merged_csv_sha256")
    != EXPECTED_ROUND1_MERGED_CSV_SHA256
    or round3_plan.get("round2_merged_csv_sha256")
    != EXPECTED_ROUND2_MERGED_CSV_SHA256
    or round3_plan.get("source_expanded_boundary_group_count") != 12
    or round3_plan.get("unchanged_union_selection_count") != 36
    or round3_plan.get("incremental_candidate_counts") != EXPECTED_CANDIDATE_COUNTS
    or round3_plan.get("incremental_candidate_count") != EXPECTED_CANDIDATES
    or round3_plan.get("expected_incremental_result_rows") != EXPECTED_ROWS
    or round3_plan.get("expected_incremental_operator_updates")
    != EXPECTED_OPERATOR_UPDATES
    or round3_plan.get("formal_expansion_output_created") is not False
    or round3_plan.get("frozen_hyperparameters_created") is not False
    or round3_plan.get("test_locked_access") != "prohibited"
    or round2_plan.get("incremental_candidate_count") != 39
    or union_audit.get("status")
    != "BLOCKED_expanded_boundary_requires_additional_shell"
    or union_audit.get("expanded_boundary_selection_count") != 12
    or union_audit.get("round2_merge_audit_sha256")
    != REVIEWED_ROUND2_MERGE_AUDIT_SHA256
    or union_audit.get("round2_merged_csv_sha256")
    != EXPECTED_ROUND2_MERGED_CSV_SHA256
    or union_audit.get("frozen_hyperparameters_created") is not False
    or round2_merge_audit.get("status")
    != "PASS_complete_pending_round2_union_selection_audit"
    or round2_merge_audit.get("round2_plan_sha256") != REVIEWED_ROUND2_PLAN_SHA256
    or round2_merge_audit.get("round2_merged_csv_sha256")
    != EXPECTED_ROUND2_MERGED_CSV_SHA256
  ):
    raise ValueError("round-3 execution identity is internally inconsistent")

  cfg, contract = load_contract(config_path)
  require_development_execution(cfg)
  registration = validate_registered_validation_prior_configuration(
    root, config_path, cfg,
  )
  protocol = load_protocol(protocol_path)
  manifest_path = root / str(cfg["outputs"]["validation_manifest"])
  original_path = root / str(cfg["outputs"]["hyperparameter_sweep"])
  round1_path = root / (
    "runs/dev_trainmax_v1/hyperparameter_expansion/"
    "validation_hyperparameter_expansion.csv"
  )
  round2_path = root / "runs/dev_trainmax_v1/hyperparameter_expansion_round2" / (
    ROUND2_MERGED_NAME
  )
  frozen_path = root / str(cfg["outputs"]["frozen_hyperparameters"])
  if (
    protocol.kind != "hyperparameter_sweep"
    or protocol.split != "validation"
    or file_sha256(config_path) != union_audit.get("config_sha256")
    or not line_ending_equivalent_sha256(
      protocol_path, str(union_audit.get("protocol_sha256")),
    )
    or file_sha256(original_path) != EXPECTED_ORIGINAL_MERGED_CSV_SHA256
    or file_sha256(round1_path) != EXPECTED_ROUND1_MERGED_CSV_SHA256
    or file_sha256(round2_path) != EXPECTED_ROUND2_MERGED_CSV_SHA256
    or registration != union_audit.get("prior_registration")
    or output_dir.relative_to(root).as_posix() != OUTPUT_NAMESPACE
    or (output_dir.exists() and not allow_existing_output)
    or frozen_path.exists()
  ):
    raise ValueError("round-3 runtime inputs differ from the reviewed union chain")

  candidates = round3_candidate_map(round3_plan)
  cases = load_case_manifest(root, manifest_path, split="validation")
  validate_case_contract(cases, contract)
  shards = build_round3_shards(cases, candidates)
  return {
    "artifact_kind": "validation_hyperparameter_expansion_round3_release_check",
    "schema_version": 1,
    "status": "PASS_ready_for_round3_incremental_shard_smoke",
    "execution_git_commit": execution_commit,
    "round3_plan_sha256": REVIEWED_ROUND3_PLAN_SHA256,
    "round2_plan_sha256": REVIEWED_ROUND2_PLAN_SHA256,
    "union_selection_audit_sha256": (
      REVIEWED_ROUND2_UNION_SELECTION_AUDIT_SHA256
    ),
    "round2_merge_audit_sha256": REVIEWED_ROUND2_MERGE_AUDIT_SHA256,
    "config_sha256": file_sha256(config_path),
    "protocol_sha256": file_sha256(protocol_path),
    "protocol_binding": "exact_or_lf_crlf_equivalent_bytes",
    "manifest_sha256": file_sha256(manifest_path),
    "original_merged_csv_sha256": file_sha256(original_path),
    "round1_merged_csv_sha256": file_sha256(round1_path),
    "round2_merged_csv_sha256": file_sha256(round2_path),
    "execution_backend": "scipy_sparse_csr_cpu_float64_multiprocess",
    "worker_processes": workers,
    "gpu_used": False,
    "thread_environment": thread_environment,
    "shard_count": len(shards),
    "dataset_shard_counts": dict(Counter(item["dataset"] for item in shards)),
    "dataset_rows_per_shard": EXPECTED_ROWS_PER_SHARD,
    "incremental_candidate_count": EXPECTED_CANDIDATES,
    "total_cases": len(cases),
    "total_result_rows": EXPECTED_ROWS,
    "total_operator_updates": EXPECTED_OPERATOR_UPDATES,
    "prior_registration": registration,
    "formal_output_namespace": OUTPUT_NAMESPACE,
    "formal_output_created": False,
    "smoke_audit_created": False,
    "merge_created": False,
    "union_selection_audit_created": False,
    "frozen_hyperparameters_created": False,
    "test_locked_access": "prohibited",
  }


def validate_reviewed_round3_smoke(smoke: dict) -> None:
  """Require the exact reviewed fixed-slice Round-3 numerical evidence."""
  fragments = smoke.get("fragments", [])
  identities = {
    (item.get("dataset"), item.get("subject_id"), item.get("slice_id"))
    for item in fragments
  }
  finite_metrics = all(
    math.isfinite(float(item[key]))
    for item in fragments
    for key in ("nrmse_min", "nrmse_max", "ssim_min", "ssim_max")
  )
  if (
    smoke.get("artifact_kind")
    != "validation_hyperparameter_expansion_round3_shard_smoke"
    or smoke.get("status") != "PASS"
    or smoke.get("execution_git_commit") != REVIEWED_ROUND3_SMOKE_EXECUTION_COMMIT
    or smoke.get("round3_plan_sha256") != REVIEWED_ROUND3_PLAN_SHA256
    or smoke.get("round2_plan_sha256") != REVIEWED_ROUND2_PLAN_SHA256
    or smoke.get("union_selection_audit_sha256")
    != REVIEWED_ROUND2_UNION_SELECTION_AUDIT_SHA256
    or smoke.get("round2_merge_audit_sha256")
    != REVIEWED_ROUND2_MERGE_AUDIT_SHA256
    or smoke.get("original_merged_csv_sha256")
    != EXPECTED_ORIGINAL_MERGED_CSV_SHA256
    or smoke.get("round1_merged_csv_sha256")
    != EXPECTED_ROUND1_MERGED_CSV_SHA256
    or smoke.get("round2_merged_csv_sha256")
    != EXPECTED_ROUND2_MERGED_CSV_SHA256
    or smoke.get("execution_backend")
    != "scipy_sparse_csr_cpu_float64_multiprocess"
    or smoke.get("worker_processes") != 2
    or smoke.get("gpu_used") is not False
    or smoke.get("fixed_shard_count") != 2
    or smoke.get("fragments_written") != 2
    or smoke.get("fragments_resume_verified") != 2
    or smoke.get("total_rows") != 60
    or smoke.get("process_ids_unique") is not True
    or smoke.get("formal_declared_result_rows") != EXPECTED_ROWS
    or smoke.get("formal_declared_operator_updates") != EXPECTED_OPERATOR_UPDATES
    or identities != FIXED_SMOKE_KEYS
    or len(fragments) != 2
    or len({int(item["process_id"]) for item in fragments}) != 2
    or any(
      int(item["row_count"]) != EXPECTED_ROWS_PER_SHARD[item["dataset"]]
      or item["method_row_counts"] != EXPECTED_METHOD_ROWS[item["dataset"]]
      for item in fragments
    )
    or not finite_metrics
    or smoke.get("formal_output_created") is not False
    or smoke.get("merge_created") is not False
    or smoke.get("union_selection_audit_created") is not False
    or smoke.get("frozen_hyperparameters_created") is not False
    or smoke.get("test_locked_access") != "prohibited"
  ):
    raise ValueError("round-3 smoke audit is not the reviewed PASS artifact")


def _write_round3_progress(
  path: Path, *, status: str, total: int, reused: int, written: int,
  failed_shard: str | None = None,
) -> None:
  atomic_write_json(path, {
    "artifact_kind": "validation_hyperparameter_expansion_round3_progress",
    "schema_version": 1,
    "status": status,
    "shards_total": total,
    "shards_reused_verified": reused,
    "shards_written": written,
    "shards_completed": reused + written,
    "failed_shard": failed_shard,
    "updated_utc": datetime.now(timezone.utc).isoformat(),
  }, replace=True)


def run_round3_incremental_validation_sweep_shards(
  root: Path, config_path: Path, protocol_path: Path,
  round2_plan_path: Path, union_audit_path: Path,
  round2_merge_audit_path: Path, round3_plan_path: Path,
  smoke_audit_path: Path, output_dir: Path, *, workers: int,
  resume_verified: bool, release_check_only: bool = False,
) -> dict:
  """Release-check, run, or strictly resume the reviewed Round-3 shell."""
  root = root.resolve()
  config_path = config_path.resolve()
  protocol_path = protocol_path.resolve()
  round2_plan_path = round2_plan_path.resolve()
  union_audit_path = union_audit_path.resolve()
  round2_merge_audit_path = round2_merge_audit_path.resolve()
  round3_plan_path = round3_plan_path.resolve()
  smoke_audit_path = smoke_audit_path.resolve()
  output_dir = output_dir.resolve()
  release = release_check_round3_incremental_validation_sweep(
    root, config_path, protocol_path, round2_plan_path, union_audit_path,
    round2_merge_audit_path, round3_plan_path, output_dir,
    workers=workers, release_check_only=True, allow_existing_output=True,
  )
  if file_sha256(smoke_audit_path) != REVIEWED_ROUND3_SMOKE_AUDIT_SHA256:
    raise ValueError("round-3 smoke is not the researcher-reviewed artifact")
  smoke = _json(smoke_audit_path)
  validate_reviewed_round3_smoke(smoke)

  cfg, contract = load_contract(config_path)
  manifest_path = root / str(cfg["outputs"]["validation_manifest"])
  frozen_path = root / str(cfg["outputs"]["frozen_hyperparameters"])
  if frozen_path.exists():
    raise ValueError("round-3 execution cannot follow a hyperparameter freeze")
  plan = _json(round3_plan_path)
  candidates = round3_candidate_map(plan)
  cases = load_case_manifest(root, manifest_path, split="validation")
  validate_case_contract(cases, contract)
  shards = build_round3_shards(cases, candidates)
  grouped_cases: dict[tuple[str, str, str], list[Case]] = defaultdict(list)
  for case in cases:
    grouped_cases[(case.dataset, case.subject_id, case.slice_id)].append(case)
  for members in grouped_cases.values():
    members.sort(key=lambda case: (case.count_level, case.realization_id))

  common = {
    "execution_git_commit": release["execution_git_commit"],
    "round3_plan_sha256": REVIEWED_ROUND3_PLAN_SHA256,
    "round2_plan_sha256": REVIEWED_ROUND2_PLAN_SHA256,
    "union_selection_audit_sha256": (
      REVIEWED_ROUND2_UNION_SELECTION_AUDIT_SHA256
    ),
    "round2_merge_audit_sha256": REVIEWED_ROUND2_MERGE_AUDIT_SHA256,
    "round3_smoke_audit_sha256": REVIEWED_ROUND3_SMOKE_AUDIT_SHA256,
    "config_sha256": release["config_sha256"],
    "protocol_sha256": release["protocol_sha256"],
    "protocol_binding": release["protocol_binding"],
    "manifest_sha256": release["manifest_sha256"],
    "original_merged_csv_sha256": release["original_merged_csv_sha256"],
    "round1_merged_csv_sha256": release["round1_merged_csv_sha256"],
    "round2_merged_csv_sha256": release["round2_merged_csv_sha256"],
    "execution_backend": release["execution_backend"],
    "worker_processes": workers,
    "gpu_used": False,
    "thread_environment": release["thread_environment"],
    "shard_count": len(shards),
    "total_cases": len(cases),
    "total_result_rows": EXPECTED_ROWS,
    "total_operator_updates": EXPECTED_OPERATOR_UPDATES,
    "test_locked_access": "prohibited",
  }
  if release_check_only:
    if resume_verified:
      raise ValueError("round-3 release check cannot use --resume-verified")
    if output_dir.exists():
      raise FileExistsError("round-3 release check requires an absent namespace")
    return {
      "artifact_kind": "validation_hyperparameter_expansion_round3_release_check",
      "schema_version": 1,
      "status": "PASS_ready_for_round3_incremental_shard_execution",
      **common,
      "prior_registration": release["prior_registration"],
      "dataset_shard_counts": release["dataset_shard_counts"],
      "dataset_rows_per_shard": EXPECTED_ROWS_PER_SHARD,
      "incremental_candidate_count": EXPECTED_CANDIDATES,
      "formal_output_namespace": OUTPUT_NAMESPACE,
      "formal_output_created": False,
      "merge_created": False,
      "union_selection_audit_created": False,
      "frozen_hyperparameters_created": False,
    }

  execution_record_path = output_dir / "execution_record.json"
  progress_path = output_dir / "progress.json"
  completion_path = output_dir / "shard_batch_SUCCESS.json"
  if output_dir.exists():
    if not resume_verified:
      raise FileExistsError("round-3 namespace exists; use --resume-verified")
    if not execution_record_path.is_file():
      raise ValueError("round-3 namespace lacks its immutable execution record")
    execution_record = _json(execution_record_path)
    if any(execution_record.get(key) != value for key, value in common.items()):
      raise ValueError("resume identity differs from the existing round-3 execution")
  else:
    output_dir.mkdir(parents=True)
    execution_record = {
      "artifact_kind": "validation_hyperparameter_expansion_round3_execution",
      "schema_version": 1,
      "status": "running",
      **common,
      "python": sys.version,
      "platform": platform.platform(),
      "numpy_version": np.__version__,
      "scipy_version": scipy.__version__,
      "created_utc": datetime.now(timezone.utc).isoformat(),
    }
    atomic_write_json(execution_record_path, execution_record)
  execution_record_sha256 = file_sha256(execution_record_path)

  verified = []
  pending = []
  for shard in shards:
    item = verify_incremental_shard(
      output_dir, shard, candidates,
      execution_commit=release["execution_git_commit"],
      execution_record_sha256=execution_record_sha256,
      plan_sha256=REVIEWED_ROUND3_PLAN_SHA256,
      run_id=contract.run_id,
    )
    (pending if item is None else verified).append(shard if item is None else item)
  if verified and not resume_verified:
    raise FileExistsError("verified round-3 shards exist; use --resume-verified")
  if completion_path.exists():
    completion = _json(completion_path)
    if (
      pending
      or completion.get("status")
      != "complete_pending_round3_incremental_integrity_merge"
      or completion.get("execution_git_commit") != release["execution_git_commit"]
      or completion.get("execution_record_sha256") != execution_record_sha256
      or completion.get("round3_plan_sha256") != REVIEWED_ROUND3_PLAN_SHA256
      or completion.get("round3_smoke_audit_sha256")
      != REVIEWED_ROUND3_SMOKE_AUDIT_SHA256
      or completion.get("shards_completed") != len(shards)
      or completion.get("total_result_rows") != EXPECTED_ROWS
      or completion.get("shard_set_sha256") != _shard_set_sha256(verified)
      or completion.get("merge_created") is not False
      or completion.get("union_selection_audit_created") is not False
      or completion.get("frozen_hyperparameters_created") is not False
    ):
      raise ValueError("existing round-3 completion marker failed verification")
    _write_round3_progress(
      progress_path, status="complete", total=len(shards),
      reused=len(verified), written=0,
    )
    return completion

  _write_round3_progress(
    progress_path, status="running", total=len(shards),
    reused=len(verified), written=0,
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
        release["execution_git_commit"], execution_record_sha256,
        REVIEWED_ROUND3_PLAN_SHA256,
      )
      futures[executor.submit(_write_shard_process, payload)] = shard["shard_id"]
    for future in as_completed(futures):
      failed_shard = futures[future]
      written.append(future.result())
      failed_shard = None
      print(json.dumps({
        "status": "running",
        "shards_total": len(shards),
        "shards_reused_verified": len(verified),
        "shards_written": len(written),
        "shards_completed": len(verified) + len(written),
        "latest_shard": written[-1]["shard_id"],
      }, sort_keys=True), flush=True)
      _write_round3_progress(
        progress_path, status="running", total=len(shards),
        reused=len(verified), written=len(written),
      )
  except BaseException:
    for future in futures:
      future.cancel()
    _write_round3_progress(
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
      output_dir, shard, candidates,
      execution_commit=release["execution_git_commit"],
      execution_record_sha256=execution_record_sha256,
      plan_sha256=REVIEWED_ROUND3_PLAN_SHA256,
      run_id=contract.run_id,
    )
    if item is None:
      raise RuntimeError(f"round-3 shard missing after run: {shard['shard_id']}")
    completed.append(item)
  result = {
    "artifact_kind": "validation_hyperparameter_expansion_round3_shard_batch_completion",
    "schema_version": 1,
    "status": "complete_pending_round3_incremental_integrity_merge",
    **common,
    "execution_record_sha256": execution_record_sha256,
    "shards_total": len(shards),
    "shards_reused_verified": len(verified),
    "shards_written": len(written),
    "shards_completed": len(completed),
    "incremental_candidate_count": EXPECTED_CANDIDATES,
    "total_result_rows": sum(int(item["row_count"]) for item in completed),
    "shard_set_sha256": _shard_set_sha256(completed),
    "merge_created": False,
    "union_selection_audit_created": False,
    "frozen_hyperparameters_created": False,
  }
  atomic_write_json(completion_path, result)
  _write_round3_progress(
    progress_path, status="complete", total=len(shards),
    reused=len(verified), written=len(written),
  )
  return result
