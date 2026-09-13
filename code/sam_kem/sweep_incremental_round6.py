"""SHA-bound pre-release identity for the reviewed sixth incremental shell."""

from __future__ import annotations

import json
import os
from collections import Counter, defaultdict
from pathlib import Path

from .artifacts import file_sha256
from .config import load_contract, require_development_execution
from .experiments import Case, load_case_manifest, validate_case_contract
from .prior_registration import validate_registered_validation_prior_configuration
from .protocols import load_protocol
from .sweep_expansion_round6 import (
  EXPECTED_CANDIDATE_COUNTS,
  EXPECTED_CANDIDATES,
  EXPECTED_OPERATOR_UPDATES,
  EXPECTED_ROWS,
  REVIEWED_ROUND5_MERGE_AUDIT_SHA256,
  REVIEWED_ROUND5_UNION_SELECTION_AUDIT_SHA256,
)
from .sweep_incremental_round2 import (
  EXPECTED_ORIGINAL_MERGED_CSV_SHA256,
  EXPECTED_ROUND1_MERGED_CSV_SHA256,
)
from .sweep_incremental_round2_merge import ROUND2_MERGED_NAME
from .sweep_incremental_round3 import EXPECTED_ROUND2_MERGED_CSV_SHA256
from .sweep_incremental_round3_merge import ROUND3_MERGED_NAME
from .sweep_incremental_round4 import EXPECTED_ROUND3_MERGED_CSV_SHA256
from .sweep_incremental_round4_merge import ROUND4_MERGED_NAME
from .sweep_incremental_round5 import (
  EXPECTED_ROUND4_MERGED_CSV_SHA256,
  REVIEWED_ROUND5_PLAN_SHA256,
)
from .sweep_incremental_round5_merge import ROUND5_MERGED_NAME
from .sweep_sharded import (
  THREAD_VARIABLES,
  _clean_commit,
  line_ending_equivalent_sha256,
)
from .sweep_sharding import group_validation_cases

REVIEWED_ROUND6_PLAN_SHA256 = (
  "6db4189966d8703fd63aeb31bb329a6b440c74c5796c46013aeb2b7b3d81c287"
)
EXPECTED_ROUND5_MERGED_CSV_SHA256 = (
  "db90380839ebe405ddbb9100aab79af97d81bea57c023d4a1b1786fc10c99df2"
)
OUTPUT_NAMESPACE = "runs/dev_trainmax_v1/hyperparameter_expansion_round6_shards"
FIXED_SMOKE_KEYS = {
  ("brainweb", "subject04", "z038"),
  ("hecktor", "10", "z072"),
}
EXPECTED_ROWS_PER_SHARD = {"brainweb": 9, "hecktor": 0}
EXPECTED_METHOD_ROWS = {
  "brainweb": {"MR-KEM": 9},
  "hecktor": {},
}


def _json(path: Path) -> dict:
  value = json.loads(path.read_text(encoding="utf-8"))
  if not isinstance(value, dict):
    raise TypeError(f"expected a JSON object: {path}")
  return value


def round6_candidate_map(plan: dict) -> dict[tuple[str, int], list[dict]]:
  """Flatten the reviewed Round-6 groups by dataset and count level."""
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
        raise ValueError(f"duplicate round-6 candidate: {identity}")
      seen.add(identity)
      result[(dataset, count)].append(candidate)
  for dataset in ("brainweb", "hecktor"):
    for count in (100_000, 200_000, 500_000, 1_000_000):
      result.setdefault((dataset, count), [])
  counts = Counter()
  for (dataset, _count), items in result.items():
    counts[dataset] += len(items)
  nonzero_counts = {key: value for key, value in counts.items() if value}
  if (
    len(seen) != EXPECTED_CANDIDATES
    or nonzero_counts != EXPECTED_CANDIDATE_COUNTS
  ):
    raise ValueError("round-6 candidate inventory differs from the reviewed plan")
  return dict(result)


def build_round6_shards(cases: list[Case], candidates: dict) -> list[dict]:
  """Attach exact Round-6 row counts, including the empty HE domain."""
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
    or dict(row_counts) != {("brainweb", 9): 696, ("hecktor", 0): 45}
    or sum(int(shard["expected_result_rows"]) for shard in shards) != EXPECTED_ROWS
  ):
    raise ValueError("round-6 shard inventory differs from the reviewed validation set")
  return shards


def release_check_round6_incremental_validation_sweep(
  root: Path, config_path: Path, protocol_path: Path,
  union_audit_path: Path, round5_plan_path: Path,
  round5_merge_audit_path: Path, round6_plan_path: Path,
  output_dir: Path, *, workers: int, release_check_only: bool,
  allow_existing_output: bool = False,
) -> dict:
  """Validate the Round-6 identity without releasing formal execution."""
  root = root.resolve()
  config_path = config_path.resolve()
  protocol_path = protocol_path.resolve()
  union_audit_path = union_audit_path.resolve()
  round5_plan_path = round5_plan_path.resolve()
  round5_merge_audit_path = round5_merge_audit_path.resolve()
  round6_plan_path = round6_plan_path.resolve()
  output_dir = output_dir.resolve()
  if not release_check_only:
    raise RuntimeError("formal round-6 execution awaits a reviewed smoke audit")
  if workers != 8:
    raise ValueError("the round-6 release requires exactly 8 worker processes")
  thread_environment = {name: os.environ.get(name) for name in THREAD_VARIABLES}
  if any(value is not None for value in thread_environment.values()):
    raise RuntimeError(
      f"round-6 execution requires unset BLAS thread overrides: {thread_environment}"
    )
  execution_commit = _clean_commit(root)
  if (
    file_sha256(union_audit_path)
    != REVIEWED_ROUND5_UNION_SELECTION_AUDIT_SHA256
    or file_sha256(round5_plan_path) != REVIEWED_ROUND5_PLAN_SHA256
    or file_sha256(round5_merge_audit_path)
    != REVIEWED_ROUND5_MERGE_AUDIT_SHA256
    or file_sha256(round6_plan_path) != REVIEWED_ROUND6_PLAN_SHA256
  ):
    raise ValueError("round-6 inputs are not the researcher-reviewed artifacts")

  union_audit = _json(union_audit_path)
  round5_plan = _json(round5_plan_path)
  round5_merge_audit = _json(round5_merge_audit_path)
  round6_plan = _json(round6_plan_path)
  if (
    round6_plan.get("artifact_kind")
    != "validation_hyperparameter_incremental_expansion_round6_plan"
    or round6_plan.get("status") != "ready_for_researcher_review_no_execution"
    or round6_plan.get("expansion_round") != 6
    or round6_plan.get("source_union_selection_audit_sha256")
    != REVIEWED_ROUND5_UNION_SELECTION_AUDIT_SHA256
    or round6_plan.get("previous_round5_expansion_plan_sha256")
    != REVIEWED_ROUND5_PLAN_SHA256
    or round6_plan.get("round5_incremental_merge_audit_sha256")
    != REVIEWED_ROUND5_MERGE_AUDIT_SHA256
    or round6_plan.get("original_merged_csv_sha256")
    != EXPECTED_ORIGINAL_MERGED_CSV_SHA256
    or round6_plan.get("round1_merged_csv_sha256")
    != EXPECTED_ROUND1_MERGED_CSV_SHA256
    or round6_plan.get("round2_merged_csv_sha256")
    != EXPECTED_ROUND2_MERGED_CSV_SHA256
    or round6_plan.get("round3_merged_csv_sha256")
    != EXPECTED_ROUND3_MERGED_CSV_SHA256
    or round6_plan.get("round4_merged_csv_sha256")
    != EXPECTED_ROUND4_MERGED_CSV_SHA256
    or round6_plan.get("round5_merged_csv_sha256")
    != EXPECTED_ROUND5_MERGED_CSV_SHA256
    or round6_plan.get("source_expanded_boundary_group_count") != 3
    or round6_plan.get("unchanged_union_selection_count") != 45
    or round6_plan.get("incremental_candidate_counts") != EXPECTED_CANDIDATE_COUNTS
    or round6_plan.get("incremental_candidate_count") != EXPECTED_CANDIDATES
    or round6_plan.get("expected_incremental_result_rows") != EXPECTED_ROWS
    or round6_plan.get("expected_incremental_operator_updates")
    != EXPECTED_OPERATOR_UPDATES
    or round6_plan.get("cumulative_incremental_candidate_count") != 154
    or round6_plan.get("cumulative_incremental_result_rows") != 198_513
    or round6_plan.get("expected_next_union_result_rows") != 5_355_873
    or round6_plan.get("formal_expansion_output_created") is not False
    or round6_plan.get("frozen_hyperparameters_created") is not False
    or round6_plan.get("test_locked_access") != "prohibited"
    or round5_plan.get("incremental_candidate_count") != 6
    or union_audit.get("status")
    != "BLOCKED_expanded_boundary_requires_additional_shell"
    or union_audit.get("expanded_boundary_selection_count") != 3
    or union_audit.get("union_rows_streamed") != 5_349_609
    or union_audit.get("round5_merge_audit_sha256")
    != REVIEWED_ROUND5_MERGE_AUDIT_SHA256
    or union_audit.get("round5_merged_csv_sha256")
    != EXPECTED_ROUND5_MERGED_CSV_SHA256
    or union_audit.get("frozen_hyperparameters_created") is not False
    or round5_merge_audit.get("status")
    != "PASS_complete_pending_round5_union_selection_audit"
    or round5_merge_audit.get("round5_plan_sha256") != REVIEWED_ROUND5_PLAN_SHA256
    or round5_merge_audit.get("round5_merged_csv_sha256")
    != EXPECTED_ROUND5_MERGED_CSV_SHA256
  ):
    raise ValueError("round-6 execution identity is internally inconsistent")

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
  round3_path = root / "runs/dev_trainmax_v1/hyperparameter_expansion_round3" / (
    ROUND3_MERGED_NAME
  )
  round4_path = root / "runs/dev_trainmax_v1/hyperparameter_expansion_round4" / (
    ROUND4_MERGED_NAME
  )
  round5_path = root / "runs/dev_trainmax_v1/hyperparameter_expansion_round5" / (
    ROUND5_MERGED_NAME
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
    or file_sha256(round3_path) != EXPECTED_ROUND3_MERGED_CSV_SHA256
    or file_sha256(round4_path) != EXPECTED_ROUND4_MERGED_CSV_SHA256
    or file_sha256(round5_path) != EXPECTED_ROUND5_MERGED_CSV_SHA256
    or registration != union_audit.get("prior_registration")
    or output_dir.relative_to(root).as_posix() != OUTPUT_NAMESPACE
    or (output_dir.exists() and not allow_existing_output)
    or frozen_path.exists()
  ):
    raise ValueError("round-6 runtime inputs differ from the reviewed union chain")

  candidates = round6_candidate_map(round6_plan)
  cases = load_case_manifest(root, manifest_path, split="validation")
  validate_case_contract(cases, contract)
  shards = build_round6_shards(cases, candidates)
  return {
    "artifact_kind": "validation_hyperparameter_expansion_round6_release_check",
    "schema_version": 1,
    "status": "PASS_ready_for_round6_incremental_shard_smoke",
    "execution_git_commit": execution_commit,
    "round6_plan_sha256": REVIEWED_ROUND6_PLAN_SHA256,
    "round5_plan_sha256": REVIEWED_ROUND5_PLAN_SHA256,
    "union_selection_audit_sha256": (
      REVIEWED_ROUND5_UNION_SELECTION_AUDIT_SHA256
    ),
    "round5_merge_audit_sha256": REVIEWED_ROUND5_MERGE_AUDIT_SHA256,
    "config_sha256": file_sha256(config_path),
    "protocol_sha256": file_sha256(protocol_path),
    "protocol_binding": "exact_or_lf_crlf_equivalent_bytes",
    "manifest_sha256": file_sha256(manifest_path),
    "original_merged_csv_sha256": file_sha256(original_path),
    "round1_merged_csv_sha256": file_sha256(round1_path),
    "round2_merged_csv_sha256": file_sha256(round2_path),
    "round3_merged_csv_sha256": file_sha256(round3_path),
    "round4_merged_csv_sha256": file_sha256(round4_path),
    "round5_merged_csv_sha256": file_sha256(round5_path),
    "execution_backend": "scipy_sparse_csr_cpu_float64_multiprocess",
    "worker_processes": workers,
    "gpu_used": False,
    "thread_environment": thread_environment,
    "shard_count": len(shards),
    "dataset_shard_counts": dict(Counter(item["dataset"] for item in shards)),
    "dataset_rows_per_shard": EXPECTED_ROWS_PER_SHARD,
    "dataset_method_rows_per_shard": EXPECTED_METHOD_ROWS,
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
