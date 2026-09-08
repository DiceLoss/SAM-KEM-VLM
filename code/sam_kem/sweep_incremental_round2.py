"""Zero-write release gate for the reviewed second incremental shell."""

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
from .sweep_expansion_round2 import EXPECTED_CANDIDATE_COUNTS
from .sweep_incremental import REVIEWED_EXPANSION_PLAN_SHA256
from .sweep_sharded import THREAD_VARIABLES, _clean_commit, line_ending_equivalent_sha256
from .sweep_sharding import group_validation_cases


REVIEWED_ROUND2_PLAN_SHA256 = (
  "a326b070ffefd791f14969d7841b907954b4f041916607fdeba0a5ce227b44ce"
)
REVIEWED_UNION_SELECTION_AUDIT_SHA256 = (
  "854edf296ee368282125d225d93a0746399a0598d9319691e15a1ed062be6b33"
)
REVIEWED_INCREMENTAL_MERGE_AUDIT_SHA256 = (
  "121f9cc1aadefb0ee1c59b2680d2a2f91a9cba325b26ee6ed8a5f45927f60e20"
)
EXPECTED_ORIGINAL_MERGED_CSV_SHA256 = (
  "cbdc5beea03939819a0348f684544bccb36b3bda6f3f477152324e70784966bc"
)
EXPECTED_ROUND1_MERGED_CSV_SHA256 = (
  "f41c731f731f718388648a99120045fc2bd40a68559c0e6332d5bbc707265260"
)
EXPECTED_CANDIDATES = 39
EXPECTED_ROWS = 48_231
EXPECTED_OPERATOR_UPDATES = 3_949_830
OUTPUT_NAMESPACE = "runs/dev_trainmax_v1/hyperparameter_expansion_round2_shards"


def _json(path: Path) -> dict:
  value = json.loads(path.read_text(encoding="utf-8"))
  if not isinstance(value, dict):
    raise TypeError(f"expected a JSON object: {path}")
  return value


def round2_candidate_map(plan: dict) -> dict[tuple[str, int], list[dict]]:
  """Flatten the reviewed round-2 groups by dataset and count level."""
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
        raise ValueError(f"duplicate round-2 candidate: {identity}")
      seen.add(identity)
      result[(dataset, count)].append(candidate)
  counts = Counter()
  for (dataset, _count), items in result.items():
    counts[dataset] += len(items)
  if len(seen) != EXPECTED_CANDIDATES or dict(counts) != EXPECTED_CANDIDATE_COUNTS:
    raise ValueError("round-2 candidate inventory differs from the reviewed plan")
  return dict(result)


def build_round2_shards(cases: list[Case], candidates: dict) -> list[dict]:
  """Attach exact round-2 row counts to all validation slice shards."""
  shards = group_validation_cases(cases)
  for shard in shards:
    shard["expected_result_rows"] = sum(
      len(candidates[(shard["dataset"], int(key["count_level"]))])
      for key in shard["case_keys"]
    )
  counts = Counter(shard["dataset"] for shard in shards)
  row_counts = Counter(
    (shard["dataset"], int(shard["expected_result_rows"])) for shard in shards
  )
  if (
    len(shards) != 741
    or dict(counts) != {"brainweb": 696, "hecktor": 45}
    or dict(row_counts) != {("brainweb", 66): 696, ("hecktor", 51): 45}
    or sum(int(shard["expected_result_rows"]) for shard in shards) != EXPECTED_ROWS
  ):
    raise ValueError("round-2 shard inventory differs from the reviewed validation set")
  return shards


def _require_smoke_before_formal_execution(*, release_check_only: bool) -> None:
  if not release_check_only:
    raise RuntimeError(
      "round-2 formal execution remains locked pending a reviewed fixed-sample smoke audit"
    )


def run_round2_incremental_validation_sweep_shards(
  root: Path, config_path: Path, protocol_path: Path,
  previous_plan_path: Path, union_audit_path: Path,
  incremental_merge_audit_path: Path, round2_plan_path: Path,
  output_dir: Path, *, workers: int, resume_verified: bool,
  release_check_only: bool = False,
) -> dict:
  """Validate round-2 execution identity without writing or running reconstruction."""
  root = root.resolve()
  config_path = config_path.resolve()
  protocol_path = protocol_path.resolve()
  previous_plan_path = previous_plan_path.resolve()
  union_audit_path = union_audit_path.resolve()
  incremental_merge_audit_path = incremental_merge_audit_path.resolve()
  round2_plan_path = round2_plan_path.resolve()
  output_dir = output_dir.resolve()
  if workers != 8:
    raise ValueError("the round-2 release requires exactly 8 worker processes")
  thread_environment = {name: os.environ.get(name) for name in THREAD_VARIABLES}
  if any(value is not None for value in thread_environment.values()):
    raise RuntimeError(
      f"round-2 execution requires unset BLAS thread overrides: {thread_environment}"
    )
  execution_commit = _clean_commit(root)
  if (
    file_sha256(round2_plan_path) != REVIEWED_ROUND2_PLAN_SHA256
    or file_sha256(previous_plan_path) != REVIEWED_EXPANSION_PLAN_SHA256
    or file_sha256(union_audit_path) != REVIEWED_UNION_SELECTION_AUDIT_SHA256
    or file_sha256(incremental_merge_audit_path)
    != REVIEWED_INCREMENTAL_MERGE_AUDIT_SHA256
  ):
    raise ValueError("round-2 inputs are not the researcher-reviewed artifacts")

  previous_plan = _json(previous_plan_path)
  union_audit = _json(union_audit_path)
  incremental_merge_audit = _json(incremental_merge_audit_path)
  round2_plan = _json(round2_plan_path)
  if (
    round2_plan.get("artifact_kind")
    != "validation_hyperparameter_incremental_expansion_round2_plan"
    or round2_plan.get("status") != "ready_for_researcher_review_no_execution"
    or round2_plan.get("expansion_round") != 2
    or round2_plan.get("source_union_selection_audit_sha256")
    != REVIEWED_UNION_SELECTION_AUDIT_SHA256
    or round2_plan.get("previous_expansion_plan_sha256")
    != REVIEWED_EXPANSION_PLAN_SHA256
    or round2_plan.get("incremental_merge_audit_sha256")
    != REVIEWED_INCREMENTAL_MERGE_AUDIT_SHA256
    or round2_plan.get("original_merged_csv_sha256")
    != EXPECTED_ORIGINAL_MERGED_CSV_SHA256
    or round2_plan.get("round1_merged_csv_sha256")
    != EXPECTED_ROUND1_MERGED_CSV_SHA256
    or round2_plan.get("source_expanded_boundary_group_count") != 25
    or round2_plan.get("unchanged_union_selection_count") != 23
    or round2_plan.get("incremental_candidate_counts") != EXPECTED_CANDIDATE_COUNTS
    or round2_plan.get("incremental_candidate_count") != EXPECTED_CANDIDATES
    or round2_plan.get("expected_incremental_result_rows") != EXPECTED_ROWS
    or round2_plan.get("expected_incremental_operator_updates")
    != EXPECTED_OPERATOR_UPDATES
    or round2_plan.get("formal_expansion_output_created") is not False
    or round2_plan.get("union_selection_audit_created") is not False
    or round2_plan.get("frozen_hyperparameters_created") is not False
    or round2_plan.get("test_locked_access") != "prohibited"
    or previous_plan.get("incremental_candidate_count") != 76
    or previous_plan.get("expected_incremental_result_rows") != 80_568
    or union_audit.get("status")
    != "BLOCKED_expanded_boundary_requires_additional_shell"
    or union_audit.get("expanded_boundary_selection_count") != 25
    or union_audit.get("incremental_merge_audit_sha256")
    != REVIEWED_INCREMENTAL_MERGE_AUDIT_SHA256
    or union_audit.get("original_merged_csv_sha256")
    != EXPECTED_ORIGINAL_MERGED_CSV_SHA256
    or union_audit.get("incremental_merged_csv_sha256")
    != EXPECTED_ROUND1_MERGED_CSV_SHA256
    or union_audit.get("frozen_hyperparameters_created") is not False
    or incremental_merge_audit.get("status")
    != "PASS_complete_pending_union_selection_audit"
    or incremental_merge_audit.get("plan_sha256")
    != REVIEWED_EXPANSION_PLAN_SHA256
    or incremental_merge_audit.get("original_merged_csv_sha256")
    != EXPECTED_ORIGINAL_MERGED_CSV_SHA256
    or incremental_merge_audit.get("incremental_merged_csv_sha256")
    != EXPECTED_ROUND1_MERGED_CSV_SHA256
  ):
    raise ValueError("round-2 execution identity is internally inconsistent")

  cfg, contract = load_contract(config_path)
  require_development_execution(cfg)
  registration = validate_registered_validation_prior_configuration(root, config_path, cfg)
  protocol = load_protocol(protocol_path)
  manifest_path = root / str(cfg["outputs"]["validation_manifest"])
  original_merged_path = root / str(cfg["outputs"]["hyperparameter_sweep"])
  round1_merged_path = (
    root / "runs/dev_trainmax_v1/hyperparameter_expansion/"
    "validation_hyperparameter_expansion.csv"
  )
  frozen_path = root / str(cfg["outputs"]["frozen_hyperparameters"])
  if (
    protocol.kind != "hyperparameter_sweep"
    or protocol.split != "validation"
    or file_sha256(config_path) != union_audit.get("config_sha256")
    or not line_ending_equivalent_sha256(
      protocol_path, str(union_audit.get("protocol_sha256")),
    )
    or file_sha256(original_merged_path) != EXPECTED_ORIGINAL_MERGED_CSV_SHA256
    or file_sha256(round1_merged_path) != EXPECTED_ROUND1_MERGED_CSV_SHA256
    or registration != union_audit.get("prior_registration")
    or output_dir.relative_to(root).as_posix() != OUTPUT_NAMESPACE
    or frozen_path.exists()
  ):
    raise ValueError("round-2 runtime inputs differ from the reviewed union chain")

  candidates = round2_candidate_map(round2_plan)
  cases = load_case_manifest(root, manifest_path, split="validation")
  validate_case_contract(cases, contract)
  shards = build_round2_shards(cases, candidates)
  if resume_verified:
    raise ValueError("round-2 release check cannot be combined with --resume-verified")
  if output_dir.exists():
    raise FileExistsError("round-2 release check requires an absent output namespace")
  _require_smoke_before_formal_execution(release_check_only=release_check_only)
  return {
    "artifact_kind": "validation_hyperparameter_expansion_round2_release_check",
    "schema_version": 1,
    "status": "PASS_ready_for_round2_incremental_shard_smoke",
    "execution_git_commit": execution_commit,
    "round2_plan_sha256": REVIEWED_ROUND2_PLAN_SHA256,
    "previous_expansion_plan_sha256": REVIEWED_EXPANSION_PLAN_SHA256,
    "union_selection_audit_sha256": REVIEWED_UNION_SELECTION_AUDIT_SHA256,
    "incremental_merge_audit_sha256": REVIEWED_INCREMENTAL_MERGE_AUDIT_SHA256,
    "config_sha256": file_sha256(config_path),
    "protocol_sha256": file_sha256(protocol_path),
    "protocol_binding": "exact_or_lf_crlf_equivalent_bytes",
    "manifest_sha256": file_sha256(manifest_path),
    "original_merged_csv_sha256": file_sha256(original_merged_path),
    "round1_merged_csv_sha256": file_sha256(round1_merged_path),
    "prior_registration": registration,
    "execution_backend": "scipy_sparse_csr_cpu_float64_multiprocess",
    "worker_processes": workers,
    "gpu_used": False,
    "thread_environment": thread_environment,
    "shard_count": len(shards),
    "dataset_shard_counts": dict(Counter(item["dataset"] for item in shards)),
    "dataset_rows_per_shard": {"brainweb": 66, "hecktor": 51},
    "total_cases": len(cases),
    "incremental_candidate_count": EXPECTED_CANDIDATES,
    "total_result_rows": EXPECTED_ROWS,
    "total_operator_updates": EXPECTED_OPERATOR_UPDATES,
    "formal_output_namespace": OUTPUT_NAMESPACE,
    "formal_output_created": False,
    "smoke_audit_created": False,
    "merge_created": False,
    "union_selection_audit_created": False,
    "frozen_hyperparameters_created": False,
    "test_locked_access": "prohibited",
  }
