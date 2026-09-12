"""Plan the fifth reviewed local shell after the five-source union remains bounded."""

from __future__ import annotations

import json
import subprocess
from collections import Counter
from pathlib import Path

from .artifacts import atomic_write_json, file_sha256
from .sweep_expansion import CASE_COUNTS_PER_DATASET_COUNT, PARAMETER_FIELDS
from .sweep_expansion_round2 import _candidate_identities
from .sweep_incremental_round4 import REVIEWED_ROUND4_PLAN_SHA256

REVIEWED_ROUND4_UNION_SELECTION_AUDIT_SHA256 = (
  "ed584b813aaa5a0504f13ceb33f4a83f267d72a5e64e956a5b05cba61a53f971"
)
REVIEWED_ROUND4_MERGE_AUDIT_SHA256 = (
  "2012c94053b23a519f1bc8b238f5ed1ccf7f14c22c3ff3bc9ddc2bd6d34b5518"
)
EXPECTED_EXPANDED_AXIS_COUNTS = {
  "bowsher_neighbours": 2,
  "iterations": 1,
  "radius_upper": 3,
}
ROUND5_EXPANSION_RULES = {
  "bowsher_neighbours": ("neighbours", 28, 32, "continue_step_4_upper"),
  "radius_upper": ("radius", 7, 8, "next_integer_upper"),
  "iterations": ("iterations", 200, 220, "continue_step_20_upper"),
}
EXPECTED_CANDIDATE_COUNTS = {"brainweb": 6}
EXPECTED_CANDIDATES = 6
EXPECTED_ROWS = 12_528
EXPECTED_OPERATOR_UPDATES = 1_085_760
EXPECTED_CUMULATIVE_CANDIDATES = 151
EXPECTED_CUMULATIVE_ROWS = 192_249
EXPECTED_NEXT_UNION_ROWS = 5_349_609


def _json(path: Path) -> dict:
  value = json.loads(path.read_text(encoding="utf-8"))
  if not isinstance(value, dict):
    raise TypeError(f"expected a JSON object: {path}")
  return value


def _clean_commit(root: Path) -> str:
  status = subprocess.run(
    ["git", "status", "--porcelain"], cwd=root, check=True,
    capture_output=True, text=True,
  ).stdout.strip()
  if status:
    raise RuntimeError(
      f"round-5 expansion planning requires a clean worktree: {status.splitlines()}"
    )
  return subprocess.run(
    ["git", "rev-parse", "HEAD"], cwd=root, check=True,
    capture_output=True, text=True,
  ).stdout.strip()


def _parameters(selection: dict) -> dict:
  return {field: selection[field] for field in PARAMETER_FIELDS}


def round5_local_shell_candidates(selection: dict, axes: list[str]) -> list[dict]:
  """Apply every nonempty interaction of the fifth immediate outward steps."""
  changes = []
  changed_fields = set()
  for axis in axes:
    if axis not in ROUND5_EXPANSION_RULES:
      raise ValueError(f"no round-5 expansion rule for boundary axis: {axis}")
    field, expected, expanded, rule = ROUND5_EXPANSION_RULES[axis]
    if selection[field] != expected:
      raise ValueError(
        f"selection does not occupy the expected round-5 {axis} boundary: "
        f"{selection[field]!r}"
      )
    if field in changed_fields:
      raise ValueError(f"multiple round-5 axes map to parameter field {field!r}")
    changed_fields.add(field)
    changes.append((axis, field, expanded, rule))
  base = _parameters(selection)
  candidates = []
  for mask in range(1, 1 << len(changes)):
    candidate = dict(base)
    expanded_axes = []
    for index, (axis, field, expanded, _) in enumerate(changes):
      if mask & (1 << index):
        candidate[field] = expanded
        expanded_axes.append(axis)
    candidates.append({**candidate, "expanded_axes": expanded_axes})
  return candidates


def build_round5_expansion_plan(union_audit: dict) -> dict:
  """Build the exact fifth shell from reviewed Round-4 boundary winners."""
  boundaries = union_audit.get("expanded_boundary_selections", [])
  if (
    union_audit.get("status")
    != "BLOCKED_expanded_boundary_requires_additional_shell"
    or union_audit.get("selection_count") != 48
    or union_audit.get("reviewed_boundary_group_count") != 8
    or union_audit.get("unchanged_prior_union_selection_count") != 40
    or union_audit.get("unchanged_prior_union_selection_exact_match_count") != 40
    or union_audit.get("source_boundary_winner_count") != 2
    or union_audit.get("resolved_reviewed_boundary_group_count") != 2
    or union_audit.get("round4_incremental_candidate_selected_count") != 6
    or union_audit.get("expanded_boundary_selection_count") != 6
    or union_audit.get("expanded_boundary_axis_counts")
    != EXPECTED_EXPANDED_AXIS_COUNTS
    or len(boundaries) != 6
    or union_audit.get("union_rows_streamed") != 5_337_081
    or union_audit.get("union_subject_configuration_aggregates") != 116_843
    or union_audit.get("union_csv_created") is not False
    or union_audit.get("frozen_hyperparameters_created") is not False
    or union_audit.get("test_locked_access") != "prohibited"
  ):
    raise ValueError("round-4 union audit does not match the reviewed boundary result")

  selected = {
    (item["dataset"], int(item["count_level"]), item["method"]): item
    for item in union_audit["selected_hyperparameters"]
  }
  groups = []
  candidate_counts: Counter = Counter()
  total_rows = 0
  total_updates = 0
  identities = set()
  candidate_identities = set()
  for boundary in boundaries:
    selection = boundary["selection"]
    axes = list(boundary["expanded_boundary_axes"])
    dataset = selection["dataset"]
    count = int(selection["count_level"])
    method = selection["method"]
    identity = (dataset, count, method)
    if identity in identities or selected.get(identity) != selection:
      raise ValueError(f"invalid or duplicate round-5 boundary group: {identity}")
    identities.add(identity)
    candidates = round5_local_shell_candidates(selection, axes)
    case_count = CASE_COUNTS_PER_DATASET_COUNT[dataset]
    result_rows = case_count * len(candidates)
    operator_updates = case_count * sum(
      int(candidate["iterations"]) for candidate in candidates
    )
    for candidate in candidates:
      candidate_identity = (
        *identity, *(candidate[field] for field in PARAMETER_FIELDS),
      )
      if candidate_identity in candidate_identities:
        raise ValueError(f"duplicate round-5 candidate: {candidate_identity}")
      candidate_identities.add(candidate_identity)
    candidate_counts[dataset] += len(candidates)
    total_rows += result_rows
    total_updates += operator_updates
    groups.append({
      "dataset": dataset,
      "count_level": count,
      "method": method,
      "source_selected_parameters": _parameters(selection),
      "source_validation_subject_mean_nrmse": selection[
        "validation_subject_mean_nrmse"
      ],
      "source_validation_subject_mean_ssim": selection[
        "validation_subject_mean_ssim"
      ],
      "expanded_boundary_axes": axes,
      "case_count": case_count,
      "incremental_candidate_count": len(candidates),
      "expected_result_rows": result_rows,
      "expected_operator_updates": operator_updates,
      "incremental_candidates": candidates,
    })
  if (
    dict(candidate_counts) != EXPECTED_CANDIDATE_COUNTS
    or len(candidate_identities) != EXPECTED_CANDIDATES
    or total_rows != EXPECTED_ROWS
    or total_updates != EXPECTED_OPERATOR_UPDATES
  ):
    raise ValueError("round-5 shell totals differ from reviewed boundary evidence")
  return {
    "expansion_round": 5,
    "expansion_strategy": (
      "immediate_outward_local_shell_with_all_touched_axis_interactions"
    ),
    "selection_rule_after_expansion": (
      "subject_balanced_argmin_over_union_of_original_grid_round1_round2_round3_"
      "round4_and_round5_shells"
    ),
    "adaptive_stopping_rule": (
      "repeat_a_predeclared_immediate_outward_shell_if_and_only_if_the_union_"
      "optimum_occupies_a_round5_expanded_boundary"
    ),
    "source_union_selection_count": 48,
    "source_expanded_boundary_group_count": len(groups),
    "unchanged_union_selection_count": 48 - len(groups),
    "expansion_rules": {
      axis: {
        "parameter": field,
        "reviewed_boundary_value": boundary,
        "new_value": expanded,
        "rule": rule,
      }
      for axis, (field, boundary, expanded, rule) in ROUND5_EXPANSION_RULES.items()
    },
    "incremental_candidate_counts": dict(sorted(candidate_counts.items())),
    "incremental_candidate_count": len(candidate_identities),
    "expected_incremental_result_rows": total_rows,
    "expected_incremental_operator_updates": total_updates,
    "cumulative_incremental_candidate_count": 145 + len(candidate_identities),
    "cumulative_incremental_result_rows": 179_721 + total_rows,
    "expected_next_union_result_rows": 5_337_081 + total_rows,
    "groups": groups,
  }


def plan_round5_validation_hyperparameter_expansion(
  root: Path, union_audit_path: Path, round4_plan_path: Path,
  round4_merge_audit_path: Path, output_path: Path,
) -> dict:
  """Write a SHA-bound fifth expansion plan and execute no reconstruction."""
  root = root.resolve()
  union_audit_path = union_audit_path.resolve()
  round4_plan_path = round4_plan_path.resolve()
  round4_merge_audit_path = round4_merge_audit_path.resolve()
  output_path = output_path.resolve()
  if output_path.exists():
    raise FileExistsError(f"refusing to overwrite round-5 expansion plan: {output_path}")
  planning_commit = _clean_commit(root)
  if (
    file_sha256(union_audit_path)
    != REVIEWED_ROUND4_UNION_SELECTION_AUDIT_SHA256
    or file_sha256(round4_plan_path) != REVIEWED_ROUND4_PLAN_SHA256
    or file_sha256(round4_merge_audit_path) != REVIEWED_ROUND4_MERGE_AUDIT_SHA256
  ):
    raise ValueError("round-5 planning inputs are not the reviewed artifacts")
  union_audit = _json(union_audit_path)
  round4_plan = _json(round4_plan_path)
  round4_merge_audit = _json(round4_merge_audit_path)
  frozen_path = root / "results/dev_trainmax_v1/frozen_hyperparameters.csv"
  if (
    frozen_path.exists()
    or round4_plan.get("status") != "ready_for_researcher_review_no_execution"
    or round4_plan.get("incremental_candidate_count") != 10
    or round4_plan.get("cumulative_incremental_candidate_count") != 145
    or round4_plan.get("cumulative_incremental_result_rows") != 179_721
    or round4_merge_audit.get("status")
    != "PASS_complete_pending_round4_union_selection_audit"
    or union_audit.get("round4_expansion_plan_sha256")
    != REVIEWED_ROUND4_PLAN_SHA256
    or union_audit.get("round4_merge_audit_sha256")
    != REVIEWED_ROUND4_MERGE_AUDIT_SHA256
    or union_audit.get("round4_merged_csv_sha256")
    != round4_merge_audit.get("round4_merged_csv_sha256")
  ):
    raise ValueError("round-5 planning chain is inconsistent")
  plan = build_round5_expansion_plan(union_audit)
  round4_candidates = _candidate_identities(round4_plan)
  round5_candidates = _candidate_identities(plan)
  if (
    len(round4_candidates) != 10
    or len(round5_candidates) != EXPECTED_CANDIDATES
    or round4_candidates & round5_candidates
  ):
    raise ValueError("round-5 candidates overlap or corrupt the fourth shell")
  result = {
    "artifact_kind": "validation_hyperparameter_incremental_expansion_round5_plan",
    "schema_version": 1,
    "status": "ready_for_researcher_review_no_execution",
    "planning_git_commit": planning_commit,
    "source_union_selection_audit_path": union_audit_path.relative_to(root).as_posix(),
    "source_union_selection_audit_sha256": (
      REVIEWED_ROUND4_UNION_SELECTION_AUDIT_SHA256
    ),
    "previous_round4_expansion_plan_sha256": REVIEWED_ROUND4_PLAN_SHA256,
    "round4_incremental_merge_audit_sha256": REVIEWED_ROUND4_MERGE_AUDIT_SHA256,
    "original_merged_csv_sha256": union_audit["original_merged_csv_sha256"],
    "round1_merged_csv_sha256": union_audit["round1_merged_csv_sha256"],
    "round2_merged_csv_sha256": union_audit["round2_merged_csv_sha256"],
    "round3_merged_csv_sha256": union_audit["round3_merged_csv_sha256"],
    "round4_merged_csv_sha256": union_audit["round4_merged_csv_sha256"],
    **plan,
    "read_only_inputs": True,
    "formal_expansion_output_created": False,
    "union_selection_audit_created": False,
    "frozen_hyperparameters_created": False,
    "test_locked_access": "prohibited",
  }
  if (
    result["cumulative_incremental_candidate_count"]
    != EXPECTED_CUMULATIVE_CANDIDATES
    or result["cumulative_incremental_result_rows"] != EXPECTED_CUMULATIVE_ROWS
    or result["expected_next_union_result_rows"] != EXPECTED_NEXT_UNION_ROWS
  ):
    raise ValueError("round-5 cumulative totals differ from the reviewed chain")
  atomic_write_json(output_path, result)
  return result
