"""Plan the second reviewed local shell after the first union remains bounded."""

from __future__ import annotations

import json
import subprocess
from collections import Counter
from pathlib import Path

from .artifacts import atomic_write_json, file_sha256
from .sweep_expansion import CASE_COUNTS_PER_DATASET_COUNT, PARAMETER_FIELDS
from .sweep_incremental import REVIEWED_EXPANSION_PLAN_SHA256
from .sweep_union_selection import REVIEWED_INCREMENTAL_MERGE_AUDIT_SHA256


REVIEWED_UNION_SELECTION_AUDIT_SHA256 = (
  "854edf296ee368282125d225d93a0746399a0598d9319691e15a1ed062be6b33"
)
EXPECTED_EXPANDED_AXIS_COUNTS = {
  "bowsher_neighbours": 5,
  "bowsher_radius_upper": 4,
  "iterations": 2,
  "radius_upper": 12,
  "sigma": 9,
}
ROUND2_EXPANSION_RULES = {
  "sigma": ("sigma", 0.2, 0.4, "continue_geometric_x2_upper"),
  "radius_upper": ("radius", 4, 5, "next_integer_upper"),
  "bowsher_neighbours": ("neighbours", 16, 20, "continue_step_4_upper"),
  "bowsher_radius_upper": ("radius", 4, 5, "next_integer_upper"),
  "iterations": ("iterations", 140, 160, "continue_step_20_upper"),
}
EXPECTED_CANDIDATE_COUNTS = {"brainweb": 22, "hecktor": 17}
EXPECTED_CANDIDATES = 39
EXPECTED_ROWS = 48_231
EXPECTED_OPERATOR_UPDATES = 3_949_830


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
      f"round-2 expansion planning requires a clean worktree: {status.splitlines()}"
    )
  return subprocess.run(
    ["git", "rev-parse", "HEAD"], cwd=root, check=True,
    capture_output=True, text=True,
  ).stdout.strip()


def _parameters(selection: dict) -> dict:
  return {field: selection[field] for field in PARAMETER_FIELDS}


def _candidate_identities(plan: dict) -> set[tuple]:
  return {
    (
      group["dataset"], int(group["count_level"]), group["method"],
      *(candidate[field] for field in PARAMETER_FIELDS),
    )
    for group in plan["groups"]
    for candidate in group["incremental_candidates"]
  }


def round2_local_shell_candidates(selection: dict, axes: list[str]) -> list[dict]:
  """Apply every nonempty interaction of the second immediate outward steps."""
  changes = []
  changed_fields = set()
  for axis in axes:
    if axis not in ROUND2_EXPANSION_RULES:
      raise ValueError(f"no round-2 expansion rule for boundary axis: {axis}")
    field, expected, expanded, rule = ROUND2_EXPANSION_RULES[axis]
    if selection[field] != expected:
      raise ValueError(
        f"selection does not occupy the expected round-2 {axis} boundary: "
        f"{selection[field]!r}"
      )
    if field in changed_fields:
      raise ValueError(f"multiple round-2 axes map to parameter field {field!r}")
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


def build_round2_expansion_plan(union_audit: dict) -> dict:
  """Build the exact second shell from reviewed expanded-boundary winners."""
  boundaries = union_audit.get("expanded_boundary_selections", [])
  if (
    union_audit.get("status")
    != "BLOCKED_expanded_boundary_requires_additional_shell"
    or union_audit.get("selection_count") != 48
    or union_audit.get("reviewed_boundary_group_count") != 30
    or union_audit.get("expanded_boundary_selection_count") != 25
    or union_audit.get("expanded_boundary_axis_counts")
    != EXPECTED_EXPANDED_AXIS_COUNTS
    or union_audit.get("source_boundary_winner_count") != 5
    or union_audit.get("resolved_reviewed_boundary_group_count") != 5
    or union_audit.get("interior_selection_exact_match_count") != 18
    or len(boundaries) != 25
    or union_audit.get("union_csv_created") is not False
    or union_audit.get("frozen_hyperparameters_created") is not False
    or union_audit.get("test_locked_access") != "prohibited"
  ):
    raise ValueError("union audit does not match the reviewed first-shell result")

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
      raise ValueError(f"invalid or duplicate round-2 boundary group: {identity}")
    identities.add(identity)
    candidates = round2_local_shell_candidates(selection, axes)
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
        raise ValueError(f"duplicate round-2 candidate: {candidate_identity}")
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
    raise ValueError("round-2 shell totals differ from the reviewed boundary evidence")
  return {
    "expansion_round": 2,
    "expansion_strategy": (
      "immediate_outward_local_shell_with_all_touched_axis_interactions"
    ),
    "selection_rule_after_expansion": (
      "subject_balanced_argmin_over_union_of_original_grid_round1_and_round2_shells"
    ),
    "adaptive_stopping_rule": (
      "repeat_a_predeclared_immediate_outward_shell_if_and_only_if_the_union_optimum_"
      "occupies_a_round2_expanded_boundary"
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
      for axis, (field, boundary, expanded, rule) in ROUND2_EXPANSION_RULES.items()
    },
    "incremental_candidate_counts": dict(sorted(candidate_counts.items())),
    "incremental_candidate_count": len(candidate_identities),
    "expected_incremental_result_rows": total_rows,
    "expected_incremental_operator_updates": total_updates,
    "cumulative_incremental_candidate_count": 76 + len(candidate_identities),
    "cumulative_incremental_result_rows": 80_568 + total_rows,
    "expected_next_union_result_rows": 5_237_928 + total_rows,
    "groups": groups,
  }


def plan_round2_validation_hyperparameter_expansion(
  root: Path, union_audit_path: Path, previous_plan_path: Path,
  incremental_merge_audit_path: Path, output_path: Path,
) -> dict:
  """Write a SHA-bound second expansion plan and execute no reconstruction."""
  root = root.resolve()
  union_audit_path = union_audit_path.resolve()
  previous_plan_path = previous_plan_path.resolve()
  incremental_merge_audit_path = incremental_merge_audit_path.resolve()
  output_path = output_path.resolve()
  if output_path.exists():
    raise FileExistsError(f"refusing to overwrite round-2 expansion plan: {output_path}")
  planning_commit = _clean_commit(root)
  if (
    file_sha256(union_audit_path) != REVIEWED_UNION_SELECTION_AUDIT_SHA256
    or file_sha256(previous_plan_path) != REVIEWED_EXPANSION_PLAN_SHA256
    or file_sha256(incremental_merge_audit_path)
    != REVIEWED_INCREMENTAL_MERGE_AUDIT_SHA256
  ):
    raise ValueError("round-2 planning inputs are not the reviewed artifacts")
  union_audit = _json(union_audit_path)
  previous_plan = _json(previous_plan_path)
  incremental_merge_audit = _json(incremental_merge_audit_path)
  frozen_path = root / "results/dev_trainmax_v1/frozen_hyperparameters.csv"
  if (
    frozen_path.exists()
    or previous_plan.get("status") != "ready_for_researcher_review_no_execution"
    or previous_plan.get("incremental_candidate_count") != 76
    or incremental_merge_audit.get("status")
    != "PASS_complete_pending_union_selection_audit"
    or union_audit.get("expansion_plan_sha256") != REVIEWED_EXPANSION_PLAN_SHA256
    or union_audit.get("incremental_merge_audit_sha256")
    != REVIEWED_INCREMENTAL_MERGE_AUDIT_SHA256
    or union_audit.get("incremental_merged_csv_sha256")
    != incremental_merge_audit.get("incremental_merged_csv_sha256")
  ):
    raise ValueError("round-2 planning chain is inconsistent")
  plan = build_round2_expansion_plan(union_audit)
  previous_candidates = _candidate_identities(previous_plan)
  round2_candidates = _candidate_identities(plan)
  if (
    len(previous_candidates) != 76
    or len(round2_candidates) != EXPECTED_CANDIDATES
    or previous_candidates & round2_candidates
  ):
    raise ValueError("round-2 candidates overlap or corrupt the first incremental shell")
  result = {
    "artifact_kind": "validation_hyperparameter_incremental_expansion_round2_plan",
    "schema_version": 1,
    "status": "ready_for_researcher_review_no_execution",
    "planning_git_commit": planning_commit,
    "source_union_selection_audit_path": union_audit_path.relative_to(root).as_posix(),
    "source_union_selection_audit_sha256": REVIEWED_UNION_SELECTION_AUDIT_SHA256,
    "previous_expansion_plan_sha256": REVIEWED_EXPANSION_PLAN_SHA256,
    "incremental_merge_audit_sha256": REVIEWED_INCREMENTAL_MERGE_AUDIT_SHA256,
    "original_merged_csv_sha256": union_audit["original_merged_csv_sha256"],
    "round1_merged_csv_sha256": union_audit["incremental_merged_csv_sha256"],
    **plan,
    "read_only_inputs": True,
    "formal_expansion_output_created": False,
    "union_selection_audit_created": False,
    "frozen_hyperparameters_created": False,
    "test_locked_access": "prohibited",
  }
  atomic_write_json(output_path, result)
  return result
