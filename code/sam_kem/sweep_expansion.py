"""Plan a predeclared local-shell expansion for boundary sweep optima."""

from __future__ import annotations

import json
import subprocess
from collections import Counter
from pathlib import Path

from .artifacts import atomic_write_json, file_sha256


REVIEWED_SELECTION_AUDIT_SHA256 = (
  "d2ea068f0dbbb3643512f84637ea450a053a2c2b0b8d233897ea39c2d166af28"
)
EXPECTED_BOUNDARY_AXIS_COUNTS = {
  "bowsher_neighbours": 5,
  "bowsher_radius_upper": 4,
  "iterations": 3,
  "iterations_lower": 2,
  "radius_upper": 17,
  "sigma": 20,
}
PARAMETER_FIELDS = ("sigma", "radius", "neighbours", "iterations", "alpha")
EXPANSION_RULES = {
  "sigma": ("sigma", 0.1, 0.2, "continue_geometric_x2_upper"),
  "radius_upper": ("radius", 3, 4, "next_integer_upper"),
  "bowsher_neighbours": ("neighbours", 12, 16, "continue_step_4_upper"),
  "bowsher_radius_upper": ("radius", 3, 4, "next_integer_upper"),
  "iterations": ("iterations", 120, 140, "continue_step_20_upper"),
  "iterations_lower": ("iterations", 10, 5, "adjacent_declared_lower"),
}
CASE_COUNTS_PER_DATASET_COUNT = {"brainweb": 2_088, "hecktor": 135}


def _json(path: Path) -> dict:
  value = json.loads(path.read_text(encoding="utf-8"))
  if not isinstance(value, dict):
    raise ValueError(f"expected a JSON object: {path}")
  return value


def _clean_commit(root: Path) -> str:
  status = subprocess.run(
    ["git", "status", "--porcelain"], cwd=root, check=True,
    capture_output=True, text=True,
  ).stdout.strip()
  if status:
    raise RuntimeError(f"expansion planning requires a clean worktree: {status.splitlines()}")
  return subprocess.run(
    ["git", "rev-parse", "HEAD"], cwd=root, check=True,
    capture_output=True, text=True,
  ).stdout.strip()


def _parameters(selection: dict) -> dict:
  return {field: selection[field] for field in PARAMETER_FIELDS}


def local_shell_candidates(selection: dict, boundary_axes: list[str]) -> list[dict]:
  """Return every nonempty interaction of immediate outward boundary steps."""
  changes = []
  changed_fields = set()
  for axis in boundary_axes:
    if axis not in EXPANSION_RULES:
      raise ValueError(f"no reviewed expansion rule for boundary axis: {axis}")
    field, expected, expanded, rule = EXPANSION_RULES[axis]
    if selection[field] != expected:
      raise ValueError(
        f"selection does not occupy the expected {axis} boundary: {selection[field]!r}"
      )
    if field in changed_fields:
      raise ValueError(f"multiple boundary axes map to parameter field {field!r}")
    changed_fields.add(field)
    changes.append((axis, field, expanded, rule))
  base = _parameters(selection)
  candidates = []
  for mask in range(1, 1 << len(changes)):
    candidate = dict(base)
    applied_axes = []
    for index, (axis, field, expanded, _) in enumerate(changes):
      if mask & (1 << index):
        candidate[field] = expanded
        applied_axes.append(axis)
    candidates.append({**candidate, "expanded_axes": applied_axes})
  return candidates


def build_incremental_expansion_plan(selection_audit: dict) -> dict:
  """Convert reviewed boundary findings into a minimal interaction-complete shell."""
  boundaries = selection_audit.get("boundary_selections", [])
  if (
    selection_audit.get("status") != "BLOCKED_grid_boundary_expansion_required"
    or selection_audit.get("selection_count") != 48
    or selection_audit.get("boundary_selection_count") != 30
    or selection_audit.get("boundary_axis_counts") != EXPECTED_BOUNDARY_AXIS_COUNTS
    or len(boundaries) != 30
    or selection_audit.get("frozen_hyperparameters_created") is not False
    or selection_audit.get("test_locked_access") != "prohibited"
  ):
    raise ValueError("selection audit does not match the reviewed blocked result")
  groups = []
  candidate_counts: Counter = Counter()
  total_rows = 0
  total_updates = 0
  identities = set()
  for boundary in boundaries:
    selection = boundary["selection"]
    axes = list(boundary["boundary_axes"])
    candidates = local_shell_candidates(selection, axes)
    dataset = selection["dataset"]
    count = int(selection["count_level"])
    method = selection["method"]
    identity = (dataset, count, method)
    if identity in identities:
      raise ValueError(f"duplicate boundary-selection identity: {identity}")
    identities.add(identity)
    case_count = CASE_COUNTS_PER_DATASET_COUNT[dataset]
    result_rows = case_count * len(candidates)
    operator_updates = case_count * sum(int(row["iterations"]) for row in candidates)
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
      "boundary_axes": axes,
      "case_count": case_count,
      "incremental_candidate_count": len(candidates),
      "expected_result_rows": result_rows,
      "expected_operator_updates": operator_updates,
      "incremental_candidates": candidates,
    })
  return {
    "expansion_strategy": "immediate_outward_local_shell_with_all_touched_axis_interactions",
    "selection_rule_after_expansion": (
      "subject_balanced_argmin_over_union_of_reviewed_original_grid_and_incremental_shell"
    ),
    "adaptive_stopping_rule": (
      "repeat_a_predeclared_immediate_outward_shell_if_and_only_if_the_union_optimum_"
      "occupies_an_expanded_boundary"
    ),
    "original_selection_count": 48,
    "boundary_group_count": len(groups),
    "unchanged_interior_selection_count": 48 - len(groups),
    "expansion_rules": {
      axis: {
        "parameter": field, "reviewed_boundary_value": boundary,
        "new_value": expanded, "rule": rule,
      }
      for axis, (field, boundary, expanded, rule) in EXPANSION_RULES.items()
    },
    "incremental_candidate_counts": dict(sorted(candidate_counts.items())),
    "incremental_candidate_count": sum(candidate_counts.values()),
    "expected_incremental_result_rows": total_rows,
    "expected_incremental_operator_updates": total_updates,
    "groups": groups,
  }


def plan_validation_hyperparameter_expansion(
  root: Path, selection_audit_path: Path, output_path: Path,
) -> dict:
  """Write only a deterministic expansion plan; execute no reconstruction."""
  root = root.resolve()
  selection_audit_path = selection_audit_path.resolve()
  output_path = output_path.resolve()
  if output_path.exists():
    raise FileExistsError(f"refusing to overwrite expansion plan: {output_path}")
  planning_commit = _clean_commit(root)
  if file_sha256(selection_audit_path) != REVIEWED_SELECTION_AUDIT_SHA256:
    raise ValueError("selection audit is not the researcher-reviewed artifact")
  selection_audit = _json(selection_audit_path)
  frozen_path = root / "results/dev_trainmax_v1/frozen_hyperparameters.csv"
  if frozen_path.exists():
    raise FileExistsError("frozen hyperparameters already exist")
  plan = build_incremental_expansion_plan(selection_audit)
  result = {
    "artifact_kind": "validation_hyperparameter_incremental_expansion_plan",
    "schema_version": 1,
    "status": "ready_for_researcher_review_no_execution",
    "read_only_inputs": True,
    "planning_git_commit": planning_commit,
    "selection_audit_path": selection_audit_path.relative_to(root).as_posix(),
    "selection_audit_sha256": REVIEWED_SELECTION_AUDIT_SHA256,
    "merged_csv_sha256": selection_audit["merged_csv_sha256"],
    "shard_set_sha256": selection_audit["shard_set_sha256"],
    **plan,
    "formal_expansion_output_created": False,
    "frozen_hyperparameters_created": False,
    "test_locked_access": "prohibited",
  }
  atomic_write_json(output_path, result)
  return result
