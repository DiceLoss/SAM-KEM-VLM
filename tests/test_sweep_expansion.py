from collections import Counter

import pytest

from sam_kem.sweep_expansion import (
  EXPANSION_RULES,
  EXPECTED_BOUNDARY_AXIS_COUNTS,
  build_incremental_expansion_plan,
  local_shell_candidates,
)


def _selection(**updates):
  value = {
    "dataset": "brainweb", "count_level": 100_000, "method": "SelfKEM",
    "sigma": 0.1, "radius": 3, "neighbours": 0, "iterations": 20,
    "alpha": 0.0, "validation_subject_mean_nrmse": 0.2,
    "validation_subject_mean_ssim": 0.9,
  }
  value.update(updates)
  return value


def test_local_shell_contains_each_axis_and_interaction():
  candidates = local_shell_candidates(_selection(), ["sigma", "radius_upper"])
  assert candidates == [
    {
      "sigma": 0.2, "radius": 3, "neighbours": 0, "iterations": 20,
      "alpha": 0.0, "expanded_axes": ["sigma"],
    },
    {
      "sigma": 0.1, "radius": 4, "neighbours": 0, "iterations": 20,
      "alpha": 0.0, "expanded_axes": ["radius_upper"],
    },
    {
      "sigma": 0.2, "radius": 4, "neighbours": 0, "iterations": 20,
      "alpha": 0.0, "expanded_axes": ["sigma", "radius_upper"],
    },
  ]


def test_local_shell_rejects_wrong_reviewed_boundary():
  with pytest.raises(ValueError, match="expected sigma boundary"):
    local_shell_candidates(_selection(sigma=0.05), ["sigma"])


def test_expansion_plan_counts_interaction_complete_candidates():
  boundaries = []
  axis_groups = (
    [["bowsher_neighbours", "bowsher_radius_upper"]] * 3
    + [["iterations", "bowsher_neighbours", "bowsher_radius_upper"]]
    + [["sigma", "radius_upper"]] * 2
    + [["sigma"], ["iterations", "sigma"]]
    + [["radius_upper"]] * 3
    + [["iterations", "radius_upper"]]
    + [["sigma"]] * 4
    + [["iterations_lower"], ["bowsher_neighbours"]]
    + [["iterations_lower", "sigma", "radius_upper"]]
    + [["sigma", "radius_upper"]] * 3
    + [["sigma", "radius_upper"]] * 3
    + [["sigma"]]
    + [["sigma", "radius_upper"]] * 4
  )
  assert len(axis_groups) == 30
  assert dict(Counter(axis for group in axis_groups for axis in group)) == (
    EXPECTED_BOUNDARY_AXIS_COUNTS
  )
  for index, group_axes in enumerate(axis_groups):
    selection = _selection(count_level=100_000 + index, method=f"method-{index}")
    for axis in group_axes:
      field, boundary, _, _ = EXPANSION_RULES[axis]
      selection[field] = boundary
    boundaries.append({"selection": selection, "boundary_axes": group_axes})
  audit = {
    "status": "BLOCKED_grid_boundary_expansion_required",
    "selection_count": 48,
    "boundary_selection_count": 30,
    "boundary_axis_counts": EXPECTED_BOUNDARY_AXIS_COUNTS,
    "boundary_selections": boundaries,
    "frozen_hyperparameters_created": False,
    "test_locked_access": "prohibited",
  }
  plan = build_incremental_expansion_plan(audit)
  assert plan["boundary_group_count"] == 30
  assert plan["unchanged_interior_selection_count"] == 18
  assert plan["incremental_candidate_count"] == 76
  assert plan["expected_incremental_result_rows"] > 0
