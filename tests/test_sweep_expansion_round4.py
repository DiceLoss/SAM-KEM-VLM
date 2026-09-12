import json
from pathlib import Path

from sam_kem.sweep_expansion_round4 import (
  build_round4_expansion_plan,
  round4_local_shell_candidates,
)

ROOT = Path(__file__).resolve().parents[1]


def test_round4_local_shell_contains_each_axis_and_interaction():
  selection = {
    "sigma": 0.05, "radius": 6, "neighbours": 0,
    "iterations": 180, "alpha": 0.0,
  }
  assert round4_local_shell_candidates(
    selection, ["iterations", "radius_upper"],
  ) == [
    {
      "sigma": 0.05, "radius": 6, "neighbours": 0,
      "iterations": 200, "alpha": 0.0, "expanded_axes": ["iterations"],
    },
    {
      "sigma": 0.05, "radius": 7, "neighbours": 0,
      "iterations": 180, "alpha": 0.0, "expanded_axes": ["radius_upper"],
    },
    {
      "sigma": 0.05, "radius": 7, "neighbours": 0,
      "iterations": 200, "alpha": 0.0,
      "expanded_axes": ["iterations", "radius_upper"],
    },
  ]


def test_round4_plan_matches_reviewed_union_boundary_evidence():
  audit = json.loads(
    (ROOT / "logs/validation_hyperparameter_round3_union_selection_audit.json")
    .read_text(encoding="utf-8")
  )
  plan = build_round4_expansion_plan(audit)
  assert plan["source_expanded_boundary_group_count"] == 8
  assert plan["unchanged_union_selection_count"] == 40
  assert plan["incremental_candidate_counts"] == {"brainweb": 9, "hecktor": 1}
  assert plan["incremental_candidate_count"] == 10
  assert plan["expected_incremental_result_rows"] == 18_927
  assert plan["expected_incremental_operator_updates"] == 2_007_180
  assert plan["cumulative_incremental_candidate_count"] == 145
  assert plan["cumulative_incremental_result_rows"] == 179_721
  assert plan["expected_next_union_result_rows"] == 5_337_081
