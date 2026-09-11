import json
from pathlib import Path

from sam_kem.sweep_expansion_round3 import (
  build_round3_expansion_plan,
  round3_local_shell_candidates,
)


ROOT = Path(__file__).resolve().parents[1]


def test_round3_local_shell_contains_each_axis_and_interaction():
  selection = {
    "sigma": 0.4, "radius": 5, "neighbours": 0,
    "iterations": 20, "alpha": 0.0,
  }
  assert round3_local_shell_candidates(
    selection, ["sigma", "radius_upper"],
  ) == [
    {
      "sigma": 0.8, "radius": 5, "neighbours": 0,
      "iterations": 20, "alpha": 0.0, "expanded_axes": ["sigma"],
    },
    {
      "sigma": 0.4, "radius": 6, "neighbours": 0,
      "iterations": 20, "alpha": 0.0, "expanded_axes": ["radius_upper"],
    },
    {
      "sigma": 0.8, "radius": 6, "neighbours": 0,
      "iterations": 20, "alpha": 0.0,
      "expanded_axes": ["sigma", "radius_upper"],
    },
  ]


def test_round3_plan_matches_reviewed_union_boundary_evidence():
  audit = json.loads(
    (ROOT / "logs/validation_hyperparameter_round2_union_selection_audit.json")
    .read_text(encoding="utf-8")
  )
  plan = build_round3_expansion_plan(audit)
  assert plan["source_expanded_boundary_group_count"] == 12
  assert plan["unchanged_union_selection_count"] == 36
  assert plan["incremental_candidate_counts"] == {"brainweb": 15, "hecktor": 5}
  assert plan["incremental_candidate_count"] == 20
  assert plan["expected_incremental_result_rows"] == 31_995
  assert plan["expected_incremental_operator_updates"] == 2_647_080
  assert plan["cumulative_incremental_candidate_count"] == 135
  assert plan["cumulative_incremental_result_rows"] == 160_794
  assert plan["expected_next_union_result_rows"] == 5_318_154
