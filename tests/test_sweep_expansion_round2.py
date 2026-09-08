import json
from pathlib import Path

from sam_kem.sweep_expansion_round2 import (
  build_round2_expansion_plan,
  round2_local_shell_candidates,
)


ROOT = Path(__file__).resolve().parents[1]


def test_round2_local_shell_contains_each_axis_and_interaction():
  selection = {
    "sigma": 0.2, "radius": 4, "neighbours": 0,
    "iterations": 20, "alpha": 0.0,
  }
  assert round2_local_shell_candidates(
    selection, ["sigma", "radius_upper"],
  ) == [
    {
      "sigma": 0.4, "radius": 4, "neighbours": 0,
      "iterations": 20, "alpha": 0.0, "expanded_axes": ["sigma"],
    },
    {
      "sigma": 0.2, "radius": 5, "neighbours": 0,
      "iterations": 20, "alpha": 0.0, "expanded_axes": ["radius_upper"],
    },
    {
      "sigma": 0.4, "radius": 5, "neighbours": 0,
      "iterations": 20, "alpha": 0.0,
      "expanded_axes": ["sigma", "radius_upper"],
    },
  ]


def test_round2_plan_matches_reviewed_union_boundary_evidence():
  audit = json.loads(
    (ROOT / "logs/validation_hyperparameter_union_selection_audit.json").read_text(
      encoding="utf-8",
    )
  )
  plan = build_round2_expansion_plan(audit)
  assert plan["source_expanded_boundary_group_count"] == 25
  assert plan["unchanged_union_selection_count"] == 23
  assert plan["incremental_candidate_counts"] == {"brainweb": 22, "hecktor": 17}
  assert plan["incremental_candidate_count"] == 39
  assert plan["expected_incremental_result_rows"] == 48_231
  assert plan["expected_incremental_operator_updates"] == 3_949_830
