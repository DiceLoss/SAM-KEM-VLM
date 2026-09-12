import json
from pathlib import Path

from sam_kem.sweep_expansion_round5 import (
  build_round5_expansion_plan,
  round5_local_shell_candidates,
)

ROOT = Path(__file__).resolve().parents[1]


def test_round5_local_shell_continues_each_reviewed_axis():
  assert round5_local_shell_candidates({
    "sigma": 0.0, "radius": 5, "neighbours": 28,
    "iterations": 40, "alpha": 0.0,
  }, ["bowsher_neighbours"]) == [{
    "sigma": 0.0, "radius": 5, "neighbours": 32,
    "iterations": 40, "alpha": 0.0,
    "expanded_axes": ["bowsher_neighbours"],
  }]
  assert round5_local_shell_candidates({
    "sigma": 0.05, "radius": 7, "neighbours": 0,
    "iterations": 60, "alpha": 0.0,
  }, ["radius_upper"]) == [{
    "sigma": 0.05, "radius": 8, "neighbours": 0,
    "iterations": 60, "alpha": 0.0,
    "expanded_axes": ["radius_upper"],
  }]
  assert round5_local_shell_candidates({
    "sigma": 0.05, "radius": 6, "neighbours": 0,
    "iterations": 200, "alpha": 0.0,
  }, ["iterations"]) == [{
    "sigma": 0.05, "radius": 6, "neighbours": 0,
    "iterations": 220, "alpha": 0.0,
    "expanded_axes": ["iterations"],
  }]


def test_round5_plan_matches_reviewed_union_boundary_evidence():
  audit = json.loads(
    (ROOT / "logs/validation_hyperparameter_round4_union_selection_audit.json")
    .read_text(encoding="utf-8")
  )
  plan = build_round5_expansion_plan(audit)
  assert plan["source_expanded_boundary_group_count"] == 6
  assert plan["unchanged_union_selection_count"] == 42
  assert plan["incremental_candidate_counts"] == {"brainweb": 6}
  assert plan["incremental_candidate_count"] == 6
  assert plan["expected_incremental_result_rows"] == 12_528
  assert plan["expected_incremental_operator_updates"] == 1_085_760
  assert plan["cumulative_incremental_candidate_count"] == 151
  assert plan["cumulative_incremental_result_rows"] == 192_249
  assert plan["expected_next_union_result_rows"] == 5_349_609
