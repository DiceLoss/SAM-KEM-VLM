import json
from pathlib import Path

from sam_kem.sweep_expansion_round6 import (
  build_round6_expansion_plan,
  round6_local_shell_candidates,
)

ROOT = Path(__file__).resolve().parents[1]


def test_round6_local_shell_continues_each_reviewed_axis():
  assert round6_local_shell_candidates({
    "sigma": 0.05, "radius": 8, "neighbours": 0,
    "iterations": 40, "alpha": 0.0,
  }, ["radius_upper"]) == [{
    "sigma": 0.05, "radius": 9, "neighbours": 0,
    "iterations": 40, "alpha": 0.0,
    "expanded_axes": ["radius_upper"],
  }]
  assert round6_local_shell_candidates({
    "sigma": 0.05, "radius": 6, "neighbours": 0,
    "iterations": 220, "alpha": 0.0,
  }, ["iterations"]) == [{
    "sigma": 0.05, "radius": 6, "neighbours": 0,
    "iterations": 240, "alpha": 0.0,
    "expanded_axes": ["iterations"],
  }]


def test_round6_plan_matches_reviewed_union_boundary_evidence():
  audit = json.loads(
    (ROOT / "logs/validation_hyperparameter_round5_union_selection_audit.json")
    .read_text(encoding="utf-8")
  )
  plan = build_round6_expansion_plan(audit)
  assert plan["source_expanded_boundary_group_count"] == 3
  assert plan["unchanged_union_selection_count"] == 45
  assert plan["incremental_candidate_counts"] == {"brainweb": 3}
  assert plan["incremental_candidate_count"] == 3
  assert plan["expected_incremental_result_rows"] == 6_264
  assert plan["expected_incremental_operator_updates"] == 709_920
  assert plan["cumulative_incremental_candidate_count"] == 154
  assert plan["cumulative_incremental_result_rows"] == 198_513
  assert plan["expected_next_union_result_rows"] == 5_355_873

  groups = {
    (item["count_level"], item["method"]): item
    for item in plan["groups"]
  }
  assert set(groups) == {
    (100_000, "MR-KEM"),
    (200_000, "MR-KEM"),
    (1_000_000, "MR-KEM"),
  }
  assert groups[(100_000, "MR-KEM")]["incremental_candidates"][0]["radius"] == 9
  assert groups[(200_000, "MR-KEM")]["incremental_candidates"][0]["radius"] == 9
  assert (
    groups[(1_000_000, "MR-KEM")]["incremental_candidates"][0]["iterations"]
    == 240
  )
