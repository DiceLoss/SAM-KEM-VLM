import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from sam_kem.sweep_incremental_round6 import (
  build_round6_shards,
  release_check_round6_incremental_validation_sweep,
  round6_candidate_map,
)

ROOT = Path(__file__).resolve().parents[1]


def _plan() -> dict:
  return json.loads(
    (ROOT / "logs/validation_hyperparameter_expansion_round6_plan.json").read_text(
      encoding="utf-8",
    )
  )


def _synthetic_cases() -> list[SimpleNamespace]:
  cases = []
  for dataset, slice_count in (("brainweb", 696), ("hecktor", 45)):
    for slice_index in range(slice_count):
      for count in (100_000, 200_000, 500_000, 1_000_000):
        for realization in (0, 1, 2):
          cases.append(SimpleNamespace(
            dataset=dataset,
            subject_id=f"{dataset}-subject",
            slice_id=f"z{slice_index:04d}",
            count_level=count,
            realization_id=realization,
          ))
  return cases


def test_reviewed_round6_plan_has_exact_empty_hecktor_domain():
  candidates = round6_candidate_map(_plan())
  assert {key: len(value) for key, value in candidates.items()} == {
    ("brainweb", 100_000): 1,
    ("brainweb", 200_000): 1,
    ("brainweb", 500_000): 0,
    ("brainweb", 1_000_000): 1,
    ("hecktor", 100_000): 0,
    ("hecktor", 200_000): 0,
    ("hecktor", 500_000): 0,
    ("hecktor", 1_000_000): 0,
  }


def test_round6_shards_preserve_all_slices_and_zero_row_hecktor():
  shards = build_round6_shards(_synthetic_cases(), round6_candidate_map(_plan()))
  assert len(shards) == 741
  assert {item["expected_result_rows"] for item in shards[:696]} == {9}
  assert {item["expected_result_rows"] for item in shards[696:]} == {0}
  assert sum(item["expected_result_rows"] for item in shards) == 6_264


def test_round6_release_check_requires_exact_worker_count(tmp_path):
  with pytest.raises(ValueError, match="exactly 8"):
    release_check_round6_incremental_validation_sweep(
      ROOT, tmp_path / "config", tmp_path / "protocol",
      tmp_path / "union", tmp_path / "round5-plan",
      tmp_path / "round5-merge", tmp_path / "round6-plan",
      tmp_path / "output", workers=7, release_check_only=True,
    )
