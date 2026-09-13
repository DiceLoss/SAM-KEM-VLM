import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import sam_kem.sweep_incremental_round6_smoke as smoke_module
from sam_kem.sweep_incremental_round6 import (
  build_round6_shards,
  release_check_round6_incremental_validation_sweep,
  round6_candidate_map,
)
from sam_kem.sweep_incremental_round6_smoke import (
  _metric_ranges,
  fixed_round6_smoke_shards,
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


def test_round6_smoke_selection_includes_fixed_empty_domain_shard(monkeypatch):
  shards = [
    {"shard_id": "shard_0000", "dataset": "brainweb", "subject_id": "subject04",
     "slice_id": "z038"},
    {"shard_id": "shard_0696", "dataset": "hecktor", "subject_id": "10",
     "slice_id": "z072"},
    {"shard_id": "other", "dataset": "hecktor", "subject_id": "11",
     "slice_id": "z072"},
  ]
  monkeypatch.setattr(smoke_module, "build_round6_shards", lambda cases, candidates: shards)
  cases = []
  for dataset, subject, slice_id in (
    ("brainweb", "subject04", "z038"), ("hecktor", "10", "z072"),
  ):
    for count in (100_000, 200_000, 500_000, 1_000_000):
      for realization in (0, 1, 2):
        cases.append(SimpleNamespace(
          dataset=dataset, subject_id=subject, slice_id=slice_id,
          count_level=count, realization_id=realization,
        ))
  selected, grouped = fixed_round6_smoke_shards(cases, {})
  assert [item["shard_id"] for item in selected] == ["shard_0000", "shard_0696"]
  assert {key: len(value) for key, value in grouped.items()} == {
    ("brainweb", "subject04", "z038"): 12,
    ("hecktor", "10", "z072"): 12,
  }


def test_round6_metric_ranges_accept_reviewed_empty_domain():
  assert _metric_ranges([]) == {
    "metrics_finite": True,
    "nrmse_min": None,
    "nrmse_max": None,
    "ssim_min": None,
    "ssim_max": None,
  }
  with pytest.raises(ValueError, match="non-finite"):
    _metric_ranges([{"nrmse": "nan", "ssim": "0.9"}])
