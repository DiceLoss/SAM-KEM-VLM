import json
from pathlib import Path
from types import SimpleNamespace

import sam_kem.sweep_incremental_round3_smoke as smoke_module
from sam_kem.sweep_incremental import _expected_case_rows
from sam_kem.sweep_incremental_round3 import (
  build_round3_shards,
  round3_candidate_map,
)
from sam_kem.sweep_incremental_round3_smoke import fixed_round3_smoke_shards


ROOT = Path(__file__).resolve().parents[1]


def test_reviewed_round3_plan_has_exact_incremental_case_workload():
  plan = json.loads(
    (ROOT / "logs/validation_hyperparameter_expansion_round3_plan.json").read_text(
      encoding="utf-8",
    )
  )
  candidates = round3_candidate_map(plan)
  assert {key: len(value) for key, value in candidates.items()} == {
    ("brainweb", 100_000): 4,
    ("brainweb", 200_000): 4,
    ("brainweb", 500_000): 4,
    ("brainweb", 1_000_000): 3,
    ("hecktor", 100_000): 2,
    ("hecktor", 200_000): 2,
    ("hecktor", 500_000): 1,
    ("hecktor", 1_000_000): 0,
  }


def test_round3_candidate_map_exposes_reviewed_empty_cells_to_shard_writer():
  plan = json.loads(
    (ROOT / "logs/validation_hyperparameter_expansion_round3_plan.json").read_text(
      encoding="utf-8",
    )
  )
  candidates = round3_candidate_map(plan)
  assert candidates[("hecktor", 1_000_000)] == []


def test_expected_case_rows_omits_reviewed_zero_candidate_cells():
  shard = {
    "dataset": "hecktor",
    "case_keys": [
      {"count_level": 500_000, "realization_id": 0},
      {"count_level": 1_000_000, "realization_id": 0},
    ],
  }
  candidates = {
    ("hecktor", 500_000): [{"method": "CT-KEM"}],
    ("hecktor", 1_000_000): [],
  }
  assert _expected_case_rows(shard, candidates) == {(500_000, 0): 1}


def test_round3_shards_have_dataset_specific_row_counts():
  plan = json.loads(
    (ROOT / "logs/validation_hyperparameter_expansion_round3_plan.json").read_text(
      encoding="utf-8",
    )
  )
  candidates = round3_candidate_map(plan)
  cases = []
  for dataset, slice_count in (("brainweb", 696), ("hecktor", 45)):
    for slice_index in range(slice_count):
      for count in (100_000, 200_000, 500_000, 1_000_000):
        for realization in (0, 1, 2):
          cases.append(SimpleNamespace(
            dataset=dataset, subject_id=f"{dataset}-subject",
            slice_id=f"z{slice_index:04d}", count_level=count,
            realization_id=realization,
          ))
  shards = build_round3_shards(cases, candidates)
  assert len(shards) == 741
  assert {item["expected_result_rows"] for item in shards[:696]} == {45}
  assert {item["expected_result_rows"] for item in shards[696:]} == {15}
  assert sum(item["expected_result_rows"] for item in shards) == 31_995


def test_round3_smoke_selection_uses_only_fixed_slice_keys(monkeypatch):
  shards = [
    {"shard_id": "shard_0000", "dataset": "brainweb", "subject_id": "subject04",
     "slice_id": "z038"},
    {"shard_id": "shard_0696", "dataset": "hecktor", "subject_id": "10",
     "slice_id": "z072"},
    {"shard_id": "other", "dataset": "hecktor", "subject_id": "11",
     "slice_id": "z072"},
  ]
  monkeypatch.setattr(smoke_module, "build_round3_shards", lambda cases, candidates: shards)
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
  selected, grouped = fixed_round3_smoke_shards(cases, {})
  assert [item["shard_id"] for item in selected] == ["shard_0000", "shard_0696"]
  assert {key: len(value) for key, value in grouped.items()} == {
    ("brainweb", "subject04", "z038"): 12,
    ("hecktor", "10", "z072"): 12,
  }
