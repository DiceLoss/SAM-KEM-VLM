import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from sam_kem.sweep_incremental_round2 import (
  build_round2_shards,
  round2_candidate_map,
  validate_reviewed_round2_smoke,
)
from sam_kem.sweep_incremental_round2_smoke import fixed_round2_smoke_shards


ROOT = Path(__file__).resolve().parents[1]


def test_reviewed_round2_plan_has_exact_incremental_case_workload():
  plan = json.loads(
    (ROOT / "logs/validation_hyperparameter_expansion_round2_plan.json").read_text(
      encoding="utf-8",
    )
  )
  candidates = round2_candidate_map(plan)
  assert {key: len(value) for key, value in candidates.items()} == {
    ("brainweb", 100_000): 4,
    ("brainweb", 200_000): 5,
    ("brainweb", 500_000): 5,
    ("brainweb", 1_000_000): 8,
    ("hecktor", 100_000): 5,
    ("hecktor", 200_000): 6,
    ("hecktor", 500_000): 3,
    ("hecktor", 1_000_000): 3,
  }


def test_round2_shards_have_dataset_specific_row_counts():
  plan = json.loads(
    (ROOT / "logs/validation_hyperparameter_expansion_round2_plan.json").read_text(
      encoding="utf-8",
    )
  )
  candidates = round2_candidate_map(plan)
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
  shards = build_round2_shards(cases, candidates)
  assert len(shards) == 741
  assert {item["expected_result_rows"] for item in shards[:696]} == {66}
  assert {item["expected_result_rows"] for item in shards[696:]} == {51}
  assert sum(item["expected_result_rows"] for item in shards) == 48_231


def test_round2_formal_execution_accepts_only_reviewed_smoke():
  path = ROOT / "logs/validation_hyperparameter_expansion_round2_shard_smoke.json"
  smoke = json.loads(path.read_text(encoding="utf-8"))
  validate_reviewed_round2_smoke(smoke)
  smoke["total_rows"] = 116
  with pytest.raises(ValueError, match="not the reviewed PASS artifact"):
    validate_reviewed_round2_smoke(smoke)


def test_round2_smoke_selection_uses_only_fixed_slice_keys(monkeypatch):
  shards = [
    {"shard_id": "shard_0000", "dataset": "brainweb", "subject_id": "subject04",
     "slice_id": "z038"},
    {"shard_id": "shard_0696", "dataset": "hecktor", "subject_id": "10",
     "slice_id": "z072"},
    {"shard_id": "other", "dataset": "hecktor", "subject_id": "11",
     "slice_id": "z072"},
  ]
  import sam_kem.sweep_incremental_round2_smoke as module

  monkeypatch.setattr(module, "build_round2_shards", lambda cases, candidates: shards)
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
  selected, grouped = fixed_round2_smoke_shards(cases, {})
  assert [item["shard_id"] for item in selected] == ["shard_0000", "shard_0696"]
  assert {key: len(value) for key, value in grouped.items()} == {
    ("brainweb", "subject04", "z038"): 12,
    ("hecktor", "10", "z072"): 12,
  }
