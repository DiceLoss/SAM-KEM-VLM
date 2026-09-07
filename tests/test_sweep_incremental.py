import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from sam_kem.sweep_evaluation import evaluate_incremental_validation_case
from sam_kem.sweep_incremental import (
  build_incremental_shards,
  incremental_candidate_map,
)
from sam_kem.sweep_incremental_smoke import fixed_incremental_smoke_shards


def test_reviewed_plan_has_exact_incremental_case_workload():
  plan = json.loads(Path(
    "logs/validation_hyperparameter_expansion_plan.json"
  ).read_text(encoding="utf-8"))
  candidates = incremental_candidate_map(plan)
  assert {
    key: len(value) for key, value in candidates.items()
  } == {
    ("brainweb", 100_000): 8,
    ("brainweb", 200_000): 8,
    ("brainweb", 500_000): 6,
    ("brainweb", 1_000_000): 14,
    ("hecktor", 100_000): 14,
    ("hecktor", 200_000): 10,
    ("hecktor", 500_000): 9,
    ("hecktor", 1_000_000): 7,
  }


def test_incremental_shards_have_dataset_specific_row_counts():
  plan = json.loads(Path(
    "logs/validation_hyperparameter_expansion_plan.json"
  ).read_text(encoding="utf-8"))
  candidates = incremental_candidate_map(plan)
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
  shards = build_incremental_shards(cases, candidates)
  assert len(shards) == 741
  assert sum(item["expected_result_rows"] for item in shards) == 80_568


def test_incremental_evaluator_runs_only_declared_candidates(tmp_path, monkeypatch):
  array = np.ones((2, 2), dtype=np.float64)
  for name in ("measured", "target", "anatomy", "prior"):
    np.save(tmp_path / f"{name}.npy", array)
  case = SimpleNamespace(
    sinogram_path=tmp_path / "measured.npy", target_path=tmp_path / "target.npy",
    anatomy_path=tmp_path / "anatomy.npy", sam_prior_path=tmp_path / "prior.npy",
    mask_path=None, projection_scale=1.0, dataset="brainweb", subject_id="s",
    slice_id="z", realization_id=0, count_level=100_000,
  )
  import sam_kem.sweep_evaluation as module

  monkeypatch.setattr(module, "mlem", lambda *args, **kwargs: {10: array})
  monkeypatch.setattr(module, "normalize_kernel_feature", lambda value: value)
  monkeypatch.setattr(module, "rbf_kernel", lambda *args, **kwargs: "kernel")
  monkeypatch.setattr(module, "bowsher_kernel", lambda *args, **kwargs: "bowsher")
  monkeypatch.setattr(module, "kem", lambda *args, save_iterations, **kwargs: {
    value: array * value / 10 for value in save_iterations
  })
  monkeypatch.setattr(module, "nrmse", lambda reference, image, support: float(image.mean()))
  monkeypatch.setattr(
    module, "ssim", lambda reference, image, data_range, mask: 1.0 - float(image.mean()),
  )
  candidates = [
    {"method": "SelfKEM", "sigma": 0.2, "radius": 3, "neighbours": 0,
     "iterations": 20, "alpha": 0.0},
    {"method": "SelfKEM", "sigma": 0.2, "radius": 3, "neighbours": 0,
     "iterations": 140, "alpha": 0.0},
  ]
  records = evaluate_incremental_validation_case(
    case, object(), np.ones((2, 2), dtype=bool), {
      "normalization": {"expected_target_range": [0.0, 1.0]},
      "reconstruction": {"prompt_mlem_iterations": 10},
      "evaluation": {"nonnegative_tolerance": 1e-7},
    }, "run", candidates,
  )
  assert len(records) == 2
  assert [row["iterations"] for row in records] == [20, 140]
  assert all(row["method"] == "SelfKEM" for row in records)


def test_smoke_selection_uses_only_predeclared_slice_keys(monkeypatch):
  shards = [
    {"shard_id": "shard_0000", "dataset": "brainweb", "subject_id": "subject04",
     "slice_id": "z038"},
    {"shard_id": "shard_0696", "dataset": "hecktor", "subject_id": "10",
     "slice_id": "z072"},
    {"shard_id": "other", "dataset": "hecktor", "subject_id": "11",
     "slice_id": "z072"},
  ]
  import sam_kem.sweep_incremental_smoke as module

  monkeypatch.setattr(module, "build_incremental_shards", lambda cases, candidates: shards)
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
  selected, grouped = fixed_incremental_smoke_shards(cases, {})
  assert [item["shard_id"] for item in selected] == ["shard_0000", "shard_0696"]
  assert {key: len(value) for key, value in grouped.items()} == {
    ("brainweb", "subject04", "z038"): 12,
    ("hecktor", "10", "z072"): 12,
  }
