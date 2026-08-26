import numpy as np
import json

from sam_kem.noise import array_sha256
from sam_kem.normalization import (
  apply_candidate, build_normalization_activity_manifest, fit_candidates,
)


def test_scale_fitting_uses_training_data_only():
  candidates = fit_candidates(
    [np.array([[0.0, 2.0], [4.0, 8.0]])], configured_fixed_C=None,
  )
  fixed = next(item for item in candidates if item.name == "train_max_fixed_C")
  assert fixed.scale == 8.0 and fixed.offset == 0.0 and not fixed.clip
  transformed, below, above = apply_candidate(np.array([[16.0]]), fixed)
  assert transformed.item() == 2.0 and below == 0.0 and above == 1.0


def test_global_and_per_image_minmax_are_explicitly_distinct():
  candidates = fit_candidates(
    [np.array([[2.0, 6.0]])], configured_fixed_C=10.0,
  )
  global_mm = next(item for item in candidates if item.name == "train_global_minmax")
  per_image = next(item for item in candidates if item.name == "per_image_minmax")
  global_value, _, _ = apply_candidate(np.array([[4.0, 10.0]]), global_mm)
  local_value, _, _ = apply_candidate(np.array([[4.0, 10.0]]), per_image)
  assert np.allclose(global_value, [[0.5, 1.0]])
  assert np.allclose(local_value, [[0.0, 1.0]])
  assert per_image.paper_role == "diagnostic_only_never_primary"


def test_normalization_manifest_is_built_only_from_verified_development_activity(tmp_path):
  activity_dir = tmp_path / "arrays/brainweb/s1/z001"
  activity_dir.mkdir(parents=True)
  value = np.ones((128, 128), dtype=np.float32)
  activity_path = activity_dir / "raw_activity.npy"
  np.save(activity_path, value, allow_pickle=False)
  activity_path.with_suffix(".json").write_text(json.dumps({
    "schema_version": 1, "artifact_kind": "canonical_activity",
    "dataset": "brainweb", "subject_id": "s1", "slice_id": "z001",
    "split": "train", "activity_unit": "simulation_activity",
    "activity_sha256": array_sha256(value),
    "support_policy": "complete_square_no_content_crop_or_circular_mask",
  }), encoding="utf-8")
  output = tmp_path / "normalization.csv"
  build_normalization_activity_manifest(tmp_path, tmp_path / "arrays", output)
  assert output.is_file()
  assert output.with_suffix(".csv.manifest.json").is_file()
