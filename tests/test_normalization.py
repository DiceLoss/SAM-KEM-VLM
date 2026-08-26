import numpy as np

from sam_kem.normalization import apply_candidate, fit_candidates


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
