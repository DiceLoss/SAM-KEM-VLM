import numpy as np

import json
import pytest

from sam_kem.noise import simulate_low_count, write_low_count_simulation
from sam_kem.reproducibility import CaseKey


def test_count_simulation_is_reproducible_and_case_independent():
  projection = np.arange(1, 17, dtype=float).reshape(4, 4)
  first = CaseKey("brainweb", "s1", "z1", 1000, 0)
  second = CaseKey("brainweb", "s2", "z1", 1000, 0)
  y1, seed1, scale1 = simulate_low_count(projection, expected_true_counts=1000, case=first, experiment_seed=9)
  y1_repeat, seed1_repeat, scale1_repeat = simulate_low_count(projection, expected_true_counts=1000, case=first, experiment_seed=9)
  y2, seed2, _ = simulate_low_count(projection, expected_true_counts=1000, case=second, experiment_seed=9)
  np.testing.assert_array_equal(y1, y1_repeat)
  assert seed1 == seed1_repeat and scale1 == scale1_repeat == 1000 / projection.sum()
  assert seed1 != seed2 and not np.array_equal(y1, y2)


def test_total_counts_follow_poisson_mean_and_variance():
  projection = np.arange(1, 65, dtype=float).reshape(8, 8)
  totals = []
  for realization in range(1000):
    key = CaseKey("brainweb", "s1", "z1", 2000, realization)
    measured, _, _ = simulate_low_count(
      projection, expected_true_counts=2000, case=key, experiment_seed=17,
    )
    totals.append(int(measured.sum()))
  assert abs(np.mean(totals) - 2000) < 8
  assert abs(np.var(totals, ddof=1) - 2000) < 220


def test_simulation_sidecar_binds_identity_and_array_hashes(tmp_path):
  projection_path = tmp_path / "projection.npy"
  output_path = tmp_path / "sinogram.npy"
  np.save(projection_path, np.ones((4, 4), dtype=np.float64))
  key = CaseKey("hecktor", "7", "z2", 100000, 2)
  write_low_count_simulation(
    projection_path, output_path, expected_true_counts=100000,
    case=key, experiment_seed=23,
  )
  metadata = json.loads(output_path.with_suffix(".json").read_text())
  assert metadata["observation_model"] == "independent_poisson_scaled_noise_free_projection"
  assert metadata["dataset"] == "hecktor" and metadata["realization_id"] == 2
  assert metadata["expected_total_counts"] == 100000
  assert metadata["realized_total_counts"] == int(np.load(output_path).sum())


def test_count_argument_must_match_case_identity():
  with pytest.raises(ValueError, match="differs"):
    simulate_low_count(
      np.ones((2, 2)), expected_true_counts=2000,
      case=CaseKey("brainweb", "s1", "z1", 1000, 0), experiment_seed=1,
    )
