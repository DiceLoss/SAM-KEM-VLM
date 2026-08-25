from pathlib import Path

import numpy as np

from sam_kem.operator_factory import build_discrete_radon
from sam_kem.operators import relative_adjoint_error


ROOT = Path(__file__).resolve().parents[1]


def _operator(size=24, angles=30):
  return build_discrete_radon(repo_root=ROOT, size=size, angles=angles)


def test_parallel_beam_operator_is_strictly_adjoint():
  assert relative_adjoint_error(_operator()) < 1e-12


def test_projection_conserves_circular_fov_mass_at_every_angle():
  size, angles = 24, 30
  centre = (size - 1) / 2.0
  yy, xx = np.indices((size, size), dtype=float)
  image = np.zeros((size, size), dtype=float)
  image[(xx - centre) ** 2 + (yy - centre) ** 2 <= centre ** 2] = 1.0
  projection = _operator(size, angles).forward(image)
  np.testing.assert_allclose(
    projection.sum(axis=1), image.sum(), rtol=1e-13, atol=1e-11,
  )


def test_projection_is_nonnegative_for_nonnegative_activity():
  rng = np.random.default_rng(5)
  size = 24
  centre = (size - 1) / 2.0
  yy, xx = np.indices((size, size), dtype=float)
  image = rng.random((size, size))
  image[(xx - centre) ** 2 + (yy - centre) ** 2 > centre ** 2] = 0.0
  assert np.all(_operator(size, 30).forward(image) >= 0)
