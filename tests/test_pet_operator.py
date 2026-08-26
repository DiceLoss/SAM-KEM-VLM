from pathlib import Path

import numpy as np

from sam_kem.operator_factory import build_discrete_radon
from sam_kem.operators import relative_adjoint_error


ROOT = Path(__file__).resolve().parents[1]


def _operator(size=24, angles=30):
  return build_discrete_radon(repo_root=ROOT, size=size, angles=angles)


def test_parallel_beam_operator_is_strictly_adjoint():
  assert relative_adjoint_error(_operator()) < 1e-12


def test_projection_conserves_full_square_mass_at_every_angle():
  size, angles = 24, 30
  image = np.ones((size, size), dtype=float)
  projection = _operator(size, angles).forward(image)
  assert projection.shape == (angles, 34)
  np.testing.assert_allclose(
    projection.sum(axis=1), image.sum(), rtol=1e-13, atol=1e-11,
  )


def test_projection_is_nonnegative_for_nonnegative_activity():
  rng = np.random.default_rng(5)
  size = 24
  image = rng.random((size, size))
  assert np.all(_operator(size, 30).forward(image) >= 0)


def test_corner_pixels_are_observed():
  image = np.zeros((24, 24), dtype=float)
  image[0, 0] = 1.0
  projection = _operator(24, 30).forward(image)
  np.testing.assert_allclose(projection.sum(axis=1), 1.0, rtol=0.0, atol=1e-12)
