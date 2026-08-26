"""Canonical idealized parallel-beam PET system matrix.

This is the sole forward/adjoint implementation used by the clean pipeline.
It models independent 2-D ring positions on a circular field of view. It is an
idealized count experiment, not a scanner model: attenuation, scatter, randoms,
normalization, PSF, TOF, and dead time are intentionally absent.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from scipy.sparse import coo_matrix, csr_matrix


def system_matrix(size: int, angles: int) -> csr_matrix:
  """Build a mass-preserving pixel-driven matrix on an inscribed circle."""
  if size < 2 or angles < 1:
    raise ValueError("size must be >=2 and angles must be >=1")
  centre = (size - 1) / 2.0
  coordinates = np.arange(size, dtype=np.float64) - centre
  xx, yy = np.meshgrid(coordinates, coordinates, indexing="xy")
  pixels = np.arange(size * size, dtype=np.int64).reshape(size, size)
  circular = xx * xx + yy * yy <= centre * centre
  rows: list[np.ndarray] = []
  columns: list[np.ndarray] = []
  weights: list[np.ndarray] = []
  for angle_index, theta in enumerate(np.linspace(0.0, np.pi, angles, endpoint=False)):
    coordinate = xx * np.cos(theta) + yy * np.sin(theta) + centre
    lower = np.floor(coordinate).astype(np.int64)
    upper_weight = coordinate - lower
    for detector, weight in ((lower, 1.0 - upper_weight), (lower + 1, upper_weight)):
      valid = circular & (detector >= 0) & (detector < size) & (weight > 0)
      rows.append(angle_index * size + detector[valid])
      columns.append(pixels[valid])
      weights.append(weight[valid])
  matrix = coo_matrix(
    (np.concatenate(weights), (np.concatenate(rows), np.concatenate(columns))),
    shape=(angles * size, size * size), dtype=np.float64,
  ).tocsr()
  matrix.sum_duplicates()
  return matrix


def build_pair(
  size: int = 128, angles: int = 180,
) -> tuple[Callable[[np.ndarray], np.ndarray], Callable[[np.ndarray], np.ndarray]]:
  matrix = system_matrix(size, angles)

  def forward(image: np.ndarray) -> np.ndarray:
    value = np.asarray(image, dtype=np.float64)
    if value.shape != (size, size):
      raise ValueError(f"expected image shape {(size, size)}, got {value.shape}")
    return (matrix @ value.reshape(-1)).reshape(angles, size)

  def adjoint(sinogram: np.ndarray) -> np.ndarray:
    value = np.asarray(sinogram, dtype=np.float64)
    if value.shape != (angles, size):
      raise ValueError(f"expected sinogram shape {(angles, size)}, got {value.shape}")
    return (matrix.T @ value.reshape(-1)).reshape(size, size)

  return forward, adjoint
