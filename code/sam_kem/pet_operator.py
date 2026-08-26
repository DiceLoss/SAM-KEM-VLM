"""Canonical idealized parallel-beam PET system matrix.

This is the sole forward/adjoint implementation used by the clean pipeline.
It models independent 2-D ring positions over a complete square image grid. It is an
idealized count experiment, not a scanner model: attenuation, scatter, randoms,
normalization, PSF, TOF, and dead time are intentionally absent.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from scipy.sparse import coo_matrix, csr_matrix


def minimum_detector_bins(size: int) -> int:
  """Return a detector-bin count that covers the complete image diagonal."""
  if size < 2:
    raise ValueError("size must be >=2")
  return int(np.ceil(np.sqrt(2.0) * size))


def system_matrix(size: int, angles: int, detector_bins: int | None = None) -> csr_matrix:
  """Build a mass-preserving pixel-driven matrix for every square-grid pixel."""
  if size < 2 or angles < 1:
    raise ValueError("size must be >=2 and angles must be >=1")
  bins = minimum_detector_bins(size) if detector_bins is None else int(detector_bins)
  if bins < minimum_detector_bins(size):
    raise ValueError(
      f"detector_bins={bins} cannot cover the diagonal of a {size}x{size} image; "
      f"use at least {minimum_detector_bins(size)}"
    )
  centre = (size - 1) / 2.0
  detector_centre = (bins - 1) / 2.0
  coordinates = np.arange(size, dtype=np.float64) - centre
  xx, yy = np.meshgrid(coordinates, coordinates, indexing="xy")
  pixels = np.arange(size * size, dtype=np.int64).reshape(size, size)
  rows: list[np.ndarray] = []
  columns: list[np.ndarray] = []
  weights: list[np.ndarray] = []
  for angle_index, theta in enumerate(np.linspace(0.0, np.pi, angles, endpoint=False)):
    coordinate = xx * np.cos(theta) + yy * np.sin(theta) + detector_centre
    lower = np.floor(coordinate).astype(np.int64)
    upper_weight = coordinate - lower
    for detector, weight in ((lower, 1.0 - upper_weight), (lower + 1, upper_weight)):
      valid = (detector >= 0) & (detector < bins) & (weight > 0)
      rows.append(angle_index * bins + detector[valid])
      columns.append(pixels[valid])
      weights.append(weight[valid])
  matrix = coo_matrix(
    (np.concatenate(weights), (np.concatenate(rows), np.concatenate(columns))),
    shape=(angles * bins, size * size), dtype=np.float64,
  ).tocsr()
  matrix.sum_duplicates()
  return matrix


def build_pair(
  size: int = 128, angles: int = 180, detector_bins: int | None = None,
) -> tuple[Callable[[np.ndarray], np.ndarray], Callable[[np.ndarray], np.ndarray]]:
  bins = minimum_detector_bins(size) if detector_bins is None else int(detector_bins)
  matrix = system_matrix(size, angles, bins)

  def forward(image: np.ndarray) -> np.ndarray:
    value = np.asarray(image, dtype=np.float64)
    if value.shape != (size, size):
      raise ValueError(f"expected image shape {(size, size)}, got {value.shape}")
    return (matrix @ value.reshape(-1)).reshape(angles, bins)

  def adjoint(sinogram: np.ndarray) -> np.ndarray:
    value = np.asarray(sinogram, dtype=np.float64)
    if value.shape != (angles, bins):
      raise ValueError(f"expected sinogram shape {(angles, bins)}, got {value.shape}")
    return (matrix.T @ value.reshape(-1)).reshape(size, size)

  return forward, adjoint
