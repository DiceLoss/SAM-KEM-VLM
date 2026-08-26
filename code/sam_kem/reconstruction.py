"""Reference MLEM and coefficient-space KEM implementations.

The functions preserve the fixed physical intensity scale. They never perform
per-image normalization inside an iterative update.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
from scipy.sparse import csr_matrix, identity, spmatrix

from .operators import LinearOperator


def _save_set(iterations: int, save_iterations: Iterable[int] | None) -> set[int]:
  saves = {iterations} if save_iterations is None else {int(x) for x in save_iterations}
  if any(x < 0 or x > iterations for x in saves):
    raise ValueError("save_iterations must lie between zero and iterations")
  return saves


def _observation_terms(
  measured: np.ndarray,
  multiplicative: np.ndarray | None,
  additive: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
  y = np.asarray(measured, dtype=np.float64)
  gain = np.ones_like(y) if multiplicative is None else np.asarray(multiplicative, dtype=np.float64)
  background = np.zeros_like(y) if additive is None else np.asarray(additive, dtype=np.float64)
  if gain.shape != y.shape or background.shape != y.shape:
    raise ValueError("measured, multiplicative and additive terms must have identical shapes")
  if np.any(y < 0) or np.any(gain < 0) or np.any(background < 0):
    raise ValueError("Poisson observation terms must be non-negative")
  return y, gain, background


def mlem(
  measured: np.ndarray,
  operator: LinearOperator,
  *,
  iterations: int,
  initial: np.ndarray | None = None,
  multiplicative: np.ndarray | None = None,
  additive: np.ndarray | None = None,
  save_iterations: Iterable[int] | None = None,
  epsilon: float = 1e-8,
) -> dict[int, np.ndarray]:
  """Run Poisson MLEM with a matched forward/adjoint pair."""
  if iterations < 1:
    raise ValueError("iterations must be positive")
  saves = _save_set(iterations, save_iterations)
  y, gain, background = _observation_terms(measured, multiplicative, additive)
  if y.shape != operator.projection_shape:
    raise ValueError(f"sinogram shape {y.shape} != {operator.projection_shape}")
  x = (
    np.ones(operator.image_shape, dtype=np.float64)
    if initial is None
    else np.asarray(initial, dtype=np.float64).copy()
  )
  if x.shape != operator.image_shape or np.any(x < 0):
    raise ValueError("initial image has invalid shape or negative values")
  x = np.maximum(x, epsilon)
  sensitivity = np.maximum(operator.adjoint(gain), epsilon)
  outputs: dict[int, np.ndarray] = {}
  if 0 in saves:
    outputs[0] = x.astype(np.float32)
  for iteration in range(1, iterations + 1):
    expectation = gain * operator.forward(x) + background
    ratio = np.divide(y, np.maximum(expectation, epsilon))
    x *= operator.adjoint(gain * ratio) / sensitivity
    x = np.maximum(x, epsilon)
    if iteration in saves:
      outputs[iteration] = x.astype(np.float32)
  return outputs


def rbf_kernel(feature: np.ndarray, *, sigma: float, radius: int) -> csr_matrix:
  """Build a row-normalized local RBF kernel on a Manhattan neighborhood."""
  values = np.asarray(feature, dtype=np.float64)
  if values.ndim != 2 or sigma <= 0 or radius < 0:
    raise ValueError("feature must be 2-D, sigma > 0 and radius >= 0")
  height, width = values.shape
  rows: list[int] = []
  columns: list[int] = []
  weights: list[float] = []
  scale = 2.0 * sigma * sigma
  for row in range(height):
    for column in range(width):
      source = row * width + column
      for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
          if abs(dy) + abs(dx) > radius:
            continue
          yy, xx = row + dy, column + dx
          if not (0 <= yy < height and 0 <= xx < width):
            continue
          target = yy * width + xx
          difference = values[row, column] - values[yy, xx]
          rows.append(source)
          columns.append(target)
          weights.append(float(np.exp(-(difference * difference) / scale)))
  kernel = csr_matrix((weights, (rows, columns)), shape=(values.size, values.size))
  row_sums = np.asarray(kernel.sum(axis=1)).ravel()
  return csr_matrix(kernel.multiply((1.0 / np.maximum(row_sums, 1e-12))[:, None]))


def normalize_kernel_feature(feature: np.ndarray, *, epsilon: float = 1e-8) -> np.ndarray:
  """Per-prior min-max normalization, allowed only for kernel construction."""
  array = np.asarray(feature, dtype=np.float64)
  low, high = float(array.min()), float(array.max())
  if high - low <= epsilon:
    return np.zeros_like(array)
  return (array - low) / (high - low)


def kem(
  measured: np.ndarray,
  operator: LinearOperator,
  kernel: spmatrix,
  *,
  iterations: int,
  initial_coefficients: np.ndarray | None = None,
  multiplicative: np.ndarray | None = None,
  additive: np.ndarray | None = None,
  save_iterations: Iterable[int] | None = None,
  epsilon: float = 1e-8,
) -> dict[int, np.ndarray]:
  """Run coefficient-space KEM with ``x = K alpha``.

  The update is
  ``alpha <- alpha / (K^T A^T g) * K^T A^T(g * y/(g*A*K*alpha+b))``.
  This is not equivalent to smoothing an MLEM image after each iteration.
  """
  if iterations < 1:
    raise ValueError("iterations must be positive")
  pixel_count = int(np.prod(operator.image_shape))
  K = csr_matrix(kernel, dtype=np.float64)
  if K.shape != (pixel_count, pixel_count):
    raise ValueError(f"kernel shape {K.shape} != {(pixel_count, pixel_count)}")
  y, gain, background = _observation_terms(measured, multiplicative, additive)
  alpha = (
    np.ones(pixel_count, dtype=np.float64)
    if initial_coefficients is None
    else np.asarray(initial_coefficients, dtype=np.float64).reshape(-1).copy()
  )
  if alpha.size != pixel_count or np.any(alpha < 0):
    raise ValueError("initial coefficients have invalid shape or negative values")
  alpha = np.maximum(alpha, epsilon)
  sensitivity_image = operator.adjoint(gain).reshape(-1)
  sensitivity = np.maximum(K.T @ sensitivity_image, epsilon)
  saves = _save_set(iterations, save_iterations)
  outputs: dict[int, np.ndarray] = {}

  def image_from_coefficients() -> np.ndarray:
    return np.asarray(K @ alpha).reshape(operator.image_shape)

  if 0 in saves:
    outputs[0] = image_from_coefficients().astype(np.float32)
  for iteration in range(1, iterations + 1):
    image = image_from_coefficients()
    expectation = gain * operator.forward(image) + background
    ratio = np.divide(y, np.maximum(expectation, epsilon))
    coefficient_backprojection = K.T @ operator.adjoint(gain * ratio).reshape(-1)
    alpha *= np.asarray(coefficient_backprojection).ravel() / sensitivity
    alpha = np.maximum(alpha, epsilon)
    if iteration in saves:
      outputs[iteration] = image_from_coefficients().astype(np.float32)
  return outputs


def identity_kernel(image_shape: tuple[int, int]) -> csr_matrix:
  return identity(int(np.prod(image_shape)), format="csr", dtype=np.float64)


def osem(
  measured: np.ndarray,
  operator: LinearOperator,
  *,
  iterations: int,
  subsets: int,
  initial: np.ndarray | None = None,
  multiplicative: np.ndarray | None = None,
  additive: np.ndarray | None = None,
  save_iterations: Iterable[int] | None = None,
  epsilon: float = 1e-8,
) -> dict[int, np.ndarray]:
  """Ordered-subsets EM using interleaved angular subsets."""
  y, gain, background = _observation_terms(measured, multiplicative, additive)
  if y.ndim != 2 or subsets < 1 or y.shape[0] % subsets:
    raise ValueError("the number of projection angles must be divisible by subsets")
  x = np.ones(operator.image_shape) if initial is None else np.asarray(initial, dtype=np.float64).copy()
  if x.shape != operator.image_shape or np.any(x < 0):
    raise ValueError("initial image has invalid shape or negative values")
  x = np.maximum(x, epsilon)
  saves = _save_set(iterations, save_iterations)
  outputs: dict[int, np.ndarray] = {}
  if 0 in saves:
    outputs[0] = x.astype(np.float32)
  for iteration in range(1, iterations + 1):
    for subset in range(subsets):
      selector = np.zeros_like(y)
      selector[subset::subsets] = 1.0
      subset_gain = gain * selector
      sensitivity = np.maximum(operator.adjoint(subset_gain), epsilon)
      expectation = gain * operator.forward(x) + background
      ratio = np.divide(y, np.maximum(expectation, epsilon)) * selector
      x *= operator.adjoint(gain * ratio) / sensitivity
      x = np.maximum(x, epsilon)
    if iteration in saves:
      outputs[iteration] = x.astype(np.float32)
  return outputs


def bowsher_kernel(feature: np.ndarray, *, neighbours: int, radius: int) -> csr_matrix:
  """Build a Bowsher-neighborhood coefficient kernel.

  This is an anatomy-selected KEM basis, not the classic quadratic/relative-
  difference MAP-Bowsher penalty.  Results must therefore be reported as
  ``Bowsher-KEM`` rather than as an unqualified ``Bowsher`` reconstruction.
  """
  values = np.asarray(feature, dtype=np.float64)
  if values.ndim != 2 or neighbours < 1 or radius < 1:
    raise ValueError("feature must be 2-D; neighbours and radius must be positive")
  height, width = values.shape
  rows: list[int] = []
  columns: list[int] = []
  weights: list[float] = []
  for row in range(height):
    for column in range(width):
      source = row * width + column
      candidates: list[tuple[float, int]] = []
      for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
          yy, xx = row + dy, column + dx
          if not (0 <= yy < height and 0 <= xx < width) or (dy == 0 and dx == 0):
            continue
          target = yy * width + xx
          candidates.append((abs(values[row, column] - values[yy, xx]), target))
      chosen = sorted(candidates)[:neighbours]
      rows.append(source); columns.append(source); weights.append(1.0)
      for _, target in chosen:
        rows.append(source); columns.append(target); weights.append(1.0)
  kernel = csr_matrix((weights, (rows, columns)), shape=(values.size, values.size))
  sums = np.asarray(kernel.sum(axis=1)).ravel()
  return csr_matrix(kernel.multiply((1.0 / sums)[:, None]))
