"""Metrics evaluated on the declared fixed activity scale."""

from __future__ import annotations

import numpy as np


def _masked(reference: np.ndarray, estimate: np.ndarray, mask: np.ndarray | None) -> tuple[np.ndarray, np.ndarray]:
  target = np.asarray(reference, dtype=np.float64)
  reconstruction = np.asarray(estimate, dtype=np.float64)
  if target.shape != reconstruction.shape:
    raise ValueError("reference and estimate shapes differ")
  if mask is None:
    return target.ravel(), reconstruction.ravel()
  selected = np.asarray(mask, dtype=bool)
  if selected.shape != target.shape or not selected.any():
    raise ValueError("evaluation mask has invalid shape or is empty")
  return target[selected], reconstruction[selected]


def nrmse(reference: np.ndarray, estimate: np.ndarray, mask: np.ndarray | None = None) -> float:
  target, reconstruction = _masked(reference, estimate, mask)
  return float(np.linalg.norm(reconstruction - target) / max(np.linalg.norm(target), 1e-12))


def mae(reference: np.ndarray, estimate: np.ndarray, mask: np.ndarray | None = None) -> float:
  target, reconstruction = _masked(reference, estimate, mask)
  return float(np.mean(np.abs(reconstruction - target)))


def psnr(reference: np.ndarray, estimate: np.ndarray, *, data_range: float, mask: np.ndarray | None = None) -> float:
  if data_range <= 0:
    raise ValueError("data_range must be the positive declared fixed range")
  target, reconstruction = _masked(reference, estimate, mask)
  mse = float(np.mean((reconstruction - target) ** 2))
  return float("inf") if mse == 0 else float(20 * np.log10(data_range / np.sqrt(mse)))


def ssim(reference: np.ndarray, estimate: np.ndarray, *, data_range: float, mask: np.ndarray | None = None) -> float:
  """SSIM averaged on an optional, fixed geometry-derived support."""
  from skimage.metrics import structural_similarity

  target = np.asarray(reference, dtype=np.float64)
  reconstruction = np.asarray(estimate, dtype=np.float64)
  if target.shape != reconstruction.shape:
    raise ValueError("reference and estimate shapes differ")
  if mask is not None:
    selected = np.asarray(mask, dtype=bool)
    if selected.shape != target.shape or not selected.any():
      raise ValueError("evaluation mask is empty")
    _, similarity_map = structural_similarity(
      target, reconstruction, data_range=data_range, gaussian_weights=True,
      sigma=1.5, use_sample_covariance=False, full=True,
    )
    return float(np.mean(similarity_map[selected]))
  return float(structural_similarity(
    target, reconstruction, data_range=data_range, gaussian_weights=True,
    sigma=1.5, use_sample_covariance=False,
  ))


def poisson_deviance(measured: np.ndarray, expected: np.ndarray, *, epsilon: float = 1e-12) -> float:
  y = np.asarray(measured, dtype=np.float64)
  mu = np.asarray(expected, dtype=np.float64)
  if y.shape != mu.shape or np.any(y < 0) or np.any(mu < 0):
    raise ValueError("measured and expected counts must match and be non-negative")
  if np.any((y > 0) & (mu == 0)):
    raise ValueError("positive observed counts occur in zero-expectation bins")
  positive = y > 0
  terms = np.empty_like(y)
  terms[positive] = y[positive] * np.log(y[positive] / np.maximum(mu[positive], epsilon)) - y[positive] + mu[positive]
  terms[~positive] = mu[~positive]
  return float(2.0 * terms.sum())
