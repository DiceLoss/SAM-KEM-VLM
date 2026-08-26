"""Auditable Poisson count simulation on a noise-free projection."""

from __future__ import annotations

import numpy as np
import hashlib
import json
from pathlib import Path

from .reproducibility import CaseKey, case_seed


def array_sha256(array: np.ndarray) -> str:
  """Hash dtype, shape and contiguous bytes to identify an array exactly."""
  value = np.ascontiguousarray(array)
  digest = hashlib.sha256()
  digest.update(value.dtype.str.encode("ascii"))
  digest.update(repr(value.shape).encode("ascii"))
  digest.update(value.tobytes())
  return digest.hexdigest()


def simulate_low_count(
  noise_free_projection: np.ndarray,
  *,
  expected_true_counts: int,
  case: CaseKey,
  experiment_seed: int,
) -> tuple[np.ndarray, int, float]:
  if int(case.count_level) != int(expected_true_counts):
    raise ValueError(
      f"case count_level={case.count_level} differs from expected_true_counts={expected_true_counts}"
    )
  projection = np.asarray(noise_free_projection, dtype=np.float64)
  if np.any(projection < 0) or not np.all(np.isfinite(projection)):
    raise ValueError("noise-free projection must be finite and non-negative")
  total = float(projection.sum())
  if total <= 0 or expected_true_counts <= 0:
    raise ValueError("projection sum and expected_true_counts must be positive")
  projection_scale = float(expected_true_counts) / total
  expectation = projection * projection_scale
  seed = case_seed(case, experiment_seed=experiment_seed)
  sample = np.random.default_rng(seed).poisson(expectation)
  return sample.astype(np.int64), seed, projection_scale


def simulation_diagnostics(measured: np.ndarray, expectation: np.ndarray) -> dict[str, float | int]:
  """Return per-case diagnostics appropriate for heterogeneous Poisson bins."""
  y = np.asarray(measured, dtype=np.float64)
  lam = np.asarray(expectation, dtype=np.float64)
  if y.shape != lam.shape or np.any(y < 0) or np.any(lam < 0):
    raise ValueError("measured and expectation must be shape-matched and non-negative")
  impossible = (y > 0) & (lam == 0)
  if impossible.any():
    raise ValueError("positive observed counts occur in zero-expectation bins")
  usable = lam >= 1e-6
  if not usable.any():
    raise ValueError("no positive-rate projection bins")
  pearson = float(np.sum(((y[usable] - lam[usable]) ** 2) / lam[usable]))
  positive = (y > 0) & (lam > 0)
  deviance_terms = lam - y
  deviance_terms[positive] += y[positive] * np.log(y[positive] / lam[positive])
  return {
    "n_bins": int(y.size), "n_positive_rate_bins": int(usable.sum()),
    "expected_total_counts": float(lam.sum()),
    "realized_total_counts": int(y.sum()),
    "pearson_chi_square": pearson,
    "pearson_dispersion": pearson / max(int(usable.sum()), 1),
    "poisson_deviance": float(2.0 * deviance_terms.sum()),
  }


def write_low_count_simulation(
  noise_free_projection_path: Path,
  output_path: Path,
  *,
  expected_true_counts: int,
  case: CaseKey,
  experiment_seed: int,
) -> Path:
  projection = np.load(noise_free_projection_path)
  measured, seed, scale = simulate_low_count(
    projection, expected_true_counts=expected_true_counts,
    case=case, experiment_seed=experiment_seed,
  )
  expectation = np.asarray(projection, dtype=np.float64) * scale
  output_path.parent.mkdir(parents=True, exist_ok=True)
  if output_path.exists() or output_path.with_suffix(".json").exists():
    raise FileExistsError(f"refusing to overwrite simulation: {output_path}")
  np.save(output_path, measured)
  metadata = {
    "schema_version": 1,
    "observation_model": "independent_poisson_scaled_noise_free_projection",
    "rng": "numpy.random.Generator(PCG64)",
    "dataset": case.dataset, "subject_id": case.subject_id,
    "slice_id": case.slice_id, "count_level": int(case.count_level),
    "realization_id": int(case.realization_id), "seed": int(seed),
    "projection_scale": scale,
    "noise_free_projection_path": str(noise_free_projection_path),
    "noise_free_projection_sha256": array_sha256(projection),
    "sinogram_sha256": array_sha256(measured),
    **simulation_diagnostics(measured, expectation),
  }
  output_path.with_suffix(".json").write_text(
    json.dumps(metadata, indent=2), encoding="utf-8"
  )
  return output_path
