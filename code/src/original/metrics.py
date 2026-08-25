#!/usr/bin/env python3
"""Image-quality and statistical metrics for the SAM-KEM experiment.

Provides:
- psnr_db: PSNR with a fixed data_range
- ssim_db: structural similarity index with skimage defaults
- nrmse: L2 / ||target||_2 ratio over the evaluation mask
- mae: mean absolute error over the evaluation mask
- poisson_deviance: total and per-bin deviance D(y || lambda)

The Poisson deviance uses the canonical definition
  D(y, lambda) = 2 * sum_i [lambda_i - y_i + y_i * log(y_i / lambda_i)]
with the convention that 0 * log(0 / lambda) = 0.
"""
from __future__ import annotations

import math

import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


def psnr_db(pred: np.ndarray, target: np.ndarray, *,
            data_range: float = 1.0, mask: np.ndarray | None = None) -> float:
    p = np.asarray(pred, dtype=np.float64)
    t = np.asarray(target, dtype=np.float64)
    if mask is not None:
        m = (np.asarray(mask) > 0)
        p = p[m]
        t = t[m]
        return float(peak_signal_noise_ratio(t, p, data_range=data_range))
    return float(peak_signal_noise_ratio(t, p, data_range=data_range))


def ssim_db(pred: np.ndarray, target: np.ndarray, *,
            data_range: float = 1.0, mask: np.ndarray | None = None) -> float:
    p = np.asarray(pred, dtype=np.float64)
    t = np.asarray(target, dtype=np.float64)
    if mask is not None:
        # apply mask by zeroing background on both
        m = (np.asarray(mask) > 0).astype(np.float64)
        p = p * m
        t = t * m
    return float(structural_similarity(t, p, data_range=data_range,
                                       win_size=7, gaussian_weights=True,
                                       sigma=1.5, use_sample_covariance=False,
                                       channel_axis=None))


def nrmse(pred: np.ndarray, target: np.ndarray, *,
          mask: np.ndarray | None = None) -> float:
    p = np.asarray(pred, dtype=np.float64)
    t = np.asarray(target, dtype=np.float64)
    if mask is not None:
        m = (np.asarray(mask) > 0)
        p = p[m]
        t = t[m]
    num = float(np.linalg.norm(p - t))
    den = float(np.linalg.norm(t))
    if den <= 0:
        return float("nan")
    return num / den


def mae(pred: np.ndarray, target: np.ndarray, *,
        mask: np.ndarray | None = None) -> float:
    p = np.asarray(pred, dtype=np.float64)
    t = np.asarray(target, dtype=np.float64)
    if mask is not None:
        m = (np.asarray(mask) > 0)
        p = p[m]
        t = t[m]
    return float(np.mean(np.abs(p - t)))


def rmse(pred: np.ndarray, target: np.ndarray, *,
         mask: np.ndarray | None = None) -> float:
    p = np.asarray(pred, dtype=np.float64)
    t = np.asarray(target, dtype=np.float64)
    if mask is not None:
        m = (np.asarray(mask) > 0)
        p = p[m]
        t = t[m]
    return float(np.sqrt(np.mean((p - t) ** 2)))


def poisson_deviance(y: np.ndarray, lam: np.ndarray, *,
                     mask: np.ndarray | None = None) -> tuple[float, float]:
    """Total Poisson deviance and deviance / n_bins.

    Convention: 0 * log(0 / lambda) = 0 (so empty bins do not contribute).
    """
    y = np.asarray(y, dtype=np.float64)
    lam = np.asarray(lam, dtype=np.float64)
    if mask is not None:
        m = (np.asarray(mask) > 0)
        y = y[m]
        lam = lam[m]
    if np.any(lam <= 0):
        return float("nan"), float("nan")
    safe_y = np.where(y > 0, y, 1.0)  # avoid log(0); only used in log term
    log_term = np.where(y > 0, y * np.log(safe_y / lam), 0.0)
    d = 2.0 * float(np.sum(lam - y + log_term))
    n_bins = float(y.size)
    return d, d / n_bins


def nll_per_bin(y: np.ndarray, lam: np.ndarray, *,
                mask: np.ndarray | None = None) -> float:
    """Negative log-likelihood per bin (Poisson)."""
    y = np.asarray(y, dtype=np.float64)
    lam = np.asarray(lam, dtype=np.float64)
    if mask is not None:
        m = (np.asarray(mask) > 0)
        y = y[m]
        lam = lam[m]
    if np.any(lam <= 0):
        return float("nan")
    safe_y = np.where(y > 0, y, 1.0)
    log_term = np.where(y > 0, y * np.log(safe_y / lam), 0.0)
    return float(np.mean(lam - y + log_term))


def bootstrap_subject_mean_ci(values: np.ndarray, *,
                              n_resamples: int = 10_000,
                              seed: int = 20260817,
                              ci: float = 0.95) -> dict[str, float]:
    """Subject-cluster bootstrap mean difference.

    `values` is an array of paired differences (one per subject).
    Returns mean, ci_lower, ci_upper.
    """
    values = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    n = values.size
    if n == 0:
        return {"mean": float("nan"), "ci_lower": float("nan"),
                "ci_upper": float("nan"), "n_resamples": 0, "seed": seed}
    means = np.empty(n_resamples, dtype=np.float64)
    for i in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        means[i] = float(values[idx].mean())
    lo = float(np.percentile(means, 100 * (1 - ci) / 2))
    hi = float(np.percentile(means, 100 * (1 + ci) / 2))
    return {
        "mean": float(values.mean()),
        "ci_lower": lo,
        "ci_upper": hi,
        "n_resamples": int(n_resamples),
        "seed": int(seed),
    }


def paired_mean_delta(a: np.ndarray, b: np.ndarray) -> float:
    """Paired mean difference (a - b) across subjects."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch: {a.shape} vs {b.shape}")
    return float(np.mean(a - b))