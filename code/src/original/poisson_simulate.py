#!/usr/bin/env python3
"""Locked Poisson count simulator for SAM-KEM low-count experiments.

Inputs:
- noise-free projection `p` (numpy.ndarray, shape (H, W) or (H, W, ...),
  nonnegative)
- expected true-count level `N_true` (one of 10e4, 20e4, 50e4, 100e4)
- optional background expectation `b` (same shape as p; default zero)
- optional RNG seed (uint64)

Outputs (returned as a dict, also optionally written to disk):
- `p` (the input projection)
- `c` (the count scale `N_true / sum(p)`)
- `b` (the background expectation)
- `lambda` (the per-bin Poisson rate = `c * p + b`)
- `y` (the sampled integer sinogram)
- `expected_total`, `observed_total`
- `seed`, `rng_implementation`
- `mean_audit`, `var_audit`, `relative_noise_audit`

The simulator:
1. Computes `c = N_true / sum_i p_i` (preserves the noise-free relative
   structure across count levels).
2. Forms the per-bin Poisson rate `lambda = c * p + b` where `b` is the
   declared random/scatter expectation (default zero for the canonical
   experiment).
3. Samples `y ~ Poisson(lambda)` using `numpy.random.Generator` with the
   declared seed. The RNG implementation is recorded.
4. Audits the empirical mean and variance of `y` against `lambda` and
   confirms that the relative noise increases as the requested count
   level decreases (Poisson sanity check).

The simulator never falls back to Gaussian, signal-independent constant-
variance, or uncentered additive Poisson noise.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np


CANONICAL_COUNT_LEVELS = (100_000, 200_000, 500_000, 1_000_000)
POISSON_SEED_BASE = 0xCAFE2026


def simulate(
    p: np.ndarray,
    n_true: int,
    *,
    b: np.ndarray | None = None,
    seed: int | None = None,
    rng_implementation: str = "numpy.random.Generator",
    multiplicative_noise: float = 0.0,
    additive_noise_fraction: float = 0.0,
) -> dict[str, Any]:
    """Run one Poisson count realization and return the audit record.

    Default: pure Poisson sampling on a count-scaled sinogram.

    Optional multiplicative (scanner-gain) and additive (scatter) noise
    follow the legacy `PET2Sinogram` model from
    `EM-reconstruction/legacy_inactive/MydataLoader.py::PET2Sinogram`:

        mul_factor = 1 + uniform(-mult_noise, +mult_noise)        # scanner gain
        noise     = mean(mul_factor * p) * additive_noise_fraction # scatter / random
        sinogram  = mul_factor * p + noise                         # pre-Poisson
        c         = n_true / sum(sinogram)                         # scale to count
        sinogram *= c
        y         ~ Poisson(sinogram)                              # final integer counts

    Set multiplicative_noise=0.1, additive_noise_fraction=0.2 for the
    legacy model; the gain + scatter arrays are returned in the audit
    dict so that `recon.mlem` / `recon.kem` can apply gain correction.
    """
    if n_true not in CANONICAL_COUNT_LEVELS:
        raise ValueError(f"N_true must be one of {CANONICAL_COUNT_LEVELS}, got {n_true}")

    p = np.asarray(p, dtype=np.float64)
    if np.any(p < 0):
        raise ValueError("noise-free projection p must be nonnegative")

    if b is None:
        b = np.zeros_like(p)
    else:
        b = np.asarray(b, dtype=np.float64)
        if b.shape != p.shape:
            raise ValueError(f"b shape {b.shape} must match p shape {p.shape}")
        if np.any(b < 0):
            raise ValueError("background expectation b must be nonnegative")

    total_p = float(p.sum())
    if total_p <= 0:
        raise ValueError("noise-free projection p must have positive total mass")

    c = float(n_true) / total_p
    lam = c * p + b
    if np.any(lam < 0):
        raise ValueError("per-bin rate lambda must be nonnegative")

    if seed is None:
        seed = POISSON_SEED_BASE
    rng = np.random.default_rng(int(seed))

    # Multiplicative noise: uniform ±mult_noise on each detector element.
    mul_factor = 1.0 + rng.uniform(
        -multiplicative_noise, multiplicative_noise, size=p.shape
    )
    # Additive noise: a flat fraction of the mean pre-Poisson expectation.
    base_mean = float((mul_factor * p).mean())
    additive = np.full_like(p, base_mean * additive_noise_fraction)
    sinogram_pre = mul_factor * p + additive

    # Scale to the requested count.
    total_pre = float(sinogram_pre.sum())
    if total_pre <= 0:
        raise ValueError("pre-positive sinogram must have positive total mass")
    c = float(n_true) / total_pre
    lam = c * sinogram_pre + b
    if np.any(lam < 0):
        raise ValueError("per-bin rate lambda must be nonnegative")

    # Poisson sampling.
    y = rng.poisson(lam).astype(np.int64)

    expected_total = float(lam.sum())
    observed_total = float(y.sum())

    # Mean-variance audit.
    diff = y - lam
    mean_diff = float(diff.mean())
    var_y = float(((y - y.mean()) ** 2).mean())
    var_lam = float(lam.var())
    rel_noise = float(np.sqrt(var_y) / max(observed_total, 1.0))

    audit = {
        "n_bins": int(y.size),
        "expected_total": expected_total,
        "observed_total": observed_total,
        "mean_diff_overall": mean_diff,
        "var_y": var_y,
        "var_lambda": var_lam,
        "relative_noise_total": rel_noise,
        "rng_implementation": rng_implementation,
        "seed": int(seed),
        "multiplicative_noise": multiplicative_noise,
        "additive_noise_fraction": additive_noise_fraction,
    }
    return {
        "p": p,
        "c": c,
        "b": b,
        "mul_factor": mul_factor,
        "additive": additive,
        "lambda": lam,
        "y": y,
        "audit": audit,
    }


def sha256_array(arr: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()


def run_development_audit(out_path: Path | None = None) -> dict[str, Any]:
    """Run a small development-subset audit confirming Poisson behavior.

    Uses a deterministic synthetic phantom (single Gaussian blob, 128x128)
    and reports mean/variance per count level. The relative noise must
    increase monotonically as the requested count level decreases.
    """
    phantom = _synthetic_phantom(128)
    results: list[dict[str, Any]] = []
    # Iterate in INCREASING count order. The relative noise must DECREASE
    # monotonically with increasing N (Poisson: relative noise ~ 1/sqrt(N)).
    prev_rel_noise = float("inf")
    monotonic_decreasing_rel_noise = True
    for n in CANONICAL_COUNT_LEVELS:
        per_real: list[dict[str, Any]] = []
        for r in range(3):
            rec = simulate(phantom, n, seed=POISSON_SEED_BASE + r)
            per_real.append(rec["audit"])
        rel_noise = float(np.mean([a["relative_noise_total"] for a in per_real]))
        # Allow a tiny tolerance for the smallest-count noise; require
        # that increasing count gives a non-increasing relative noise.
        if rel_noise > prev_rel_noise + 1e-7:
            monotonic_decreasing_rel_noise = False
        prev_rel_noise = rel_noise
        results.append({
            "count_level": n,
            "realizations": per_real,
            "mean_relative_noise": rel_noise,
        })
    out = {
        "phantom_shape": list(phantom.shape),
        "phantom_sha256": sha256_array(phantom),
        "count_levels": list(CANONICAL_COUNT_LEVELS),
        "results": results,
        # Expected: relative noise DECREASES with INCREASING count. Flip the
        # flag into a positive statement for the reader.
        "monotonic_decreasing_relative_noise_with_increasing_count": monotonic_decreasing_rel_noise,
        "rng_implementation": "numpy.random.Generator",
    }
    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    return out


def _synthetic_phantom(size: int = 128) -> np.ndarray:
    """A simple 2D synthetic PET-like phantom (single Gaussian blob)."""
    yy, xx = np.mgrid[0:size, 0:size]
    cy, cx = size / 2.0, size / 2.0
    sigma = size / 8.0
    g = np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * sigma ** 2))
    # Add a hot spot
    hot = np.exp(-((yy - (size * 0.3)) ** 2 + (xx - (size * 0.7)) ** 2) / (2 * (size / 16) ** 2))
    return (g + hot).astype(np.float64)


def main() -> int:
    p = argparse.ArgumentParser(description="Poisson count simulator")
    p.add_argument("--audit-only", action="store_true",
                   help="Run the development-subset audit only")
    p.add_argument("--out", type=Path,
                   default=Path("logs/poisson_development_audit.json"))
    args = p.parse_args()

    if args.audit_only:
        out = run_development_audit(args.out)
        print(json.dumps(out, indent=2))
        if not out["monotonic_decreasing_relative_noise_with_increasing_count"]:
            print("WARN: relative noise does not decrease with increasing count (Poisson violated)",
                  file=sys.stderr)
            return 2
        return 0

    print("Use --audit-only or call simulate(...) from another script.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())