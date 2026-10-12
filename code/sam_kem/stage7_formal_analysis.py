"""Preregistered subject-level paired statistics for Stage 7 formal results."""
from __future__ import annotations

import itertools
from collections import defaultdict

import numpy as np


def paired_case_deltas(rows: list[dict]) -> list[dict]:
    """Match policy and comparator endpoints for routed Agent cases."""
    grouped: dict[tuple, dict[str, dict]] = defaultdict(dict)
    for row in rows:
        key = (
            row["dataset"], row["subject_id"], row["slice_id"],
            int(row["count_level"]), int(row["realization_id"]),
        )
        grouped[key][row["role"]] = row
    output = []
    for key, endpoints in sorted(grouped.items()):
        if "paired_comparator" not in endpoints:
            continue
        if set(endpoints) != {"policy", "paired_comparator"}:
            raise ValueError(f"incomplete or duplicate paired endpoints: {key}")
        policy = endpoints["policy"]
        comparator = endpoints["paired_comparator"]
        output.append({
            "dataset": key[0], "subject_id": key[1], "slice_id": key[2],
            "count_level": key[3], "realization_id": key[4],
            "delta_nrmse": float(policy["nrmse"]) - float(comparator["nrmse"]),
            "delta_ssim": float(policy["ssim"]) - float(comparator["ssim"]),
            "fallback_used": str(policy["fallback_used"]).lower() == "true",
        })
    return output


def subject_deltas(case_deltas: list[dict]) -> list[dict]:
    """Average paired differences within subject, separately by dataset/count."""
    grouped: dict[tuple, list[dict]] = defaultdict(list)
    for row in case_deltas:
        grouped[(row["dataset"], int(row["count_level"]), row["subject_id"])].append(row)
    output = []
    for (dataset, count, subject), values in sorted(grouped.items()):
        output.append({
            "dataset": dataset, "count_level": count, "subject_id": subject,
            "case_count": len(values),
            "mean_delta_nrmse": float(np.mean([x["delta_nrmse"] for x in values])),
            "mean_delta_ssim": float(np.mean([x["delta_ssim"] for x in values])),
            "fallback_case_count": sum(x["fallback_used"] for x in values),
        })
    return output


def bootstrap_mean_interval(values: np.ndarray, *, samples: int, seed: int) -> tuple[float, float]:
    """Return the preregistered percentile interval over resampled subjects."""
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or not array.size or not np.isfinite(array).all():
        raise ValueError("bootstrap values must be a nonempty finite vector")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, array.size, size=(samples, array.size))
    means = array[indices].mean(axis=1)
    low, high = np.percentile(means, [2.5, 97.5])
    return float(low), float(high)


def sign_flip_pvalue(values: np.ndarray, *, exact_limit: int,
                     monte_carlo_samples: int, seed: int) -> tuple[float, str]:
    """Two-sided sign-flip p-value for a subject-level mean difference."""
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or not array.size or not np.isfinite(array).all():
        raise ValueError("sign-flip values must be a nonempty finite vector")
    observed = abs(float(array.mean()))
    tolerance = 1e-15
    if array.size <= exact_limit:
        means = np.fromiter(
            (abs(float(np.mean(array * np.asarray(signs))))
             for signs in itertools.product((-1.0, 1.0), repeat=array.size)),
            dtype=np.float64, count=2 ** array.size,
        )
        return float(np.mean(means >= observed - tolerance)), "exact"
    rng = np.random.default_rng(seed)
    exceed = 0
    remaining = monte_carlo_samples
    while remaining:
        batch = min(10000, remaining)
        signs = rng.choice((-1.0, 1.0), size=(batch, array.size))
        means = np.abs((signs * array).mean(axis=1))
        exceed += int(np.sum(means >= observed - tolerance))
        remaining -= batch
    return float((exceed + 1) / (monte_carlo_samples + 1)), "monte_carlo"


def holm_adjust(pvalues: list[float]) -> list[float]:
    """Holm-adjust a fixed family of p-values while preserving input order."""
    if any(not np.isfinite(value) or not 0 <= value <= 1 for value in pvalues):
        raise ValueError("p-values must be finite in [0,1]")
    order = sorted(range(len(pvalues)), key=pvalues.__getitem__)
    adjusted = [0.0] * len(pvalues)
    running = 0.0
    total = len(pvalues)
    for rank, index in enumerate(order):
        running = max(running, min(1.0, (total - rank) * pvalues[index]))
        adjusted[index] = running
    return adjusted


def analyse_subject_cells(subject_rows: list[dict], *, bootstrap_samples: int = 10000,
                          permutation_samples: int = 100000,
                          seed: int = 20261012) -> list[dict]:
    """Compute all five-cell, two-metric preregistered results and Holm p-values."""
    grouped: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for row in subject_rows:
        grouped[(row["dataset"], int(row["count_level"]))].append(row)
    expected = {
        ("brainweb", 100000),
        ("hecktor", 100000), ("hecktor", 200000),
        ("hecktor", 500000), ("hecktor", 1000000),
    }
    if set(grouped) != expected:
        raise ValueError(f"subject cells do not match preregistration: {set(grouped)}")
    records = []
    raw_p = []
    for cell in sorted(grouped):
        values = grouped[cell]
        for metric, favourable in (("nrmse", "negative"), ("ssim", "positive")):
            array = np.asarray([row[f"mean_delta_{metric}"] for row in values])
            low, high = bootstrap_mean_interval(
                array, samples=bootstrap_samples, seed=seed,
            )
            pvalue, mode = sign_flip_pvalue(
                array, exact_limit=20, monte_carlo_samples=permutation_samples,
                seed=seed,
            )
            raw_p.append(pvalue)
            records.append({
                "dataset": cell[0], "count_level": cell[1], "metric": metric,
                "favourable_direction": favourable,
                "subject_count": len(values), "mean_delta": float(array.mean()),
                "median_delta": float(np.median(array)),
                "bootstrap_ci_low": low, "bootstrap_ci_high": high,
                "sign_flip_pvalue": pvalue, "sign_flip_mode": mode,
                "favourable_subject_fraction": float(np.mean(array < 0))
                if favourable == "negative" else float(np.mean(array > 0)),
            })
    for record, adjusted in zip(records, holm_adjust(raw_p), strict=True):
        record["holm_adjusted_pvalue"] = adjusted
    return records
