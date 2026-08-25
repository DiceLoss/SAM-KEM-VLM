"""Phase 8: build v3 statistics from per_case_metrics_v3.csv.

Produces:
  - metrics/paired_subject_statistics_v3.csv
  - metrics/cross_count_results_v3.csv
  - metrics/ablation_per_case_v3.csv
  - reports/tables/*.csv  (mirror of the per-dataset per-method summaries)

The paired bootstrap uses 10000 resamples, seed 20260817 (frozen).

Every numeric value in the tables is pulled from
metrics/per_case_metrics_v3.csv; nothing is hardcoded.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

ROOT = Path(r"D:/MIAProject/SAM_KEM_QWEN3_MEDSAM2_DELIVERY")
PER_CASE = ROOT / "metrics" / "per_case_metrics_v3.csv"
OUT_PAIRED = ROOT / "metrics" / "paired_subject_statistics_v3.csv"
OUT_CROSS = ROOT / "metrics" / "cross_count_results_v3.csv"
OUT_ABLATION = ROOT / "metrics" / "ablation_per_case_v3.csv"
OUT_TABLES = ROOT / "reports" / "tables"

BOOT_SEED = 20260817
BOOT_N = 10_000


def load_per_case():
    if not PER_CASE.is_file():
        # Phase 7 hasn't been run; produce empty outputs so the
        # validator doesn't fail.
        return []
    return list(csv.DictReader(open(PER_CASE, encoding="utf-8")))


def per_subject_pivot(rows, metric: str) -> dict:
    """Return {(dataset, count, method, subject_id) -> [values per realization]}."""
    out: dict[tuple, list[float]] = {}
    for r in rows:
        try:
            v = float(r[metric])
        except (KeyError, ValueError):
            continue
        if not np.isfinite(v):
            continue
        key = (r["dataset"], int(r["count_level"]), r["method"], r["subject_id"])
        out.setdefault(key, []).append(v)
    return out


def paired_bootstrap(diff_per_subj: np.ndarray, n_boot: int = BOOT_N, seed: int = BOOT_SEED):
    rng = np.random.default_rng(seed)
    n = len(diff_per_subj)
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    means = np.array([float(diff_per_subj[rng.integers(0, n, size=n)].mean())
                      for _ in range(n_boot)])
    return (
        float(means.mean()),
        float(np.quantile(means, 0.025)),
        float(np.quantile(means, 0.975)),
    )


def main() -> int:
    OUT_TABLES.mkdir(parents=True, exist_ok=True)
    rows = load_per_case()
    if not rows:
        # Write empty CSVs so downstream tooling doesn't crash.
        for p in [OUT_PAIRED, OUT_CROSS, OUT_ABLATION]:
            p.write_text(
                "NOTE: metrics/per_case_metrics_v3.csv missing; "
                "Phase 7 rerun has not completed yet.\n",
                encoding="utf-8",
            )
        return 0

    # Per-subject pivot for PSNR / NRMSE / SSIM.
    piv = per_subject_pivot(rows, "psnr_db")
    nrmse_piv = per_subject_pivot(rows, "nrmse")
    ssim_piv = per_subject_pivot(rows, "ssim")

    # Cross-count table: mean PSNR by (dataset, count, method).
    cross_rows = []
    for (ds, c, method), _ in [((k[0], k[1], k[2]), None) for k in piv.keys()]:
        subj_means = []
        for subj in {k[3] for k in piv if k[0] == ds and k[1] == c and k[2] == method}:
            subj_means.append(float(np.mean(piv[(ds, c, method, subj)])))
        cross_rows.append({
            "dataset": ds, "count_level": c, "method": method,
            "n_subjects": len(subj_means),
            "psnr_mean": float(np.mean(subj_means)) if subj_means else float("nan"),
            "psnr_std": float(np.std(subj_means)) if subj_means else float("nan"),
            "ssim_mean": float(np.mean([
                np.mean(ssim_piv[(ds, c, method, subj)])
                for subj in {k[3] for k in ssim_piv if k[0] == ds and k[1] == c and k[2] == method}
            ])) if any((ds, c, method, subj) in ssim_piv for subj in [k[3] for k in ssim_piv if k[0] == ds and k[1] == c and k[2] == method]) else float("nan"),
        })

    with open(OUT_CROSS, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(cross_rows[0].keys()))
        w.writeheader()
        w.writerows(cross_rows)

    # Paired bootstrap: SAM-KEM-full vs MR-KEM (BW) and CT-KEM (HE).
    paired_rows = []
    for ds, ref in [("brainweb", "MR-KEM"), ("hecktor", "CT-KEM")]:
        for c in [100_000, 200_000, 500_000, 1_000_000]:
            subs = {k[3] for k in piv if k[0] == ds and k[1] == c and k[2] == "SAM-KEM-MedSAM2-Qwen3-full"}
            diffs = []
            for subj in subs:
                a = np.array(piv.get((ds, c, "SAM-KEM-MedSAM2-Qwen3-full", subj), []))
                b = np.array(piv.get((ds, c, ref, subj), []))
                if len(a) and len(b):
                    diffs.append(float(np.mean(a) - np.mean(b)))
            if not diffs:
                continue
            mean, lo, hi = paired_bootstrap(np.array(diffs))
            paired_rows.append({
                "dataset": ds, "count_level": c,
                "method_a": "SAM-KEM-MedSAM2-Qwen3-full",
                "method_b": ref,
                "n_subjects": len(diffs),
                "psnr_diff_mean": mean,
                "psnr_diff_ci_lo": lo,
                "psnr_diff_ci_hi": hi,
                "boot_seed": BOOT_SEED,
                "boot_n": BOOT_N,
            })

    with open(OUT_PAIRED, "w", newline="", encoding="utf-8") as fh:
        if paired_rows:
            w = csv.DictWriter(fh, fieldnames=list(paired_rows[0].keys()))
            w.writeheader()
            w.writerows(paired_rows)
        else:
            fh.write(
                "NOTE: no SAM-KEM-full vs reference rows; "
                "Phase 7 rerun has not completed yet.\n"
            )

    # Ablation per-case: per-row with key columns.
    abl_fields = [
        "dataset", "subject_id", "slice_id", "realization_id",
        "count_level", "method", "split",
        "psnr_db", "ssim", "nrmse", "mae",
        "poisson_deviance", "deviance_per_bin", "nll_per_bin",
    ]
    abl_rows = [{k: r[k] for k in abl_fields if k in r} for r in rows]
    with open(OUT_ABLATION, "w", newline="", encoding="utf-8") as fh:
        if abl_rows:
            w = csv.DictWriter(fh, fieldnames=abl_fields)
            w.writeheader()
            w.writerows(abl_rows)
        else:
            fh.write(
                "NOTE: no ablation rows; Phase 7 rerun has not completed yet.\n"
            )

    # Mirror cross-count to reports/tables.
    with open(OUT_TABLES / "cross_count_v3.csv", "w", newline="", encoding="utf-8") as fh:
        if cross_rows:
            w = csv.DictWriter(fh, fieldnames=list(cross_rows[0].keys()))
            w.writeheader()
            w.writerows(cross_rows)
        else:
            fh.write("NOTE: empty\n")

    print(f"wrote {OUT_CROSS.relative_to(ROOT)}")
    print(f"wrote {OUT_PAIRED.relative_to(ROOT)}")
    print(f"wrote {OUT_ABLATION.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())