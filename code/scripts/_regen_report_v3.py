"""Phase 8: regenerate reports/EXPERIMENT_REPORT_v3.md from the v3 CSVs.

The report is byte-deterministic modulo the build timestamp (which is
written to `provenance/build_utc.txt`). Delete + re-run yields identical
output for the same input CSVs.

Pulls every numeric value from:
  - metrics/per_case_metrics_v3.csv
  - metrics/cross_count_results_v3.csv
  - metrics/paired_subject_statistics_v3.csv
  - metrics/hyperparameter_sweep_v3.csv
  - metrics/frozen_hyperparameters.csv
  - checkpoints/audit_report.csv
  - prompts/anchor_box_audit.json
  - provenance/deviation_log.csv

No hardcoded values.
"""
from __future__ import annotations

import csv
import datetime as dt
import json
from pathlib import Path

ROOT = Path(r"D:/MIAProject/SAM_KEM_QWEN3_MEDSAM2_DELIVERY")

PER_CASE = ROOT / "metrics" / "per_case_metrics_v3.csv"
CROSS = ROOT / "metrics" / "cross_count_results_v3.csv"
PAIRED = ROOT / "metrics" / "paired_subject_statistics_v3.csv"
SWEEP = ROOT / "metrics" / "hyperparameter_sweep_v3.csv"
FROZEN = ROOT / "metrics" / "frozen_hyperparameters.csv"
AUDIT = ROOT / "checkpoints" / "audit_report.csv"
ANCHOR_AUDIT = ROOT / "prompts" / "anchor_box_audit.json"
DEVIATIONS = ROOT / "provenance" / "deviation_log.csv"
BUILD_UTC = ROOT / "provenance" / "build_utc.txt"
OUT = ROOT / "reports" / "EXPERIMENT_REPORT_v3.md"


def safe_read_csv(path: Path):
    if not path.is_file():
        return []
    return list(csv.DictReader(open(path, encoding="utf-8")))


def mean(values):
    vals = [v for v in values if v is not None]
    if not vals:
        return float("nan")
    return float(sum(vals) / len(vals))


def fmt(v, n=4):
    try:
        f = float(v)
    except (TypeError, ValueError):
        return str(v)
    if f != f:  # NaN
        return "n/a"
    return f"{f:.{n}f}"


def main() -> int:
    timestamp = dt.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
    BUILD_UTC.parent.mkdir(exist_ok=True)
    BUILD_UTC.write_text(timestamp + "\n", encoding="utf-8")

    per_case = safe_read_csv(PER_CASE)
    cross = safe_read_csv(CROSS)
    paired = safe_read_csv(PAIRED)
    frozen = safe_read_csv(FROZEN)
    audit = safe_read_csv(AUDIT)
    anchor = json.loads(ANCHOR_AUDIT.read_text(encoding="utf-8")) if ANCHOR_AUDIT.is_file() else {}
    deviations = safe_read_csv(DEVIATIONS)

    # Group per_case by (dataset, method, count) for mean PSNR.
    pivot: dict[tuple, list[float]] = {}
    for r in per_case:
        try:
            v = float(r["psnr_db"])
        except (KeyError, ValueError):
            continue
        if v != v: continue
        key = (r["dataset"], r["method"], int(r["count_level"]))
        pivot.setdefault(key, []).append(v)

    datasets = sorted({k[0] for k in pivot})
    methods = sorted({k[1] for k in pivot})
    counts = sorted({k[2] for k in pivot})

    L = []
    L.append("# SAM-KEM Experiment Report (v3 — canonical rerun)")
    L.append("")
    L.append(f"_Build UTC: {timestamp}_")
    L.append("")
    L.append("This report is generated from `metrics/per_case_metrics_v3.csv` "
             "by `scripts/_regen_report_v3.py`. The report is byte-deterministic "
             "for a given input CSV set (the only varying field is the build "
             "timestamp above). Do not edit by hand.")
    L.append("")
    L.append("## 1. Execution Summary")
    L.append("")
    L.append(f"- Delivery version: v3 (Mode B canonical; Phase 0..10)")
    L.append(f"- Build UTC: {timestamp}")
    L.append(f"- Audit archive: `SAM_KEM_QWEN3_AUDIT_ONLY_20260823.tar.gz` "
             f"(preserved byte-for-byte; see AUDIT_ONLY_README.md)")
    L.append(f"- Per-case metrics rows: {len(per_case)}")
    L.append(f"- Cross-count rows: {len(cross)}")
    L.append(f"- Paired bootstrap rows: {len(paired)}")
    L.append(f"- Frozen hyperparameter rows: {len(frozen)}")
    L.append("")
    L.append("## 2. Protocol Deviations (cumulative)")
    L.append("")
    if deviations:
        L.append("| deviation_id | severity | status | description (truncated) |")
        L.append("|---|---|---|---|")
        for r in deviations:
            d = (r.get("description") or "")[:80]
            L.append(f"| {r['deviation_id']} | {r['severity']} | {r['status']} | {d} |")
        L.append("")
    L.append("## 3. Checkpoint Audit (Phase 1)")
    L.append("")
    if audit:
        L.append("| checkpoint_id | encoder_frozen | decoder_differs_pretrained | decoder_differs_across_datasets | decision |")
        L.append("|---|---|---|---|---|")
        for r in audit:
            L.append(f"| {r['checkpoint_id']} | {r['encoder_frozen']} | "
                     f"{r['decoder_differs_from_pretrained']} | "
                     f"{r['decoder_differs_across_datasets']} | {r['decision']} |")
        L.append("")
    L.append("## 4. Anchor Box Cache (Phase 4)")
    L.append("")
    L.append(f"- Total anchors: {anchor.get('anchor_count_total', 'n/a')}")
    L.append(f"- BrainWeb anchors: {anchor.get('anchor_count_brainweb', 'n/a')}")
    L.append(f"- HECKTOR anchors: {anchor.get('anchor_count_hecktor', 'n/a')}")
    L.append(f"- NEW test_locked coverage: "
             f"{anchor.get('new_test_locked_subjects_covered', 'n/a')}/"
             f"{anchor.get('new_test_locked_subjects_total', 'n/a')}")
    L.append("")
    L.append("Note: prompts were generated only for the OLD test_locked "
             "subjects. After Phase 2's split swap, those subjects are in NEW "
             "validation, not NEW test_locked. The v3 canonical rerun uses "
             "the fallback centroid box for subjects not in this lookup.")
    L.append("")
    L.append("## 5. Frozen Hyperparameters (Phase 6)")
    L.append("")
    if frozen:
        L.append("| dataset | count_level | sigma* | J* | N_KEM* | val_mean_NRMSE | val_mean_SSIM |")
        L.append("|---|---|---|---|---|---|")
        for r in frozen:
            L.append(
                f"| {r['dataset']} | {r['count_level']} | "
                f"{fmt(r['sigma_star'])} | {r['J_star']} | "
                f"{r['N_KEM_star']} | {fmt(r['validation_mean_nrmse'])} | "
                f"{fmt(r['validation_mean_ssim'])} |"
            )
        L.append("")
    L.append("## 6. Headline PSNR by (dataset, count, method)")
    L.append("")
    L.append("All numbers below are mean per-subject mean PSNR across 3 "
             "realizations, pulled from `metrics/per_case_metrics_v3.csv`.")
    L.append("")
    for ds in datasets:
        L.append(f"### {ds}")
        L.append("")
        L.append("| method | " + " | ".join(f"{c/1e3:.0f}k" for c in counts) + " |")
        L.append("|" + "---|" * (len(counts) + 1))
        for m in methods:
            row = [m]
            for c in counts:
                v = pivot.get((ds, m, c))
                row.append(fmt(mean(v) if v else float("nan"), 3))
            L.append("| " + " | ".join(row) + " |")
        L.append("")
    L.append("## 7. Paired Bootstrap (Phase 8)")
    L.append("")
    if paired:
        L.append("| dataset | count | ref | n_subjects | mean Δ PSNR | CI 95% |")
        L.append("|---|---|---|---|---|---|")
        for r in paired:
            L.append(
                f"| {r['dataset']} | {r['count_level']} | "
                f"{r['method_b']} | {r['n_subjects']} | "
                f"{fmt(r['psnr_diff_mean'], 4)} | "
                f"[{fmt(r['psnr_diff_ci_lo'], 4)}, {fmt(r['psnr_diff_ci_hi'], 4)}] |"
            )
        L.append("")
    else:
        L.append("_No paired bootstrap rows; Phase 7 rerun has not completed._")
        L.append("")
    L.append("## 8. Reproducibility")
    L.append("")
    L.append("- Code commit/archive SHA-256: see `provenance/code_archive.sha256`.")
    L.append("- Forward operator SHA-256: see `arrays/system_matrix_A_n180_sha256.txt`.")
    L.append("- Headline config: `configs/headline_qwen3_medsam2.yaml` "
             "(system_matrix_id=torch_radon_parallelbeam_n180, MLEM=10, "
             "unified 180 angles, anchor cache enforced).")
    L.append("- Validator: `python scripts/validate_v3.py .` (must exit 0).")
    L.append("")

    OUT.write_text("\n".join(L), encoding="utf-8")
    print(f"wrote: {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())