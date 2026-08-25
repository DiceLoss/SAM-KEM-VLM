"""Phase 0: add AUDIT_ONLY_DO_NOT_USE_FOR_MANUSCRIPT row-1 marker to all
CSVs in metrics/ and convergence/.

The marker is added as a comment line BEFORE the CSV header (CSV has no
native comment syntax, but most consumers will ignore an unknown first
line; the validator (`scripts/validate_v3.py`) explicitly checks for the
marker).

Also appends D17, D18, D19 to provenance/deviation_log.csv documenting
the archive, the voided results, and the test-set swap rationale.
"""
from __future__ import annotations

import csv
from pathlib import Path

ROOT = Path(r"D:/MIAProject/SAM_KEM_QWEN3_MEDSAM2_DELIVERY")
MARKER = "# AUDIT_ONLY_DO_NOT_USE_FOR_MANUSCRIPT -- see AUDIT_ONLY_README.md"

CSV_DIRS = [ROOT / "metrics", ROOT / "convergence"]


def add_marker(csv_path: Path) -> None:
    if not csv_path.is_file():
        return
    txt = csv_path.read_text(encoding="utf-8")
    lines = txt.splitlines(keepends=False)
    # Avoid double-stamping.
    if lines and lines[0].startswith("# AUDIT_ONLY"):
        return
    new_lines = [MARKER] + lines
    csv_path.write_text("\n".join(new_lines) + "\n", encoding="utf-8")
    print(f"marked: {csv_path.relative_to(ROOT)}")


def main() -> None:
    for d in CSV_DIRS:
        if not d.is_dir():
            print(f"skip missing dir: {d}")
            continue
        for f in sorted(d.glob("*.csv")):
            add_marker(f)

    # Append D17/D18/D19 to deviation_log.csv.
    dev = ROOT / "provenance" / "deviation_log.csv"
    rows = list(csv.DictReader(open(dev, encoding="utf-8")))
    new_rows = [
        {
            "deviation_id": "D17",
            "description": (
                "Current SAM_KEM_QWEN3_FINAL package is preserved byte-for-byte "
                "as D:/MIAProject/SAM_KEM_QWEN3_AUDIT_ONLY_20260823.tar.gz "
                "(sha256=95dbb1a470fe1ddcf3e4f17b8801da6649e829b7fcca67253e193da12a7e52c4). "
                "All current results are voided for manuscript use; "
                "AUDIT_ONLY_README.md enumerates the 12 audit issues."
            ),
            "discovered_utc": "2026-08-23T22:00:00Z",
            "discovered_before_test_review": "yes",
            "affected_run_ids": "all",
            "severity": "high",
            "decision": "preserve_void_rebuild",
            "mitigation": (
                "scripts/_phase0_mark_audit_only.py stamps every CSV in "
                "metrics/ and convergence/ with # AUDIT_ONLY_DO_NOT_USE_FOR_MANUSCRIPT "
                "and prepends the same marker to reports/EXPERIMENT_REPORT.md."
            ),
            "status": "open",
        },
        {
            "deviation_id": "D18",
            "description": (
                "Stale numbers in reports/EXPERIMENT_REPORT.md. Report quotes "
                "BW SAM-KEM-full - MR-KEM = -0.130 NRMSE while CSV has +0.531873. "
                "All report numbers must be regenerated from the v3 CSV via "
                "scripts/_regen_report_v3.py and never hardcoded."
            ),
            "discovered_utc": "2026-08-23T22:00:00Z",
            "discovered_before_test_review": "yes",
            "affected_run_ids": "all",
            "severity": "high",
            "decision": "regenerate_from_v3_csv",
            "mitigation": (
                "EXPERIMENT_REPORT_v3.md is byte-deterministic from "
                "metrics/per_case_metrics_v3.csv (delete + re-run yields identical "
                "modulo build timestamp)."
            ),
            "status": "open",
        },
        {
            "deviation_id": "D19",
            "description": (
                "Test-set leakage: convergence/convergence.csv was generated "
                "against the original test_locked subjects (subject41/42/43 BW; "
                "34 HE), so test_locked is no longer a clean held-out test set. "
                "Splits must swap so the NEW validation = old test_locked and "
                "the NEW test_locked = old validation; train split is unchanged."
            ),
            "discovered_utc": "2026-08-23T22:00:00Z",
            "discovered_before_test_review": "yes",
            "affected_run_ids": "all",
            "severity": "high",
            "decision": "swap_val_test_locked",
            "mitigation": (
                "scripts/_swap_splits.py swaps the two splits in-place; backup "
                "is saved as splits/subject_split_v1_AUDIT_ONLY.csv.bak. "
                "splits/swap_log.csv records the mapping for both datasets."
            ),
            "status": "open",
        },
    ]
    fields = list(rows[0].keys()) if rows else [
        "deviation_id", "description", "discovered_utc",
        "discovered_before_test_review", "affected_run_ids",
        "severity", "decision", "mitigation", "status",
    ]
    # Make sure new rows have all fields.
    for r in new_rows:
        for k in fields:
            r.setdefault(k, "")
    with open(dev, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for r in rows + new_rows:
            w.writerow(r)
    print(f"appended D17/D18/D19 to: {dev.relative_to(ROOT)}")


if __name__ == "__main__":
    main()