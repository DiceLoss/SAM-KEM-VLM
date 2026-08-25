"""Phase 2: swap validation <-> test_locked for BW and HE.

Inputs:  splits/subject_split.csv
Outputs:
  - splits/subject_split_v1_AUDIT_ONLY.csv.bak   (pre-swap backup)
  - splits/subject_split.csv                     (in-place, swapped)
  - splits/swap_log.csv                          (audit trail)

Split policy:
  - BW: train = 14 unchanged.
        new validation = old test_locked = {subject41, subject42, subject43}.
        new test_locked = old validation = {subject46, subject47, subject48}.
  - HE: train = 156 unchanged.
        new validation = old test_locked (34 HE subjects).
        new test_locked = old validation (34 HE subjects).
"""
from __future__ import annotations

import csv
import shutil
from pathlib import Path

ROOT = Path(r"D:/MIAProject/SAM_KEM_QWEN3_MEDSAM2_DELIVERY")
SRC = ROOT / "splits" / "subject_split.csv"
BAK = ROOT / "splits" / "subject_split_v1_AUDIT_ONLY.csv.bak"
LOG = ROOT / "splits" / "swap_log.csv"


def main() -> int:
    if BAK.is_file():
        print(f"backup already exists at {BAK}; refusing to overwrite")
        return 2

    shutil.copy2(SRC, BAK)
    print(f"backed up to: {BAK}")

    rows = list(csv.DictReader(open(SRC, encoding="utf-8")))
    fields = list(rows[0].keys())

    swap_counts = {"brainweb": {}, "hecktor": {}}
    new_rows = []
    for r in rows:
        ds = r["dataset"]
        if ds not in swap_counts:
            new_rows.append(r)
            continue
        split = r["split"]
        if split == "validation":
            r["split"] = "test_locked"
            swap_counts[ds].setdefault("validation_to_test_locked", []).append(r["subject_id"])
        elif split == "test_locked":
            r["split"] = "validation"
            swap_counts[ds].setdefault("test_locked_to_validation", []).append(r["subject_id"])
        new_rows.append(r)

    with open(SRC, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(new_rows)
    print(f"wrote swapped splits to: {SRC}")

    # Audit log.
    with open(LOG, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(
            fh,
            fieldnames=["dataset", "direction", "subject_ids", "count"],
        )
        w.writeheader()
        for ds, d in swap_counts.items():
            for direction, ids in d.items():
                w.writerow({
                    "dataset": ds,
                    "direction": direction,
                    "subject_ids": ",".join(sorted(ids, key=lambda x: (len(x), x))),
                    "count": len(ids),
                })
    print(f"wrote swap log to: {LOG}")

    # Quick validation summary.
    summary = {"train": 0, "validation": 0, "test_locked": 0}
    for r in new_rows:
        summary[r["split"]] = summary.get(r["split"], 0) + 1
    print(f"new totals: {summary}")

    bw_val = sorted(r["subject_id"] for r in new_rows
                    if r["dataset"] == "brainweb" and r["split"] == "validation")
    bw_tst = sorted(r["subject_id"] for r in new_rows
                    if r["dataset"] == "brainweb" and r["split"] == "test_locked")
    he_val = sorted(int(r["subject_id"]) for r in new_rows
                    if r["dataset"] == "hecktor" and r["split"] == "validation")
    he_tst = sorted(int(r["subject_id"]) for r in new_rows
                    if r["dataset"] == "hecktor" and r["split"] == "test_locked")
    print(f"BW validation ({len(bw_val)}): {bw_val}")
    print(f"BW test_locked ({len(bw_tst)}): {bw_tst}")
    print(f"HE validation ({len(he_val)}): {he_val[:10]}... ({len(he_val)} total)")
    print(f"HE test_locked ({len(he_tst)}): {he_tst[:10]}... ({len(he_tst)} total)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())