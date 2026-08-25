"""Phase 6: redesign BW split with ALL usable slices per subject.

v3 used only 1 axial_center slice per BW subject, losing 93% of the
available training data. This script creates a new split file with
all usable slices per subject (172-181 slices each, ~2457 train slices
× 4 counts × 3 realizations ≈ 29492 training samples).

Output: splits/subject_split_v4.csv
Backup: splits/subject_split_v3_AUDIT.csv
"""
from __future__ import annotations
import csv
import shutil
from pathlib import Path

import nibabel as nib
import numpy as np

ROOT = Path(r"D:/MIAProject/SAM_KEM_QWEN3_MEDSAM2_DELIVERY")
BW_RAW = Path(r"D:/MIAProject/brainweb/数据/brainweb_petmr")
SPLIT_V3 = ROOT / "splits" / "subject_split.csv"
SPLIT_V4 = ROOT / "splits" / "subject_split_v4.csv"
BACKUP = ROOT / "splits" / "subject_split_v3_AUDIT.csv"

# Keep the same train/validation/test_locked split as v3.
SPLIT_MAP = {
    "subject04": "train", "subject05": "train", "subject06": "train",
    "subject18": "train", "subject20": "train", "subject38": "train",
    "subject44": "train", "subject45": "train", "subject49": "train",
    "subject50": "train", "subject51": "train", "subject52": "train",
    "subject53": "train", "subject54": "train",
    "subject41": "validation", "subject42": "validation", "subject43": "validation",
    "subject46": "test_locked", "subject47": "test_locked", "subject48": "test_locked",
}


def main():
    # Backup v3 split.
    if BACKUP.is_file():
        print(f"backup already exists: {BACKUP}")
    else:
        shutil.copy2(SPLIT_V3, BACKUP)
        print(f"backed up v3 split to: {BACKUP}")

    rows = []
    for sid, split in SPLIT_MAP.items():
        sim0 = BW_RAW / sid / "sim_0"
        true_pet_p = sim0 / "true_pet.nii.gz"
        if not true_pet_p.is_file():
            print(f"  WARNING: {sid} no true_pet.nii.gz; skip")
            continue
        img = nib.load(str(true_pet_p))
        data = img.get_fdata()
        usable = [i for i in range(data.shape[2]) if data[:,:,i].sum() > 0]
        for slice_idx in usable:
            rows.append({
                "dataset": "brainweb",
                "subject_id": sid,
                "source_subject_id": sid,
                "split": split,
                "included": "yes",
                "n_slices": len(usable),
                "slice_id": f"axial_z{slice_idx:03d}",
                "split_locked_utc": "2026-08-24T00:00:00Z",
                "source_manifest_sha256": "TBD_BW_RAW_SHA",
                "split_script_sha256": "TBD_THIS_SCRIPT_SHA",
                "notes": f"raw_nii_slice_{slice_idx}",
            })

    fields = list(rows[0].keys())
    with open(SPLIT_V4, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    n_train = sum(1 for r in rows if r["split"] == "train")
    n_val = sum(1 for r in rows if r["split"] == "validation")
    n_test = sum(1 for r in rows if r["split"] == "test_locked")
    print(f"wrote {len(rows)} BW rows to: {SPLIT_V4}")
    print(f"  train: {n_train} slices")
    print(f"  validation: {n_val} slices")
    print(f"  test_locked: {n_test} slices")
    print(f"  with 4 counts × 3 realizations:")
    print(f"    train samples: {n_train * 12}")
    print(f"    validation samples: {n_val * 12}")
    print(f"    test samples: {n_test * 12}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())