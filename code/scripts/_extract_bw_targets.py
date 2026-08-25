"""Extract all BW targets from raw nii.gz and save as 128x128 .npy files.

For every (subject, slice_idx) in the new split, extract the 128x128
normalized target and save to:
    arrays/brainweb/<subject>/<slice_id>/r0/target.npy

This runs in Windows (has nibabel). The WSL pipeline reads these .npy
files directly without needing nibabel.
"""
from __future__ import annotations
import csv
import sys
from pathlib import Path

import nibabel as nib
import numpy as np

ROOT = Path(r"D:/MIAProject/SAM_KEM_QWEN3_MEDSAM2_DELIVERY")
BW_RAW = Path(r"D:/MIAProject/brainweb/数据/brainweb_petmr")
ARRAYS = ROOT / "arrays" / "brainweb"
SPLIT = ROOT / "splits" / "subject_split_v4.csv"
IMG_SIZE = 128


def main():
    rows = list(csv.DictReader(open(SPLIT, encoding="utf-8")))
    print(f"split rows: {len(rows)}")
    n_done = 0
    n_skip = 0
    n_err = 0
    for r in rows:
        sid = r["subject_id"]
        slice_id = r["slice_id"]
        slice_idx = int(slice_id.replace("axial_z", ""))
        tgt_p = ARRAYS / sid / slice_id / "r0" / "target.npy"
        if tgt_p.is_file():
            n_skip += 1
            continue
        p = BW_RAW / sid / "sim_0" / "true_pet.nii.gz"
        if not p.is_file():
            n_err += 1
            continue
        img = nib.load(str(p))
        data = img.get_fdata()[:, :, slice_idx].astype(np.float64)
        # Downsample 256->128 with mean pooling.
        h, w = data.shape
        if h != IMG_SIZE or w != IMG_SIZE:
            ds = data.reshape(IMG_SIZE, h // IMG_SIZE, IMG_SIZE, w // IMG_SIZE).mean(axis=(1, 3))
        else:
            ds = data
        mx = ds.max()
        if mx <= 0:
            n_err += 1
            continue
        ds = ds / mx
        tgt_p.parent.mkdir(parents=True, exist_ok=True)
        np.save(tgt_p, ds.astype(np.float32))
        n_done += 1
        if n_done % 500 == 0:
            print(f"  {n_done} done...")
    print(f"\nWrote {n_done} targets, skipped {n_skip} existing, errors {n_err}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())