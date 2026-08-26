"""Extract raw PET inputs for BW (true_pet.nii.gz from BrainWeb) and HE
(input[...,1] from HECKTOR.npz) for the v3 pipeline subjects.

Outputs:
  D:/MIAProject/SAM_KEM_QWEN3_V3_REPO/data/raw_pet/
    raw_pet_brainweb.npz         - 14 train subjects × sim_0 true_pet
    raw_pet_hecktor.npz          - 156 train subjects × axial_center PET channel
    RAW_PET_MANIFEST.json        - provenance

Layout:
  raw_pet_brainweb.npz:
    <subject>__true_pet          (256, 256, 258) float32 (full BrainWeb volume)
    <subject>__slice_count       scalar int  (number of non-zero slices)
  raw_pet_hecktor.npz:
    <subject>__pet_input         (144, 144) float32   PET channel (HU scale)
    <subject>__target            (144, 144) float32   normalized [0,1] target
"""
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(r"D:/MIAProject/SAM_KEM_QWEN3_V3_REPO")
BW_RAW = Path(r"D:/MIAProject/brainweb/数据/brainweb_petmr")
HE_NPZ = Path(r"D:/MIAProject/HECKTOR.npz")
SPLIT_V3 = Path(r"D:/MIAProject/SAM_KEM_QWEN3_MEDSAM2_DELIVERY/splits/subject_split.csv")
OUT = ROOT / "data" / "raw_pet"
OUT.mkdir(parents=True, exist_ok=True)


def load_split_v3():
    rows = list(csv.DictReader(open(SPLIT_V3, encoding="utf-8")))
    train = [r for r in rows if r["split"] == "train"]
    val = [r for r in rows if r["split"] == "validation"]
    test = [r for r in rows if r["split"] == "test_locked"]
    return train, val, test


def extract_bw():
    """Extract true_pet.nii.gz for all 20 BW subjects (full 256x256x258 volume)."""
    train, val, test = load_split_v3()
    subjects = sorted({r["subject_id"] for r in train + val + test
                       if r["dataset"] == "brainweb"})
    out = {}
    for sid in subjects:
        sim0 = BW_RAW / sid / "sim_0" / "true_pet.nii.gz"
        if not sim0.is_file():
            print(f"  {sid}: no true_pet.nii.gz")
            continue
        import nibabel as nib
        img = nib.load(str(sim0))
        data = img.get_fdata().astype(np.float32)
        n_slices = int(sum(1 for i in range(data.shape[2]) if data[:, :, i].sum() > 0))
        out[f"{sid}__true_pet"] = data
        out[f"{sid}__slice_count"] = np.int32(n_slices)
        print(f"  {sid}: shape={data.shape}, nonzero slices={n_slices}, sum={data.sum():.1f}")
    return out


def extract_he():
    """Extract input[...,1] (PET channel) for all 156 HE train subjects."""
    train, val, test = load_split_v3()
    subjects = sorted({int(r["subject_id"]) for r in train + val + test
                       if r["dataset"] == "hecktor"})
    print(f"HE subjects total: {len(subjects)}")
    z = np.load(HE_NPZ)
    input_arr = z["input"]  # (224, 144, 144, 144, 2) — last axis is channel
    target_arr = z["target"]  # (224, 144, 144, 144)
    print(f"HECKTOR.npz input shape: {input_arr.shape}, dtype: {input_arr.dtype}")
    print(f"HECKTOR.npz target shape: {target_arr.shape}")
    n_total = input_arr.shape[0]
    out = {}
    n_done = 0
    for sid in subjects:
        if sid >= n_total:
            print(f"  subject {sid}: skip (HECKTOR has only {n_total} subjects)")
            continue
        # Use the first axial-center slice (matches v3 split's slice_id='axial_center')
        # The v3 split uses the middle slice of HECKTOR volumes.
        # Looking at HECKTOR.npz shape (224, 144, 144, 144), axis 1 is slice.
        # The v3 split uses slice_id='axial_center' for HE — this is index 72
        # (middle of 144). The existing target.npy in arrays/hecktor/<sid>/
        # axial_center/r0/target.npy was extracted from this.
        slice_idx = 72
        pet_input = input_arr[sid, slice_idx, :, :, 1].astype(np.float32)  # PET channel
        target = target_arr[sid, slice_idx, :, :].astype(np.float32)
        out[f"{sid}__pet_input"] = pet_input
        out[f"{sid}__target"] = target
        n_done += 1
    print(f"  wrote {n_done} HE subjects")
    return out


def main():
    print("\n=== BrainWeb (true_pet.nii.gz, full 256x256x258 volumes) ===")
    bw = extract_bw()
    bw_p = OUT / "raw_pet_brainweb.npz"
    np.savez_compressed(bw_p, **bw)
    print(f"  wrote {bw_p} ({bw_p.stat().st_size/1e6:.1f} MB)")

    print("\n=== HECKTOR (PET channel input[...,1] for train subjects) ===")
    he = extract_he()
    he_p = OUT / "raw_pet_hecktor.npz"
    np.savez_compressed(he_p, **he)
    print(f"  wrote {he_p} ({he_p.stat().st_size/1e6:.1f} MB)")

    manifest = {
        "brainweb": {
            "file": str(bw_p.relative_to(ROOT)),
            "size_mb": round(bw_p.stat().st_size/1e6, 1),
            "n_subjects": len({k.split("__")[0] for k in bw.keys() if "true_pet" in k}),
            "shape": "(256, 256, 258) float32 per subject",
            "source": "D:/MIAProject/brainweb/数据/brainweb_petmr/<subject>/sim_0/true_pet.nii.gz",
        },
        "hecktor": {
            "file": str(he_p.relative_to(ROOT)),
            "size_mb": round(he_p.stat().st_size/1e6, 1),
            "n_subjects": len({k.split("__")[0] for k in he.keys() if "pet_input" in k}),
            "shape": "(144, 144) float32 per subject (PET channel, HU scale)",
            "source": "D:/MIAProject/HECKTOR.npz::input[sid, 72, :, :, 1]",
        },
    }
    manifest_p = OUT / "RAW_PET_MANIFEST.json"
    with open(manifest_p, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    print(f"\nwrote {manifest_p}")


if __name__ == "__main__":
    main()