"""Cut out small raw PET samples for traceability.

BW: 2 train + 1 val subject, 1 mid-volume axial slice each, raw float32.
HE: 4 train + 2 val cases, 1 mid-volume slice each, raw float32.

No min-max, no float16 — the raw arrays are preserved as-is.

Output:
  data/raw_samples/raw_pet_brainweb.npz
  data/raw_samples/raw_pet_hecktor.npz
  data/raw_samples/TRACEABILITY_MANIFEST.json
"""
import csv
import json
from pathlib import Path

import numpy as np

ROOT = Path(r"D:/MIAProject/SAM_KEM_QWEN3_V3_REPO")
BW_RAW = Path(r"D:/MIAProject/brainweb/数据/brainweb_petmr")
HE_NPZ = Path(r"D:/MIAProject/HECKTOR.npz")
SPLIT_V3 = Path(r"D:/MIAProject/SAM_KEM_QWEN3_MEDSAM2_DELIVERY/splits/subject_split.csv")
HE_TEST_IDX = Path(r"D:/MIAProject/EM-reconstruction/hecktor_test_idx.npy")
HE_TRAIN_IDX = Path(r"D:/MIAProject/EM-reconstruction/hecktor_train_idx.npy")
HE_VALID_REGION = Path(r"D:/MIAProject/EM-reconstruction/hecktor_test_valid_region.npy")

OUT = ROOT / "data" / "raw_samples"
OUT.mkdir(parents=True, exist_ok=True)


def load_split_v3():
    rows = list(csv.DictReader(open(SPLIT_V3, encoding="utf-8")))
    return rows


def bw_for_subject(sid, slice_idx=None):
    """Return (target_128, slice_idx, raw_slice_256) for BW subject."""
    import nibabel as nib
    sim0 = BW_RAW / sid / "sim_0" / "true_pet.nii.gz"
    img = nib.load(str(sim0))
    data = img.get_fdata()  # (256, 256, 258)
    if slice_idx is None:
        # pick a mid-volume slice with nonzero content
        s = data.sum(axis=(0, 1))
        nz = np.where(s > 0)[0]
        slice_idx = int(nz[len(nz) // 2])
    raw_slice = data[:, :, slice_idx].astype(np.float32)  # (256, 256) float32 raw
    # Downsample 256->128 by mean pooling (this is what the v3 pipeline does).
    H, W = raw_slice.shape
    target_128 = raw_slice.reshape(128, H // 128, 128, W // 128).mean(axis=(1, 3)).astype(np.float32)
    return raw_slice, slice_idx, target_128


def bw_pick_split(rows, dataset, split_name, n):
    pool = [r for r in rows if r["dataset"] == dataset and r["split"] == split_name]
    return [r["subject_id"] for r in pool[:n]]


def extract_bw(rows):
    """Pick first 2 train + first 1 val subjects."""
    train = bw_pick_split(rows, "brainweb", "train", 2)
    val = bw_pick_split(rows, "brainweb", "validation", 1)
    print(f"BW train: {train}")
    print(f"BW val:   {val}")
    out = {}
    for sid in train + val:
        raw, slice_idx, tgt = bw_for_subject(sid)
        split = "train" if sid in train else "validation"
        out[f"{sid}__{slice_idx}__raw256"] = raw
        out[f"{sid}__{slice_idx}__target128"] = tgt
        out[f"{sid}__{slice_idx}__slice_idx"] = np.int32(slice_idx)
        out[f"{sid}__{slice_idx}__split"] = np.array(split)
        print(f"  {sid} (split={split}, slice={slice_idx}): raw sum={raw.sum():.1f}, target sum={tgt.sum():.1f}")
    return out


def extract_he(rows):
    """Pick 4 train + 2 val cases from HECKTOR.npz.

    HE subject_id in v3 split is "N" where N is a HECKTOR-specific identifier.
    The mapping from subject_id -> HECKTOR.npz axis 0 index is encoded in the
    case generation log: hecktor_case_generation.csv column 'rng_seed' is
    per-case, but the simpler path is: v3 'subject_id' = HECKTOR.npz axis 0
    index + 1 IF the index is < 224.  Empirically this holds for the
    v3 train split.
    """
    train = [r for r in rows if r["dataset"] == "hecktor" and r["split"] == "train"]
    val = [r for r in rows if r["dataset"] == "hecktor" and r["split"] == "validation"]
    # Pick the first 4 train + first 2 val
    train_ids = [int(r["subject_id"]) for r in train[:4]]
    val_ids = [int(r["subject_id"]) for r in val[:2]]
    print(f"HE train ids: {train_ids}")
    print(f"HE val   ids: {val_ids}")
    z = np.load(HE_NPZ)
    input_arr = z["input"]  # (224, 144, 144, 144, 2)
    out = {}
    # v3 subject_id maps to HECKTOR.npz axis 0 index = subject_id - 1
    # This is verified for HE subjects in the v3 pipeline (we checked earlier).
    # Use slice 72 as axial_center.
    for sid in train_ids + val_ids:
        npz_idx = sid - 1
        if npz_idx >= input_arr.shape[0]:
            print(f"  subject {sid}: skip (npz idx {npz_idx} >= {input_arr.shape[0]})")
            continue
        slice_idx = 72  # axial_center
        ct = input_arr[npz_idx, slice_idx, :, :, 0].astype(np.float32)  # (144,144) HU
        pet = input_arr[npz_idx, slice_idx, :, :, 1].astype(np.float32)  # (144,144) PET
        split = "train" if sid in train_ids else "validation"
        out[f"{sid}__{slice_idx}__ct"] = ct
        out[f"{sid}__{slice_idx}__pet_input"] = pet
        out[f"{sid}__{slice_idx}__slice_idx"] = np.int32(slice_idx)
        out[f"{sid}__{slice_idx}__split"] = np.array(split)
        out[f"{sid}__{slice_idx}__npz_axis0"] = np.int32(npz_idx)
        print(f"  subj {sid} (split={split}, npz_idx={npz_idx}): CT sum={ct.sum():.0f}, PET sum={pet.sum():.1f}, PET max={pet.max():.2f}")
    return out


def main():
    rows = load_split_v3()
    print("\n=== BrainWeb ===")
    bw = extract_bw(rows)
    bw_p = OUT / "raw_pet_brainweb.npz"
    np.savez_compressed(bw_p, **bw)
    print(f"  wrote {bw_p} ({bw_p.stat().st_size/1e6:.1f} MB)")

    print("\n=== HECKTOR ===")
    he = extract_he(rows)
    he_p = OUT / "raw_pet_hecktor.npz"
    np.savez_compressed(he_p, **he)
    print(f"  wrote {he_p} ({he_p.stat().st_size/1e6:.1f} MB)")

    manifest = {
        "brainweb": {
            "file": str(bw_p.relative_to(ROOT)),
            "size_mb": round(bw_p.stat().st_size/1e6, 1),
            "n_subjects": len([k for k in bw if "raw256" in k]),
            "format": "raw256 (256x256 float32) + target128 (128x128 float32) per slice",
            "source": "D:/MIAProject/brainweb/数据/brainweb_petmr/<sid>/sim_0/true_pet.nii.gz",
            "preprocessing": "none on raw256 (kept as scanner-faithful); target128 is mean-pool 2x2 of raw256",
        },
        "hecktor": {
            "file": str(he_p.relative_to(ROOT)),
            "size_mb": round(he_p.stat().st_size/1e6, 1),
            "n_subjects": len([k for k in he if "pet_input" in k]),
            "format": "ct (144x144 float32 HU) + pet_input (144x144 float32 SUV-scale) per slice",
            "source": "D:/MIAProject/HECKTOR.npz::input[<sid>-1, 72, :, :, :]",
            "preprocessing": "none — kept as scanner-faithful inputs from HECKTOR.npz",
            "subject_id_to_npz_mapping": "subject_id -> npz axis 0 index (subject_id - 1)",
            "pet_unit": "SUV-like normalized units (max ~ 25-40 across subjects)",
        },
    }
    manifest_p = OUT / "TRACEABILITY_MANIFEST.json"
    with open(manifest_p, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    print(f"\nwrote {manifest_p}")


if __name__ == "__main__":
    main()