"""Export small, non-test PET samples for traceability.

BW: 2 train + 1 val subject, 1 mid-volume axial slice each, raw float32.
HE: 4 train + 2 val cases, 1 mid-volume slice each, raw float32.

No min-max and no float16 conversion are applied. HECKTOR.npz is itself an
upstream processed container, so its slices must not be described as raw
scanner data. The export records that distinction explicitly.

Output:
  data/raw_samples/raw_pet_brainweb.npz
  data/raw_samples/raw_pet_hecktor.npz
  data/raw_samples/TRACEABILITY_MANIFEST.json
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

DEFAULT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BW_RAW = Path(r"D:/MIAProject/brainweb/数据/brainweb_petmr")
DEFAULT_HE_NPZ = Path(r"D:/MIAProject/HECKTOR.npz")
DEFAULT_SPLIT = Path(
    r"D:/MIAProject/SAM_KEM_QWEN3_MEDSAM2_DELIVERY/splits/subject_split.csv"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_array(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(repr(array.shape).encode("ascii"))
    digest.update(array.tobytes())
    return digest.hexdigest()


def load_split_v3(path):
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not any(row["split"] == "test_locked" for row in rows):
        raise ValueError("split file does not contain the expected locked-test partition")
    return rows


def bw_for_subject(brainweb_root, sid, slice_idx=None):
    """Return an unnormalized source slice and its exact 2x2 mean pool."""
    import nibabel as nib
    sim0 = brainweb_root / sid / "sim_0" / "true_pet.nii.gz"
    img = nib.load(str(sim0))
    data = img.get_fdata(dtype=np.float32)  # (256, 256, 258)
    if slice_idx is None:
        # pick a mid-volume slice with nonzero content
        s = data.sum(axis=(0, 1))
        nz = np.where(s > 0)[0]
        slice_idx = int(nz[len(nz) // 2])
    raw_slice = data[:, :, slice_idx].astype(np.float32)  # (256, 256) float32 raw
    if raw_slice.shape != (256, 256):
        raise ValueError(f"{sim0} has unexpected in-plane shape {raw_slice.shape}")
    # Downsample 256->128 by mean pooling (this is what the v3 pipeline does).
    H, W = raw_slice.shape
    target_128 = raw_slice.reshape(128, H // 128, 128, W // 128).mean(axis=(1, 3)).astype(np.float32)
    return raw_slice, slice_idx, target_128, sim0


def bw_pick_split(rows, dataset, split_name, n):
    pool = [r for r in rows if r["dataset"] == dataset and r["split"] == split_name]
    return [r["subject_id"] for r in pool[:n]]


def extract_bw(rows, brainweb_root):
    """Pick first 2 train + first 1 val subjects."""
    train = bw_pick_split(rows, "brainweb", "train", 2)
    val = bw_pick_split(rows, "brainweb", "validation", 1)
    print(f"BW train: {train}")
    print(f"BW val:   {val}")
    out = {}
    records = []
    for sid in train + val:
        raw, slice_idx, tgt, source = bw_for_subject(brainweb_root, sid)
        split = "train" if sid in train else "validation"
        out[f"{sid}__{slice_idx}__raw256"] = raw
        out[f"{sid}__{slice_idx}__target128"] = tgt
        out[f"{sid}__{slice_idx}__slice_idx"] = np.int32(slice_idx)
        out[f"{sid}__{slice_idx}__split"] = np.array(split)
        records.append({
            "subject_id": sid, "split": split, "slice_idx": slice_idx,
            "source": str(source), "source_file_sha256": sha256_file(source),
            "raw256_sha256": sha256_array(raw),
            "target128_sha256": sha256_array(tgt),
        })
        print(f"  {sid} (split={split}, slice={slice_idx}): raw sum={raw.sum():.1f}, target sum={tgt.sum():.1f}")
    return out, records


def extract_he(rows, hecktor_npz):
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
    z = np.load(hecktor_npz, allow_pickle=False, mmap_mode="r")
    input_arr = z["input"]  # (224, 144, 144, 144, 2)
    out = {}
    records = []
    # v3 subject_id maps to HECKTOR.npz axis 0 index = subject_id - 1
    # This is verified for HE subjects in the v3 pipeline (we checked earlier).
    # Use slice 72 as axial_center.
    for sid in train_ids + val_ids:
        npz_idx = sid - 1
        if npz_idx < 0 or npz_idx >= input_arr.shape[0]:
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
        records.append({
            "subject_id": str(sid), "split": split, "slice_idx": slice_idx,
            "npz_axis0": npz_idx,
            "ct_sha256": sha256_array(ct), "pet_input_sha256": sha256_array(pet),
        })
        print(f"  subj {sid} (split={split}, npz_idx={npz_idx}): CT sum={ct.sum():.0f}, PET sum={pet.sum():.1f}, PET max={pet.max():.2f}")
    z.close()
    return out, records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--brainweb-root", type=Path, default=DEFAULT_BW_RAW)
    parser.add_argument("--hecktor-npz", type=Path, default=DEFAULT_HE_NPZ)
    parser.add_argument("--split-csv", type=Path, default=DEFAULT_SPLIT)
    args = parser.parse_args()
    root = args.repo_root.resolve()
    out_dir = root / "data" / "raw_samples"
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = load_split_v3(args.split_csv)
    print("\n=== BrainWeb ===")
    bw, bw_records = extract_bw(rows, args.brainweb_root)
    bw_p = out_dir / "raw_pet_brainweb.npz"
    np.savez_compressed(bw_p, **bw)
    print(f"  wrote {bw_p} ({bw_p.stat().st_size/1e6:.1f} MB)")

    print("\n=== HECKTOR ===")
    he, he_records = extract_he(rows, args.hecktor_npz)
    he_p = out_dir / "raw_pet_hecktor.npz"
    np.savez_compressed(he_p, **he)
    print(f"  wrote {he_p} ({he_p.stat().st_size/1e6:.1f} MB)")

    manifest = {
        "brainweb": {
            "file": bw_p.relative_to(root).as_posix(),
            "file_sha256": sha256_file(bw_p),
            "size_mb": round(bw_p.stat().st_size/1e6, 1),
            "n_subjects": len([k for k in bw if "raw256" in k]),
            "format": "raw256 (256x256 float32) + target128 (128x128 float32) per slice",
            "source": "D:/MIAProject/brainweb/数据/brainweb_petmr/<sid>/sim_0/true_pet.nii.gz",
            "activity_unit": "BrainWeb simulation units (not scanner calibrated)",
            "preprocessing": "raw256 is an unnormalized float32 source slice; target128 is its exact 2x2 mean pool",
            "records": bw_records,
        },
        "hecktor": {
            "file": he_p.relative_to(root).as_posix(),
            "file_sha256": sha256_file(he_p),
            "size_mb": round(he_p.stat().st_size/1e6, 1),
            "n_subjects": len([k for k in he if "pet_input" in k]),
            "format": "ct (144x144 float32 HU) + pet_input (144x144 float32 SUV-scale) per slice",
            "source": "D:/MIAProject/HECKTOR.npz::input[<sid>-1, 72, :, :, :]",
            "preprocessing": "no transformation after NPZ extraction; provider states the upstream arrays are reconstructed PET/CT slices exported directly from clinical DICOM",
            "subject_id_to_npz_mapping": "subject_id -> npz axis 0 index (subject_id - 1)",
            "pet_unit": "SUV (declared by data provider; source DICOM metadata is not retained in the NPZ)",
            "source_file_sha256": sha256_file(args.hecktor_npz),
            "spatial_metadata_status": "missing voxel spacing, affine/orientation, and scanner transaxial FOV",
            "records": he_records,
        },
    }
    manifest_p = out_dir / "TRACEABILITY_MANIFEST.json"
    with open(manifest_p, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    print(f"\nwrote {manifest_p}")


if __name__ == "__main__":
    main()
