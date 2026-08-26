"""Build a small end-to-end sample NPZ for each dataset.

For each of brainweb and hecktor:
  - randomly pick 32 slice_ids from the dataset's test_locked split
    (or validation if test_locked is empty)
  - for each slice, include:
      target (128x128), anatomy (128x128) if available,
      sinogram (180x128), lambda (180x128), SAM prior (128x128)
    at count_level=200000, realization=0
  - pack all into one .npz

Output:
  D:/MIAProject/SAM_KEM_QWEN3_V3_REPO/data/samples/sample_brainweb.npz
  D:/MIAProject/SAM_KEM_QWEN3_V3_REPO/data/samples/sample_hecktor.npz

Each file should be small enough to commit (< < 50 MB).
"""
import csv
import random
import sys
from pathlib import Path

import numpy as np

ROOT = Path(r"D:/MIAProject/SAM_KEM_QWEN3_V3_REPO")
SPLIT_V3 = Path(r"D:/MIAProject/SAM_KEM_QWEN3_MEDSAM2_DELIVERY/splits/subject_split.csv")
SPLIT_V4 = Path(r"D:/MIAProject/SAM_KEM_QWEN3_MEDSAM2_DELIVERY/splits/subject_split_v4.csv")
ARRAYS = Path(r"D:/MIAProject/SAM_KEM_QWEN3_MEDSAM2_DELIVERY/arrays")
OUT_DIR = ROOT / "data" / "samples"
OUT_DIR.mkdir(parents=True, exist_ok=True)

N_SLICES = 32
SEED = 20260817
COUNTS = [100_000, 200_000, 500_000, 1_000_000]
R_IDX = 0  # use only r=0 to keep NPZ small


def pick_slices(dataset: str, n: int, rng: random.Random) -> list[tuple[str, str]]:
    """Return [(subject_id, slice_id), ...] for the chosen dataset.

    Strategy:
      - Use subject_split_v4.csv for BrainWeb (has all slices per subject).
      - Use subject_split_v3.csv for HECKTOR (v3 split still uses axial_center).
    """
    if dataset == "brainweb":
        rows = list(csv.DictReader(open(SPLIT_V4, encoding="utf-8")))
        # Prefer validation split (clean, not used for sweep)
        pool = [r for r in rows if r["dataset"] == "brainweb" and r["split"] == "validation"]
        if not pool:
            pool = [r for r in rows if r["dataset"] == "brainweb" and r["split"] == "test_locked"]
    else:
        rows = list(csv.DictReader(open(SPLIT_V3, encoding="utf-8")))
        pool = [r for r in rows if r["dataset"] == "hecktor" and r["split"] == "validation"]
        if not pool:
            pool = [r for r in rows if r["dataset"] == "hecktor" and r["split"] == "test_locked"]
    # Filter to those with all required files present.
    valid = []
    for r in pool:
        sid = r["subject_id"]
        sl = r["slice_id"]
        target_p = ARRAYS / dataset / sid / sl / "r0" / "target.npy"
        # Try nested then flat layout
        sino_p = ARRAYS / dataset / sid / sl / f"r{R_IDX}" / f"c200000" / f"sinogram_count200000.npy"
        if not sino_p.is_file():
            sino_p = ARRAYS / dataset / sid / sl / f"r{R_IDX}" / f"sinogram_count200000.npy"
        if target_p.is_file() and sino_p.is_file():
            valid.append((sid, sl))
    print(f"  {dataset}: pool={len(pool)}, with all files={len(valid)}")
    return rng.sample(valid, min(n, len(valid)))


def collect_case(dataset: str, sid: str, sl: str, counts: list[int]) -> dict:
    """Collect target, anatomy, sinograms, lambdas, prior for one case."""
    sd = ARRAYS / dataset / sid / sl
    out = {
        "subject_id": np.array(sid),
        "slice_id": np.array(sl),
        "target": np.load(sd / "r0" / "target.npy").astype(np.float32),
    }
    anat_p = sd / "r0" / "anatomy.npy"
    if anat_p.is_file():
        out["anatomy"] = np.load(anat_p).astype(np.float32)
    # HE uses flat layout (sinogram_count<N>.npy directly in r<R_IDX>/);
    # BW v4 uses nested layout (r<R_IDX>/c<C>/sinogram_count<N>.npy).
    for c in counts:
        # Try nested first (BW), then flat (HE)
        sino_p = sd / f"r{R_IDX}" / f"c{c}" / f"sinogram_count{c}.npy"
        if not sino_p.is_file():
            sino_p = sd / f"r{R_IDX}" / f"sinogram_count{c}.npy"
        lam_p = sd / f"r{R_IDX}" / f"c{c}" / f"lambda_count{c}.npy"
        if not lam_p.is_file():
            lam_p = sd / f"r{R_IDX}" / f"lambda_count{c}.npy"
        prior_p = sd / f"r{R_IDX}" / f"c{c}" / "sam_prior_medsam2.npy"
        if not prior_p.is_file():
            prior_p = sd / f"r{R_IDX}" / "sam_prior_medsam2.npy"
        if sino_p.is_file():
            out[f"sinogram_{c}"] = np.load(sino_p).astype(np.int32)
        if lam_p.is_file():
            out[f"lambda_{c}"] = np.load(lam_p).astype(np.float32)
        if prior_p.is_file():
            out[f"sam_prior_{c}"] = np.load(prior_p).astype(np.float32)
    return out


def main():
    rng = random.Random(SEED)
    summary = {}
    for ds in ["brainweb", "hecktor"]:
        print(f"\n=== {ds} ===")
        picks = pick_slices(ds, N_SLICES, rng)
        print(f"  picked {len(picks)} slices")
        cases = {}
        for sid, sl in picks:
            key = f"{sid}__{sl}"
            cases[key] = collect_case(ds, sid, sl, COUNTS)
        # Save NPZ
        out_p = OUT_DIR / f"sample_{ds}.npz"
        flat = {}
        for case_id, d in cases.items():
            for k, v in d.items():
                flat[f"{case_id}__{k}"] = v
        np.savez_compressed(out_p, **flat)
        size_mb = out_p.stat().st_size / 1e6
        print(f"  wrote {out_p} ({size_mb:.1f} MB)")
        summary[ds] = {
            "n_cases": len(cases),
            "file_size_mb": round(size_mb, 1),
            "out_path": str(out_p),
            "cases": list(cases.keys()),
        }
    # Write manifest
    import json
    manifest_p = OUT_DIR / "SAMPLE_MANIFEST.json"
    with open(manifest_p, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\nwrote {manifest_p}")


if __name__ == "__main__":
    main()