"""Phase 6b: regenerate BW sinograms for ALL usable slices per subject.

Reads splits/subject_split_v4.csv and generates sinograms for every
(subject, slice, count, realization) using C++ torch_radon at 180 angles.

Usage (from WSL):
    /home/miku/kernelSAM/bin/python scripts/_wsl_regen_bw_all_slices.py
"""
import sys
import csv
import time
from pathlib import Path

import numpy as np
import torch
from torch_radon import ParallelBeam


def load_target(sid, slice_id):
    """Load pre-extracted 128x128 target .npy file."""
    p = Path(r"/mnt/d/MIAProject/SAM_KEM_QWEN3_MEDSAM2_DELIVERY/arrays/brainweb") / sid / slice_id / "r0" / "target.npy"
    if not p.is_file():
        return None
    return np.load(p).squeeze().astype(np.float32)

ROOT = Path(r"/mnt/d/MIAProject/SAM_KEM_QWEN3_MEDSAM2_DELIVERY")
sys.path.insert(0, str(ROOT / "scripts"))
BW_RAW = Path(r"/mnt/d/MIAProject/brainweb/数据/brainweb_petmr")
ARRAYS = ROOT / "arrays" / "brainweb"
SPLIT = ROOT / "splits" / "subject_split_v4.csv"
COUNTS = [100_000, 200_000, 500_000, 1_000_000]
REALIZATIONS = [0, 1, 2]
N_ANGLES = 180
IMG_SIZE = 128
SEED_BASE = 2026_0823
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def seed_for(sid, slice_id, count, r):
    h = abs(hash((sid, slice_id, count, r, "v4_180"))) % (2 ** 31)
    return SEED_BASE + h


def main():
    rows = list(csv.DictReader(open(SPLIT, encoding="utf-8")))
    print(f"split rows: {len(rows)}")

    # Pre-build the radon operator (cached).
    angles = np.linspace(0, np.pi, N_ANGLES, endpoint=False)
    radon = ParallelBeam(IMG_SIZE, angles)
    print(f"torch_radon C++ ParallelBeam: {N_ANGLES} angles, {IMG_SIZE}x{IMG_SIZE}, device={DEVICE}")

    # Pre-load all targets and forward-project once per (subject, slice).
    target_cache: dict[tuple, np.ndarray] = {}
    print("pre-loading and forward-projecting targets...")
    t0 = time.time()
    for r in rows:
        sid = r["subject_id"]
        slice_id = r["slice_id"]
        key = (sid, slice_id)
        if key in target_cache:
            continue
        tgt = load_target(sid, slice_id)
        if tgt is None:
            target_cache[key] = None
            continue
        t = torch.from_numpy(tgt).unsqueeze(0).unsqueeze(0).to(DEVICE)
        proj = radon.forward(t).squeeze().cpu().numpy().astype(np.float64)
        target_cache[key] = (tgt, proj)
    print(f"  {len(target_cache)} unique (subject, slice) targets in {time.time()-t0:.1f}s")

    total = 0
    t0 = time.time()
    for r in rows:
        sid = r["subject_id"]
        slice_id = r["slice_id"]
        cached = target_cache.get((sid, slice_id))
        if cached is None:
            continue
        tgt, proj = cached
        s = float(proj.sum())
        if s <= 0:
            continue
        slice_dir = ARRAYS / sid / slice_id
        slice_dir.mkdir(parents=True, exist_ok=True)
        for r_idx in REALIZATIONS:
            r_subdir = slice_dir / f"r{r_idx}"
            r_subdir.mkdir(parents=True, exist_ok=True)
            for c in COUNTS:
                sino_p = r_subdir / f"c{c}"
                sino_p.mkdir(parents=True, exist_ok=True)
                sino_f = sino_p / f"sinogram_count{c}.npy"
                lam_f = sino_p / f"lambda_count{c}.npy"
                if sino_f.is_file():
                    arr = np.load(sino_f)
                    if arr.shape == (N_ANGLES, IMG_SIZE):
                        continue
                lam = proj * (c / s)
                seed = seed_for(sid, slice_id, c, r_idx)
                rng = np.random.default_rng(seed)
                y = rng.poisson(np.clip(lam, 0, None)).astype(np.int64)
                np.save(sino_f, y)
                np.save(lam_f, lam.astype(np.float64))
                total += 1
    elapsed = time.time() - t0
    print(f"\nWrote {total} new sinograms in {elapsed:.1f}s ({total/max(elapsed,0.1):.1f}/s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())