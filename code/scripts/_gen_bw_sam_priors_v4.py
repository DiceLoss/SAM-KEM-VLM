"""Phase 6c: generate BW SAM priors for ALL slices using MedSAM2 decoder.

For every (subject, slice, count, realization) in the new v4 split,
run the BW MedSAM2 decoder to produce the SAM prior. Saves to:
    arrays/brainweb/<subject>/<slice_id>/r<r>/c<count>/sam_prior_medsam2.npy

Runs in Windows (has MedSAM2 / sam2 package + torch 2.10 + CUDA 13.1).
"""
from __future__ import annotations
import csv
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(r"D:/MIAProject/SAM_KEM_QWEN3_MEDSAM2_DELIVERY")
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, r"D:/MIAProject/EM-reconstruction")

from discrete_radon import build_pair  # noqa: E402
from recon import mlem  # noqa: E402
import medsam2_decoder_infer  # noqa: E402

ARRAYS = ROOT / "arrays" / "brainweb"
SPLIT = ROOT / "splits" / "subject_split_v4.csv"
COUNTS = [100_000, 200_000, 500_000, 1_000_000]
REALIZATIONS = [0, 1, 2]
N_ANGLES = 180
IMG_SIZE = 128


def fallback_box(sid, slice_id, count):
    sino_p = ARRAYS / sid / slice_id / "r0" / f"c{count}" / f"sinogram_count{count}.npy"
    if not sino_p.is_file():
        return (48, 48, 80, 80)
    sino = np.load(sino_p).squeeze().astype(np.float64)
    fwd, bwd, _ = build_pair(size=IMG_SIZE, n_angles=N_ANGLES)
    init = mlem(sino, iterations=10, save_iters=(10,), forward=fwd, backward=bwd)
    img = init[10]
    mx = img.max()
    if mx <= 0: return (48, 48, 80, 80)
    mask = img > 0.5 * mx
    if not mask.any(): return (48, 48, 80, 80)
    ys, xs = np.where(mask)
    cx, cy = int(np.mean(xs)), int(np.mean(ys))
    return (max(0, cx - 16), max(0, cy - 16), min(IMG_SIZE, cx + 16), min(IMG_SIZE, cy + 16))


def main():
    # Load BW decoder.
    bw_ckpt = ROOT / "checkpoints" / "medsam2_decoder_brainweb.pt"
    print(f"Loading BW decoder: {bw_ckpt}")
    medsam2_decoder_infer._ensure_model(ckpt_path=bw_ckpt, image_size=1024)

    rows = list(csv.DictReader(open(SPLIT, encoding="utf-8")))
    print(f"split rows: {len(rows)}")

    # Only generate priors for the v4 test_locked + validation sets (to
    # keep wall time reasonable). Training subjects don't need priors
    # since they were already used for decoder training.
    test_rows = [r for r in rows if r["split"] in ("test_locked", "validation")]
    print(f"processing {len(test_rows)} rows (test_locked + validation)")

    fwd, bwd, _ = build_pair(size=IMG_SIZE, n_angles=N_ANGLES)
    n_done = 0
    n_skip = 0
    n_err = 0
    t_start = time.time()
    for r in test_rows:
        sid = r["subject_id"]
        slice_id = r["slice_id"]
        for count in COUNTS:
            box = fallback_box(sid, slice_id, count)
            for r_idx in REALIZATIONS:
                prior_p = ARRAYS / sid / slice_id / f"r{r_idx}" / f"c{count}" / "sam_prior_medsam2.npy"
                if prior_p.is_file():
                    n_skip += 1
                    continue
                sino_p = ARRAYS / sid / slice_id / f"r{r_idx}" / f"c{count}" / f"sinogram_count{count}.npy"
                if not sino_p.is_file():
                    n_err += 1
                    continue
                sino = np.load(sino_p).squeeze().astype(np.float64)
                init = mlem(sino, iterations=10, save_iters=(10,), forward=fwd, backward=bwd)
                pet = np.clip(init[10], 0, 1).astype(np.float32)
                try:
                    prior = medsam2_decoder_infer.predict_prior(pet, box=box, ckpt_path=bw_ckpt)
                except Exception as e:
                    print(f"  inference error {sid}/{slice_id}/r{r_idx}/c{count}: {e}")
                    n_err += 1
                    continue
                prior_p.parent.mkdir(parents=True, exist_ok=True)
                np.save(prior_p, prior.astype(np.float32))
                n_done += 1
        if n_done % 50 == 0:
            elapsed = time.time() - t_start
            print(f"  {n_done} priors in {elapsed:.0f}s ({n_done/max(elapsed,1):.2f}/s)")
    print(f"\nTotal: {n_done} priors, {n_skip} skipped, {n_err} errors in {time.time()-t_start:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())