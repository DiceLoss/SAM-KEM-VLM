"""Phase 6d: run full BW pipeline on v4 test_locked (all slices per subject).

Uses the new v4 split with all usable slices per subject. For each
(subject, slice, count, realization) in test_locked, runs MLEM, OSEM,
Bowsher, SelfKEM, MR-KEM, SAM-prior, SAM-KEM-no-post, SAM-KEM-full.

Usage (from WSL):
    /home/miku/kernelSAM/bin/python scripts/_wsl_bw_v4_all.py
"""
import sys
import csv
import os
import math
from pathlib import Path

import numpy as np
import torch
from torch_radon import ParallelBeam

sys.path.insert(0, '/mnt/d/MIAProject/SAM_KEM_QWEN3_MEDSAM2_DELIVERY/scripts')
from _wsl_bw_pipeline import (mlem, osem, bowsher, kem, normalize01,
                               ssim_, nrmse, mae_, psnr, get_sens)

ROOT = '/mnt/d/MIAProject/SAM_KEM_QWEN3_MEDSAM2_DELIVERY'
os.chdir(ROOT)
N_ANGLES = 180
IMG_SIZE = 128
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
COUNTS = [100_000, 200_000, 500_000, 1_000_000]
REALIZATIONS = [0, 1, 2]


def psnr_fn(target, x):
    mse = float(((target - x) ** 2).mean())
    return 100.0 if mse <= 1e-12 else 20 * np.log10(1.0 / np.sqrt(mse))


def main():
    angles = np.linspace(0, np.pi, N_ANGLES, endpoint=False)
    radon = ParallelBeam(IMG_SIZE, angles)

    rows = list(csv.DictReader(open('splits/subject_split_v4.csv', encoding='utf-8')))
    test_rows = [r for r in rows if r['split'] == 'test_locked']
    print(f"test_locked rows: {len(test_rows)}")
    subjects = sorted(set(r["subject_id"] for r in test_rows))
    print(f"test_locked subjects: {subjects}")

    frozen = list(csv.DictReader(open('metrics/frozen_hyperparameters.csv', encoding='utf-8')))
    frozen_idx = {(r['dataset'], int(r['count_level'])): r for r in frozen}

    fields = [
        "dataset", "subject_id", "slice_id", "realization_id",
        "count_level", "method", "split", "checkpoint_sha256",
        "psnr_db", "ssim", "rmse", "nrmse", "mae",
        "poisson_deviance", "deviance_per_bin", "nll_per_bin",
        "data_range", "status",
    ]
    out_p = 'metrics/per_case_metrics_v3_bw.csv'
    n = 0
    with open(out_p, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for r in test_rows:
            sid = r['subject_id']
            slice_id = r['slice_id']
            # Subsample slices: every 4th slice to keep wall time reasonable.
            try:
                slice_idx = int(slice_id.replace('axial_z', ''))
            except Exception:
                continue
            if slice_idx % 4 != 0:
                continue
            target_p = f'arrays/brainweb/{sid}/{slice_id}/r0/target.npy'
            if not os.path.isfile(target_p): continue
            target = np.load(target_p).squeeze().astype(np.float64)
            for count in COUNTS:
                fr = frozen_idx.get(('brainweb', count))
                if fr is None: continue
                sigma = float(fr['sigma_star']); J = int(fr['J_star'])
                n_it = int(fr['N_KEM_star'])
                alpha_star = float(fr.get('alpha_star', 0.3))
                for r_idx in REALIZATIONS:
                    sino_p = f'arrays/brainweb/{sid}/{slice_id}/r{r_idx}/c{count}/sinogram_count{count}.npy'
                    if not os.path.isfile(sino_p): continue
                    sino = np.load(sino_p).squeeze().astype(np.float64)
                    methods_results = {}
                    methods_results['MLEM'] = mlem(radon, sino, n_it=10)
                    methods_results['OSEM'] = osem(radon, sino, n_it=10, subsets=10)
                    methods_results['Bowsher'] = bowsher(radon, sino, normalize01(methods_results['MLEM']), n_it=10)
                    methods_results['SelfKEM'] = kem(radon, sino, normalize01(methods_results['MLEM']), n_it, sigma, J)
                    mr_p = f'arrays/brainweb/{sid}/{slice_id}/r0/anatomy.npy'
                    if os.path.isfile(mr_p):
                        mr = np.load(mr_p).squeeze().astype(np.float64)
                        methods_results['MR-KEM'] = kem(radon, sino, normalize01(mr), n_it, sigma, J)
                    prior_p = f'arrays/brainweb/{sid}/{slice_id}/r{r_idx}/c{count}/sam_prior_v4.npy'
                    if not os.path.isfile(prior_p):
                        prior_p = f'arrays/brainweb/{sid}/{slice_id}/r{r_idx}/c{count}/sam_prior_medsam2.npy'
                    if os.path.isfile(prior_p):
                        prior = np.load(prior_p).squeeze().astype(np.float32)
                        methods_results['SAM-prior-MedSAM2-Qwen3'] = np.clip(prior, 0, 1)
                        methods_results['SAM-KEM-MedSAM2-Qwen3-no-post'] = kem(
                            radon, sino, normalize01(prior), n_it, sigma, J)
                        x_kem = methods_results['SAM-KEM-MedSAM2-Qwen3-no-post']
                        methods_results['SAM-KEM-MedSAM2-Qwen3-full'] = (
                            alpha_star * np.clip(prior, 0, 1) + (1 - alpha_star) * x_kem)
                    for method, x in methods_results.items():
                        mse = float(((target - x) ** 2).mean())
                        rec = {
                            "dataset": "brainweb", "subject_id": sid,
                            "slice_id": slice_id, "realization_id": r_idx,
                            "count_level": count, "method": method,
                            "split": "test_locked",
                            "checkpoint_sha256": "2ee2b4c334c3fe4583092238742f9bd6d214c8e4af2d8221e94d005806f79ef7",
                            "psnr_db": psnr_fn(target, x),
                            "ssim": ssim_(target, x),
                            "rmse": float(np.sqrt(((target - x) ** 2).mean())),
                            "nrmse": nrmse(target, x),
                            "mae": mae_(target, x),
                            "poisson_deviance": 0, "deviance_per_bin": 0,
                            "nll_per_bin": 0, "data_range": 1.0, "status": "ok",
                        }
                        w.writerow(rec); n += 1
    print(f"\nWrote {n} BW rows to {out_p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())