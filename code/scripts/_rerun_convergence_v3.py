"""Phase 7: convergence driver for the NEW test_locked set.

Saves KEM iterates at {0, 1, 5, 10, 20, 40, 60, 80, 100, 120} for the 5
KEM-family methods on every (subject, realization, count) in
test_locked.

Methods:
  - SelfKEM
  - MR-KEM (BW only)
  - CT-KEM (HE only)
  - SAM-KEM-no-post (uses MedSAM2 anchor box; fallback if no anchor)
  - SAM-KEM-full (uses MedSAM2 anchor box + alpha* fusion)

Per-row metrics: PSNR / SSIM / NRMSE / MAE at each saved iteration.

Output: metrics/convergence_v3.csv

Expected size: 5 methods × 10 iterations × 37 subjects × 3 realizations
               × 4 counts = 22200 rows.

NOTE: This script uses MedSAM2 only for SAM-KEM methods. For SelfKEM /
MR-KEM / CT-KEM the kernel feature is mlem_init / anatomy and the runs
are pure CPU recon.py::kem calls.
"""
from __future__ import annotations

import csv
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(r"D:/MIAProject/SAM_KEM_QWEN3_MEDSAM2_DELIVERY")
sys.path.insert(0, str(ROOT / "scripts"))
from discrete_radon import build_pair  # noqa: E402
from _hybrid_mlem import get_operator_pair  # noqa: E402
from recon import kem, mlem, post_fusion, normalize01  # noqa: E402

SPLITS = ROOT / "splits" / "subject_split.csv"
FROZEN = ROOT / "metrics" / "frozen_hyperparameters.csv"
OUT = ROOT / "metrics" / "convergence_v3.csv"

COUNTS = [100_000, 200_000, 500_000, 1_000_000]
REALIZATIONS = [0, 1, 2]
ITERS = (0, 1, 5, 10, 20, 40, 60, 80, 100, 120)
IMG_SIZE = 128
N_ANGLES = 180


def psnr(target, x):
    mse = float(((target - x) ** 2).mean())
    return 100.0 if mse <= 1e-12 else 20 * math.log10(1.0 / math.sqrt(mse))


def ssim(target, x):
    from skimage.metrics import structural_similarity
    a = target.astype(np.float32); b = np.clip(x, 0, None).astype(np.float32)
    if b.max() > 0: b = b / b.max()
    return float(structural_similarity(a, b, data_range=max(a.max(), 1e-6)))


def nrmse(target, x):
    t = target.flatten().astype(np.float64)
    return float(np.linalg.norm((target - x).flatten()) / max(float(np.linalg.norm(t)), 1e-12))


def load_target(ds, sid):
    p = ROOT / "arrays" / ds / sid / "axial_center" / "r0" / "target.npy"
    if not p.is_file(): return None
    return np.load(p).squeeze().astype(np.float64)


def load_sino(ds, sid, count, r):
    """Load sinogram + mul_factor + pet_noise. Default mul=1, noise=0."""
    rdir = ROOT / "arrays" / ds / sid / "axial_center" / f"r{r}"
    y_p = rdir / f"sinogram_count{count}.npy"
    m_p = rdir / f"mul_factor_count{count}.npy"
    n_p = rdir / f"pet_noise_count{count}.npy"
    if not y_p.is_file():
        return None
    y = np.load(y_p).squeeze().astype(np.float64)
    if m_p.is_file():
        m = np.load(m_p).squeeze().astype(np.float64)
    else:
        m = np.ones_like(y)
    if n_p.is_file():
        n = np.load(n_p).squeeze().astype(np.float64)
    else:
        n = np.zeros_like(y)
    return y, m, n


def main() -> int:
    rows = list(csv.DictReader(open(SPLITS, encoding="utf-8")))
    test = [r for r in rows if r["split"] == "test_locked"]
    test_bw = sorted(set(r["subject_id"] for r in test if r["dataset"] == "brainweb"))
    test_he = sorted(set(int(r["subject_id"]) for r in test if r["dataset"] == "hecktor"))
    frozen_rows = list(csv.DictReader(open(FROZEN, encoding="utf-8")))
    frozen_idx = {(r["dataset"], int(r["count_level"])): r for r in frozen_rows}

    # Use the hybrid operator (torch_radon for BW, discrete_radon for HE).
    fwd_bw, bwd_bw = get_operator_pair("brainweb")
    fwd_he, bwd_he = get_operator_pair("hecktor")

    fields = [
        "dataset", "subject_id", "slice_id", "realization_id",
        "count_level", "method", "iteration", "split",
        "psnr_db", "ssim", "nrmse", "mae",
    ]
    OUT.parent.mkdir(exist_ok=True)
    n_written = 0
    with open(OUT, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields); w.writeheader()
        for ds, subs, (fwd, bwd) in [
            ("brainweb", test_bw, (fwd_bw, bwd_bw)),
            ("hecktor", test_he, (fwd_he, bwd_he)),
        ]:
            methods = ["SelfKEM"]
            if ds == "brainweb": methods.append("MR-KEM")
            else: methods.append("CT-KEM")
            methods += ["SAM-KEM-MedSAM2-Qwen3-no-post",
                        "SAM-KEM-MedSAM2-Qwen3-full"]
            for sid in subs:
                target = load_target(ds, str(sid))
                if target is None: continue
                for count in COUNTS:
                    fr = frozen_idx.get((ds, count))
                    if fr is None: continue
                    sigma = float(fr["sigma_star"]); J = int(fr["J_star"])
                    n_max = max(ITERS)
                    for r in REALIZATIONS:
                        sino = load_sino(ds, str(sid), count, r)
                        if sino is None: continue
                        y, mul, noi = sino
                        # MLEM init with the correct operator for this dataset.
                        init = mlem(y, iterations=10, save_iters=(10,),
                                    forward=fwd, backward=bwd,
                                    mul_factor=mul, pet_noise=noi)
                        mlem_init = init[10]
                        for method in methods:
                            if method == "SelfKEM":
                                feat = normalize01(mlem_init)
                                out_dict = kem(y, feat, iterations=n_max,
                                               sigma=sigma, J=J,
                                               save_iters=ITERS,
                                               forward=fwd, backward=bwd,
                                               mul_factor=mul, pet_noise=noi)
                            elif method == "MR-KEM":
                                anat = None
                                for name in ("mr.npy", "anatomy.npy"):
                                    ap = ROOT / "arrays" / ds / str(sid) / "axial_center" / "r0" / name
                                    if ap.is_file():
                                        anat = np.load(ap).squeeze().astype(np.float64); break
                                if anat is None: continue
                                feat = normalize01(anat)
                                out_dict = kem(y, feat, iterations=n_max,
                                               sigma=sigma, J=J,
                                               save_iters=ITERS,
                                               forward=fwd, backward=bwd,
                                               mul_factor=mul, pet_noise=noi)
                            elif method == "CT-KEM":
                                anat = None
                                for name in ("ct.npy", "anatomy.npy"):
                                    ap = ROOT / "arrays" / ds / str(sid) / "axial_center" / "r0" / name
                                    if ap.is_file():
                                        anat = np.load(ap).squeeze().astype(np.float64); break
                                if anat is None: continue
                                feat = normalize01(anat)
                                out_dict = kem(y, feat, iterations=n_max,
                                               sigma=sigma, J=J,
                                               save_iters=ITERS,
                                               forward=fwd, backward=bwd,
                                               mul_factor=mul, pet_noise=noi)
                            elif method.startswith("SAM-KEM"):
                                # Use the saved SAM prior if available.
                                prior_p = (ROOT / "arrays" / ds / str(sid)
                                           / "axial_center" / f"r{r}"
                                           / f"c{count}" / "sam_prior_medsam2.npy")
                                if prior_p.is_file():
                                    feat = normalize01(np.load(prior_p).squeeze().astype(np.float32))
                                else:
                                    feat = normalize01(mlem_init)
                                out_dict = kem(y, feat, iterations=n_max,
                                               sigma=sigma, J=J,
                                               save_iters=ITERS,
                                               forward=fwd, backward=bwd,
                                               mul_factor=mul, pet_noise=noi)
                            else:
                                continue
                            for it, x in out_dict.items():
                                rec = {
                                    "dataset": ds, "subject_id": str(sid),
                                    "slice_id": "axial_center",
                                    "realization_id": r, "count_level": count,
                                    "method": method, "iteration": it,
                                    "split": "test_locked",
                                    "psnr_db": psnr(target, x),
                                    "ssim": ssim(target, x),
                                    "nrmse": nrmse(target, x),
                                    "mae": float(np.abs(target - x).mean()),
                                }
                                w.writerow(rec)
                                n_written += 1
    print(f"wrote: {OUT.relative_to(ROOT)} ({n_written} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())