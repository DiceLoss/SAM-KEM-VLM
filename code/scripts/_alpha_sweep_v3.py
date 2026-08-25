"""Phase 7d: alpha sweep on the NEW validation set with real SAM priors.

After Phase 6 has frozen (sigma*, J*, N_KEM*) per (dataset, count),
this script:
  1. Generates MedSAM2 priors for every (subject, count, realization)
     in the NEW validation set.
  2. For each (dataset, count), sweeps alpha in {0, 0.1, 0.2, 0.3,
     0.4, 0.5} on the validation set.
  3. Selects the alpha that minimizes the mean subject-level NRMSE
     (SSIM tie-breaker).
  4. Updates metrics/frozen_hyperparameters.csv with the chosen alpha.

The priors are saved to
  arrays/<ds>/<sid>/axial_center/r<r>/c<count>/sam_prior_medsam2.npy
and reused by scripts/_rerun_primary_v3.py.
"""
from __future__ import annotations

import csv
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(r"D:/MIAProject/SAM_KEM_QWEN3_MEDSAM2_DELIVERY")
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, r"D:/MIAProject/EM-reconstruction")

from discrete_radon import build_pair  # noqa: E402
from recon import kem, mlem, post_fusion, normalize01  # noqa: E402
import medsam2_decoder_infer  # noqa: E402

SPLITS = ROOT / "splits" / "subject_split.csv"
ANCHOR = ROOT / "prompts" / "anchor_box_lookup.csv"
FROZEN = ROOT / "metrics" / "frozen_hyperparameters.csv"
ARRAYS = ROOT / "arrays"
COUNTS = [100_000, 200_000, 500_000, 1_000_000]
REALIZATIONS = [0, 1, 2]
ALPHAS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
IMG_SIZE = 128
N_ANGLES = 180


def psnr(target, x):
    mse = float(((target - x) ** 2).mean())
    return 100.0 if mse <= 1e-12 else 20 * np.log10(1.0 / np.sqrt(mse))


def nrmse(target, x):
    t = target.flatten().astype(np.float64)
    return float(np.linalg.norm((target - x).flatten()) / max(float(np.linalg.norm(t)), 1e-12))


def load_sino(ds, sid, count, r):
    rdir = ARRAYS / ds / sid / "axial_center" / f"r{r}"
    y_p = rdir / f"sinogram_count{count}.npy"
    if not y_p.is_file():
        return None
    y = np.load(y_p).squeeze().astype(np.float64)
    m_p = rdir / f"mul_factor_count{count}.npy"
    n_p = rdir / f"pet_noise_count{count}.npy"
    m = np.load(m_p).squeeze().astype(np.float64) if m_p.is_file() else np.ones_like(y)
    n = np.load(n_p).squeeze().astype(np.float64) if n_p.is_file() else np.zeros_like(y)
    return y, m, n


def load_anchor(ds, sid, count):
    if not ANCHOR.is_file():
        return None
    for r in csv.DictReader(open(ANCHOR, encoding="utf-8")):
        if (r["dataset"] == ds and r["subject_id"] == sid
                and int(r["count_level"]) == count):
            return (int(r["xmin"]), int(r["ymin"]),
                    int(r["xmax"]), int(r["ymax"]))
    return None


def fallback_box(ds, sid, count, fwd, bwd):
    sino = load_sino(ds, sid, count, 0)
    if sino is None: return (48, 48, 80, 80)
    y, m, n = sino
    init = mlem(y, iterations=10, save_iters=(10,),
                forward=fwd, backward=bwd, mul_factor=m, pet_noise=n)
    img = init[10]
    mx = img.max()
    if mx <= 0: return (48, 48, 80, 80)
    mask = img > 0.5 * mx
    if not mask.any(): return (48, 48, 80, 80)
    ys, xs = np.where(mask)
    cx, cy = int(np.mean(xs)), int(np.mean(ys))
    return (max(0, cx - 16), max(0, cy - 16),
            min(IMG_SIZE, cx + 16), min(IMG_SIZE, cy + 16))


def main() -> int:
    rows = list(csv.DictReader(open(SPLITS, encoding="utf-8")))
    val = [r for r in rows if r["split"] == "validation"]
    val_bw = sorted(set(r["subject_id"] for r in val if r["dataset"] == "brainweb"))
    val_he = sorted(set(r["subject_id"] for r in val if r["dataset"] == "hecktor"))
    print(f"validation: BW={val_bw}, HE={len(val_he)} subjects")

    frozen = list(csv.DictReader(open(FROZEN, encoding="utf-8")))
    frozen_idx = {(r["dataset"], int(r["count_level"])): r for r in frozen}

    fwd, bwd, _ = build_pair(size=IMG_SIZE, n_angles=N_ANGLES)
    bw_ckpt = ROOT / "checkpoints" / "medsam2_decoder_brainweb.pt"
    he_ckpt = ROOT / "checkpoints" / "medsam2_decoder_hecktor.pt"

    # Pre-load both decoders.
    print("Pre-loading BW decoder...")
    medsam2_decoder_infer._ensure_model(ckpt_path=bw_ckpt, image_size=1024)
    print("Pre-loading HE decoder...")
    medsam2_decoder_infer._ensure_model(ckpt_path=he_ckpt, image_size=1024)

    n_priors = 0
    for ds, subs, ckpt in [("brainweb", val_bw, bw_ckpt),
                           ("hecktor", val_he, he_ckpt)]:
        for sid in subs:
            for count in COUNTS:
                box = load_anchor(ds, sid, count)
                if box is None:
                    box = fallback_box(ds, sid, count, fwd, bwd)
                for r in REALIZATIONS:
                    prior_p = (ARRAYS / ds / sid / "axial_center" / f"r{r}"
                               / f"c{count}" / "sam_prior_medsam2.npy")
                    if prior_p.is_file():
                        continue
                    sino = load_sino(ds, sid, count, r)
                    if sino is None: continue
                    y, m, n = sino
                    init = mlem(y, iterations=10, save_iters=(10,),
                                forward=fwd, backward=bwd,
                                mul_factor=m, pet_noise=n)
                    pet = np.clip(init[10], 0, 1).astype(np.float32)
                    prior = medsam2_decoder_infer.predict_prior(
                        pet, box=box, ckpt_path=ckpt)
                    prior_p.parent.mkdir(parents=True, exist_ok=True)
                    np.save(prior_p, prior.astype(np.float32))
                    n_priors += 1
    print(f"Generated {n_priors} validation priors")

    # Alpha sweep.
    print("\n=== Alpha sweep ===")
    new_frozen = []
    for fr in frozen:
        ds = fr["dataset"]; count = int(fr["count_level"])
        sigma = float(fr["sigma_star"]); J = int(fr["J_star"])
        n_it = int(fr["N_KEM_star"])
        if ds == "brainweb":
            subs = val_bw
        else:
            subs = val_he
        # Compute per-alpha mean NRMSE.
        alpha_results: dict[float, list[float]] = {a: [] for a in ALPHAS}
        for sid in subs:
            for r in REALIZATIONS:
                prior_p = (ARRAYS / ds / sid / "axial_center" / f"r{r}"
                           / f"c{count}" / "sam_prior_medsam2.npy")
                if not prior_p.is_file():
                    continue
                prior = np.load(prior_p).squeeze().astype(np.float32)
                sino = load_sino(ds, sid, count, r)
                if sino is None: continue
                y, m, n = sino
                target_p = ARRAYS / ds / sid / "axial_center" / "r0" / "target.npy"
                if not target_p.is_file(): continue
                target = np.load(target_p).squeeze().astype(np.float64)
                # Run KEM at n_it.
                out = kem(y, normalize01(prior),
                          iterations=n_it, sigma=sigma, J=J,
                          save_iters=(n_it,),
                          forward=fwd, backward=bwd,
                          mul_factor=m, pet_noise=n)
                x_kem = out[n_it]
                for a in ALPHAS:
                    x = post_fusion(prior, x_kem, alpha=a)
                    alpha_results[a].append(nrmse(target, x))
        # Pick best alpha by mean NRMSE.
        best_a = None
        best_nrmse = float("inf")
        for a, vals in alpha_results.items():
            if not vals: continue
            m_nrmse = float(np.mean(vals))
            if m_nrmse < best_nrmse:
                best_nrmse = m_nrmse
                best_a = a
        if best_a is None:
            best_a = 0.3  # paper default
        new_fr = dict(fr)
        new_fr["alpha_star"] = best_a
        new_frozen.append(new_fr)
        print(f"  {ds}/{count}: best alpha = {best_a} "
              f"(val NRMSE {best_nrmse:.4f}, n={len(alpha_results.get(best_a, []))})")

    # Write back.
    fields = list(frozen[0].keys())
    with open(FROZEN, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(new_frozen)
    print(f"\nWrote updated frozen hyperparameters: {FROZEN.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())