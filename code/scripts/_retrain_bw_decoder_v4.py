"""Phase 6e: retrain BW MedSAM2 decoder with FULL v4 training data.

Uses ALL slices from the v4 training split (2460 slices × 4 counts ×
3 realizations = 29,520 training cases vs v3's 168).

Output: checkpoints/medsam2_decoder_brainweb_v4.pt with a NEW SHA.

Usage (from Windows):
    python scripts/_retrain_bw_decoder_v4.py
"""
from __future__ import annotations

import csv
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(r"D:/MIAProject/SAM_KEM_QWEN3_MEDSAM2_DELIVERY")
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, r"D:/MIAProject/EM-reconstruction")

SPLIT = ROOT / "splits" / "subject_split_v4.csv"
ARRAYS = ROOT / "arrays" / "brainweb"
CHECKPOINT_OUT = ROOT / "checkpoints" / "medsam2_decoder_brainweb_v4.pt"
LOG_PATH = ROOT / "logs" / "medsam2_train_brainweb_v4_stdout.log"
PRETRAINED = Path(r"D:/MIAProject/EM-reconstruction/MedSAM2.pt")

COUNTS = [100_000, 200_000, 500_000, 1_000_000]
REALIZATIONS = [0, 1, 2]


def build_model(image_size=1024):
    import build_medsam2
    model = build_medsam2.build_medsam2_hiera_t(
        image_size=image_size, checkpoint=str(PRETRAINED), device="cuda",
        mode="train", reconfigure=True)
    return model


def freeze_all_but_mask_decoder(model):
    frozen, trainable = 0, 0
    for n, p in model.named_parameters():
        if "mask_decoder" in n:
            p.requires_grad = True
            trainable += p.numel()
        else:
            p.requires_grad = False
            frozen += p.numel()
    print(f"[freeze] frozen={frozen/1e6:.2f}M trainable={trainable/1e6:.2f}M")


def to_rgb(pet_2d):
    img = np.clip(pet_2d, 0.0, 1.0)
    img = (img * 255.0).astype(np.uint8)
    rgb = np.stack([img, img, img], axis=-1)
    from scipy.ndimage import zoom
    return zoom(rgb, (1024 / rgb.shape[0], 1024 / rgb.shape[1], 1), order=0).astype(np.uint8)


def resize_128(mask_2d):
    return F.interpolate(
        torch.from_numpy(mask_2d.astype(np.float32))[None, None],
        size=(128, 128), mode="bilinear", align_corners=False
    )[0, 0].numpy()


def predict_logits(model, image_uint8, box_xyxy, device):
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    pred = SAM2ImagePredictor(model)
    pred.set_image(image_uint8)
    img_embed = pred.get_image_embedding()
    high_res = pred._features["high_res_feats"]
    boxes = torch.tensor([box_xyxy], dtype=torch.float32, device=device)
    box_coords = boxes.reshape(-1, 2, 2)
    box_labels = torch.tensor([[2, 3]], dtype=torch.int, device=device)
    sparse, dense = model.sam_prompt_encoder(
        points=(box_coords, box_labels), boxes=None, masks=None)
    low_res, iou_pred, _, _ = model.sam_mask_decoder(
        image_embeddings=img_embed,
        image_pe=model.sam_prompt_encoder.get_dense_pe(),
        sparse_prompt_embeddings=sparse,
        dense_prompt_embeddings=dense,
        multimask_output=False,
        repeat_image=False,
        high_res_features=high_res,
    )
    masks = pred._transforms.postprocess_masks(low_res, pred._orig_hw[0])
    return masks


def fallback_box(sid, slice_id, count):
    """Use MLEM-10 centroid box as a fallback when no anchor exists."""
    sino_p = ARRAYS / sid / slice_id / "r0" / f"c{count}" / f"sinogram_count{count}.npy"
    if not sino_p.is_file():
        return (48, 48, 80, 80)
    from discrete_radon import build_pair
    from recon import mlem
    sino = np.load(sino_p).squeeze().astype(np.float64)
    fwd, bwd, _ = build_pair(size=128, n_angles=180)
    init = mlem(sino, iterations=10, save_iters=(10,), forward=fwd, backward=bwd)
    img = init[10]
    mx = img.max()
    if mx <= 0: return (48, 48, 80, 80)
    mask = img > 0.5 * mx
    if not mask.any(): return (48, 48, 80, 80)
    ys, xs = np.where(mask)
    cx, cy = int(np.mean(xs)), int(np.mean(ys))
    return (max(0, cx - 16), max(0, cy - 16), 128, 128)


def main():
    rows = list(csv.DictReader(open(SPLIT, encoding="utf-8")))
    train_rows = [r for r in rows if r["split"] == "train"]
    print(f"[init] brainweb v4 train slices: {len(train_rows)}")

    # Enumerate (subject, slice, realization, count) train cases.
    cases = []
    for r in train_rows:
        sid = r["subject_id"]
        slice_id = r["slice_id"]
        for r_idx in REALIZATIONS:
            for n_true in COUNTS:
                cases.append((sid, slice_id, r_idx, n_true))
    print(f"[init] n_train_cases: {len(cases)}")

    # Build model.
    print(f"Building MedSAM2 model from {PRETRAINED}...")
    model = build_model(image_size=1024)
    freeze_all_but_mask_decoder(model)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.train()
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=1e-4, weight_decay=1e-4,
    )

    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    log_fh = LOG_PATH.open("w", encoding="utf-8")
    def log(msg):
        print(msg, flush=True)
        log_fh.write(msg + "\n")

    # Training: iterate over all cases for ~3 epochs (each case once per epoch).
    n_epochs = 3
    rng = np.random.default_rng(20260824)
    step = 0
    losses = []
    t0 = time.time()
    for epoch in range(n_epochs):
        order = rng.permutation(len(cases))
        log(f"[epoch {epoch}] {len(cases)} cases")
        for idx in order:
            sid, slice_id, r_idx, n_true = cases[int(idx)]
            case_dir = ARRAYS / sid / slice_id / f"r{r_idx}"
            sino_path = case_dir / f"c{n_true}" / f"sinogram_count{n_true}.npy"
            target_path = case_dir / "target.npy"
            if not sino_path.is_file() or not target_path.is_file():
                continue
            y = np.load(sino_path).astype(np.float64)
            target = np.load(target_path).astype(np.float32)
            from discrete_radon import build_pair
            from recon import mlem
            fwd, bwd, _ = build_pair(size=128, n_angles=180)
            init = mlem(y, iterations=5, save_iters=(5,), forward=fwd, backward=bwd)
            rgb = to_rgb(init[5])
            box = fallback_box(sid, slice_id, n_true)
            sx, sy = 1024 / 128, 1024 / 128
            box_1024 = (int(round(box[0] * sx)), int(round(box[1] * sy)),
                         int(round(box[2] * sx)), int(round(box[3] * sy)))
            try:
                logits = predict_logits(model, rgb, box_1024, device=device)
                pred_2d_t = torch.sigmoid(logits[0, 0])
                pred_128 = F.interpolate(
                    pred_2d_t[None, None],
                    size=(128, 128), mode="bilinear", align_corners=False)[0, 0]
                target_t = torch.from_numpy(target).to(device)
                loss = F.l1_loss(pred_128, target_t)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses.append(float(loss.item()))
                if step % 200 == 0:
                    log(f"[epoch {epoch} step {step}] loss={float(loss.item()):.5f} "
                        f"elapsed={time.time()-t0:.1f}s")
                step += 1
            except Exception as exc:
                log(f"[step {step}] {sid}/{slice_id}/r{r_idx}/c{n_true}: ERROR ({exc})")
                continue

    # Save.
    CHECKPOINT_OUT.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state": model.state_dict(),
        "dataset": "brainweb",
        "n_epochs": n_epochs,
        "n_train_cases": len(cases),
        "lr": 1e-4,
    }, CHECKPOINT_OUT)
    # Compute SHA.
    import hashlib
    sha = hashlib.sha256(open(CHECKPOINT_OUT, "rb").read()).hexdigest()
    log(f"[done] ckpt={CHECKPOINT_OUT} sha256={sha}")
    log_fh.close()
    print(f"Total steps: {step}, final loss: {losses[-1] if losses else 'N/A'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())