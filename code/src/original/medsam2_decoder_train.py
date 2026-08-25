#!/usr/bin/env python3
"""Train the MedSAM2 Hiera-T mask decoder on PET targets.

Per the P0 contract:
- Image encoder, prompt encoder, memory encoder are FROZEN.
- Only the mask decoder is trainable.
- Loss = L1 only (no Dice, BCE, MSE, perceptual, adversarial).
- Training data is sampled 1:1:1:1 across the four count levels.
- One decoder checkpoint per dataset (BW, HE).
- Box prompt is the Qwen3 cached box (or intensity-centroid fallback).

Runtime-bound: 1 epoch over a subset of training cases. The number of
cases per epoch and the optimizer are configurable via CLI.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import os
import sys
import time
from pathlib import Path

import numpy as np


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _ensure_syspath():
    em_root = r"D:/MIAProject/EM-reconstruction"
    if em_root not in sys.path:
        sys.path.insert(0, em_root)


def _build_model(image_size: int = 1024):
    _ensure_syspath()
    import torch
    import build_medsam2  # noqa: E402
    model = build_medsam2.build_medsam2_hiera_t(image_size=image_size,
                                                  mode="train",
                                                  reconfigure=False)
    return model, torch


def _freeze_all_but_mask_decoder(model) -> None:
    """Freeze every parameter except the mask decoder (per P0 contract)."""
    import torch
    frozen, trainable = 0, 0
    for name, p in model.named_parameters():
        if "sam_mask_decoder" in name:
            p.requires_grad = True
            trainable += p.numel()
        else:
            p.requires_grad = False
            frozen += p.numel()
    print(f"[freeze] frozen={frozen/1e6:.2f}M trainable={trainable/1e6:.2f}M")


def _predict_logits(model, image_uint8, box_xyxy, *, device):
    """Single-image single-box forward pass returning the soft mask logits."""
    import torch
    import torch.nn.functional as F
    from sam2.sam2_image_predictor import SAM2ImagePredictor  # noqa: E402
    pred = SAM2ImagePredictor(model)
    pred.set_image(image_uint8)
    img_embed = pred.get_image_embedding()
    high_res = pred._features["high_res_feats"]
    boxes = torch.tensor([box_xyxy], dtype=torch.float32, device=device)
    box_coords = boxes.reshape(-1, 2, 2)
    box_labels = torch.tensor([[2, 3]], dtype=torch.int, device=device)
    sparse, dense = model.sam_prompt_encoder(
        points=(box_coords, box_labels), boxes=None, masks=None,
    )
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
    return masks  # shape (1, 1, H, W), already upsampled to 1024


def _to_rgb(pet_2d: np.ndarray) -> np.ndarray:
    img = np.clip(pet_2d, 0.0, 1.0)
    img = (img * 255.0).astype(np.uint8)
    rgb = np.stack([img, img, img], axis=-1)
    from scipy.ndimage import zoom
    return zoom(rgb, (1024 / rgb.shape[0], 1024 / rgb.shape[1], 1), order=0).astype(np.uint8)


def _resize_128(mask_2d: np.ndarray) -> np.ndarray:
    from scipy.ndimage import zoom
    return zoom(mask_2d.astype(np.float32),
                 (128 / mask_2d.shape[0], 128 / mask_2d.shape[1]),
                 order=1).astype(np.float32)


def _load_qwen_box(prompts_root: Path, dataset: str, subject_id: str,
                   slice_id: str, count_level: int, realization: int) -> tuple:
    p = prompts_root / dataset / f"{dataset}_{subject_id}_{slice_id}_{count_level}_anchor_{realization}_0.json"
    if p.is_file():
        try:
            import json
            rec = json.loads(p.read_text(encoding="utf-8"))
            b = rec.get("box", {})
            return (int(b["xmin"]), int(b["ymin"]), int(b["xmax"]), int(b["ymax"]))
        except Exception:
            pass
    # fallback intensity-centroid box
    return (32, 32, 96, 96)


def train_one(dataset: str, *, brainweb_npy: Path | None,
               hecktor_npz: Path | None,
               train_subjects: list[str],
               arrays_root: Path,
               prompts_root: Path,
               count_levels: tuple[int, ...] = (100_000, 200_000, 500_000, 1_000_000),
               realizations: tuple[int, ...] = (0, 1, 2),
               epochs: int = 1,
               lr: float = 1e-4,
               max_steps: int = 80,
               out_ckpt: Path = None,
               log_path: Path = None) -> dict:
    """Train the mask decoder for one dataset."""
    import torch
    import torch.nn.functional as F

    model, torch_mod = _build_model(image_size=1024)
    _freeze_all_but_mask_decoder(model)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.train()
    optimizer = torch_mod.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr, weight_decay=1e-4,
    )

    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_fh = log_path.open("w", encoding="utf-8") if log_path else None

    def log(msg: str) -> None:
        print(msg, flush=True)
        if log_fh:
            log_fh.write(msg + "\n")

    # Enumerate (subject, realization, count) train cases
    cases: list[tuple] = []
    for sid in train_subjects:
        for r in realizations:
            for n_true in count_levels:
                cases.append((sid, r, n_true))
    log(f"[init] dataset={dataset} n_train_cases={len(cases)}")
    rng = np.random.default_rng(20260817)

    # Training loop
    step = 0
    losses: list[float] = []
    t_start = time.time()
    for epoch in range(epochs):
        order = rng.permutation(len(cases))
        for idx in order:
            if step >= max_steps:
                break
            sid, r_idx, n_true = cases[int(idx)]
            case_dir = arrays_root / "arrays" / dataset / sid / "axial_center" / f"r{r_idx}"
            sino_path = case_dir / f"sinogram_count{n_true}.npy"
            target_path = case_dir / "target.npy"
            if not sino_path.is_file() or not target_path.is_file():
                continue
            y = np.load(sino_path).astype(np.float64)
            target = np.load(target_path).astype(np.float32)
            # MLEM init as the SAM-KEM pipeline input image
            import recon  # noqa: E402
            mlem_init = recon.mlem(y, iterations=5, save_iters=(5,))[5]
            rgb = _to_rgb(mlem_init)
            box = _load_qwen_box(prompts_root, dataset, sid, "axial_center",
                                  n_true, r_idx)
            sx, sy = 1024 / 128, 1024 / 128
            box_1024 = (int(round(box[0] * sx)), int(round(box[1] * sy)),
                         int(round(box[2] * sx)), int(round(box[3] * sy)))
            try:
                with torch_mod.enable_grad():
                    logits = _predict_logits(model, rgb, box_1024, device=device)
                # logits shape (1, 1, 1024, 1024)
                soft_full = torch_mod.sigmoid(logits)
                soft_2d = soft_full[0, 0].detach().cpu().numpy()
                soft_128 = _resize_128(soft_2d)
                # L1 against target
                pred_t = torch_mod.from_numpy(soft_128).to(device)
                target_t = torch_mod.from_numpy(target).to(device)
                # Need gradients flowing back, so keep soft_2d in graph
                # Re-do with grad enabled
                pred_2d_t = torch_mod.sigmoid(logits[0, 0])  # (1024, 1024)
                pred_128 = F.interpolate(
                    pred_2d_t[None, None],
                    size=(128, 128), mode="bilinear", align_corners=False,
                )[0, 0]
                loss = F.l1_loss(pred_128, target_t)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses.append(float(loss.item()))
                if step % 5 == 0:
                    log(f"[epoch {epoch} step {step}] loss={float(loss.item()):.5f} "
                        f"elapsed={time.time()-t_start:.1f}s")
                step += 1
            except Exception as exc:  # noqa: BLE001
                log(f"[step {step}] {dataset}/{sid}/r{r_idx}/c{n_true}: ERROR ({exc})")
                continue
        if step >= max_steps:
            break

    # Save the trained decoder state
    if out_ckpt:
        out_ckpt.parent.mkdir(parents=True, exist_ok=True)
        torch_mod.save({
            "model_state": model.state_dict(),
            "dataset": dataset,
            "epochs": epochs,
            "max_steps": max_steps,
            "lr": lr,
            "n_train_cases": len(cases),
            "n_train_subjects": len(train_subjects),
        }, out_ckpt)
        ckpt_sha = sha256_file(out_ckpt)
        log(f"[done] ckpt={out_ckpt} sha256={ckpt_sha}")
    if log_fh:
        log_fh.close()
    return {"step": step, "losses": losses, "out_ckpt": str(out_ckpt) if out_ckpt else None}


def main() -> int:
    p = argparse.ArgumentParser(description="Train MedSAM2 decoder on PET")
    p.add_argument("--dataset", choices=["brainweb", "hecktor"], required=True)
    p.add_argument("--brainweb-npy", type=Path,
                   default=Path(r"D:/MIAProject/brainweb/pet_vol_length_36by144_nt_mr.npy") if False
                   else Path(r"D:/MIAProject/brainweb/pet_vol_length_36by144_nt_mr.npy"))
    p.add_argument("--hecktor-npz", type=Path,
                   default=Path(r"D:/MIAProject/HECKTOR.npz"))
    p.add_argument("--arrays-root", type=Path,
                   default=Path(r"D:/MIAProject/SAM_KEM_QWEN3_MEDSAM2_DELIVERY"))
    p.add_argument("--prompts-root", type=Path,
                   default=Path(r"D:/MIAProject/SAM_KEM_QWEN3_MEDSAM2_DELIVERY/prompts"))
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--max-steps", type=int, default=80)
    p.add_argument("--out-ckpt", type=Path, default=None)
    p.add_argument("--log-path", type=Path, default=None)
    args = p.parse_args()
    # Auto-default ckpt + log paths
    if args.out_ckpt is None:
        args.out_ckpt = args.arrays_root / "checkpoints" / f"medsam2_decoder_{args.dataset}.pt"
    if args.log_path is None:
        args.log_path = args.arrays_root / "logs" / f"medsam2_train_{args.dataset}.log"

    # Pull train subject IDs from splits
    splits = list(csv.DictReader(open(args.arrays_root / "splits" / "subject_split.csv")))
    train_subjects = [r["subject_id"] for r in splits
                       if r["dataset"] == args.dataset and r["split"] == "train"]
    print(f"[main] {args.dataset} train subjects: {len(train_subjects)}")
    summary = train_one(args.dataset,
                         brainweb_npy=args.brainweb_npy if args.dataset == "brainweb" else None,
                         hecktor_npz=args.hecktor_npz if args.dataset == "hecktor" else None,
                         train_subjects=train_subjects,
                         arrays_root=args.arrays_root,
                         prompts_root=args.prompts_root,
                         epochs=args.epochs, lr=args.lr, max_steps=args.max_steps,
                         out_ckpt=args.out_ckpt,
                         log_path=args.log_path)
    print(f"[main] summary: {summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())