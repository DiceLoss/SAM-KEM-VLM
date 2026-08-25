#!/usr/bin/env python3
"""SAM prior inference using the TRAINED MedSAM2 mask decoder.

Loads the trained decoder state (from `checkpoints/medsam2_decoder_<dataset>.pt`)
and produces PET priors the same way as `medsam2_inference.py`, but with the
fine-tuned mask decoder weights.

If the trained checkpoint does not exist for the requested dataset, falls
back to the released pretrained MedSAM2 Hiera-T (no fine-tune).
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import numpy as np


_MODEL = None
_PREDICTOR = None
_DEVICE = "cpu"
_MODEL_CACHE_KEY = None


def _ensure_model(ckpt_path: Optional[Path] = None, image_size: int = 1024):
    """Historical compatibility loader.

    New code must use :mod:`sam_kem.medsam`. This function now keys its cache by
    resolved checkpoint path and file SHA so BW and HE checkpoints cannot leak
    across calls.
    """
    global _MODEL, _PREDICTOR, _DEVICE, _MODEL_CACHE_KEY
    import hashlib
    resolved = ckpt_path.resolve() if ckpt_path is not None else None
    checkpoint_sha = None
    if resolved is not None:
        if not resolved.is_file():
            raise FileNotFoundError(f"trained decoder checkpoint missing: {resolved}")
        checkpoint_sha = hashlib.sha256(resolved.read_bytes()).hexdigest()
    cache_key = (str(resolved), checkpoint_sha, int(image_size))
    if _MODEL is not None and _MODEL_CACHE_KEY == cache_key:
        return
    em_root = r"D:/MIAProject/EM-reconstruction"
    if em_root not in sys.path:
        sys.path.insert(0, em_root)
    import torch
    import build_medsam2  # EM-reconstruction/build_medsam2.py
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    _DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    _MODEL = build_medsam2.build_medsam2_hiera_t(image_size=image_size,
                                                 mode="eval",
                                                 reconfigure=False)
    if resolved is not None:
        sd = torch.load(str(resolved), map_location=_DEVICE)
        if "model_state" in sd:
            _MODEL.load_state_dict(sd["model_state"], strict=False)
        else:
            _MODEL.load_state_dict(sd, strict=False)
        print(f"loaded trained decoder from {resolved} sha256={checkpoint_sha}")
    else:
        raise RuntimeError("implicit pretrained fallback is prohibited")
    _MODEL.to(_DEVICE)
    _MODEL.eval()
    _PREDICTOR = SAM2ImagePredictor(sam_model=_MODEL)
    _MODEL_CACHE_KEY = cache_key


def _to_rgb(pet_2d: np.ndarray) -> np.ndarray:
    img = np.clip(pet_2d, 0.0, 1.0)
    img = (img * 255.0).astype(np.uint8)
    rgb = np.stack([img, img, img], axis=-1)
    from scipy.ndimage import zoom
    return zoom(rgb, (1024 / rgb.shape[0], 1024 / rgb.shape[1], 1), order=0).astype(np.uint8)


def _resize_to_128(mask_2d: np.ndarray) -> np.ndarray:
    from scipy.ndimage import zoom
    return zoom(mask_2d.astype(np.float32),
                 (128 / mask_2d.shape[0], 128 / mask_2d.shape[1]),
                 order=1).astype(np.float32)


def predict_prior(pet_2d: np.ndarray, *, box: tuple[int, int, int, int],
                  cache_features: bool = True,
                  ckpt_path: Optional[Path] = None) -> np.ndarray:
    _ensure_model(ckpt_path=ckpt_path)
    rgb = _to_rgb(pet_2d)
    if not cache_features:
        _PREDICTOR.reset_image()
    _PREDICTOR.set_image(rgb)
    sx, sy = 1024 / pet_2d.shape[0], 1024 / pet_2d.shape[1]
    xmin = max(0, min(1024, int(round(box[0] * sx))))
    xmax = max(0, min(1024, int(round(box[2] * sx))))
    ymin = max(0, min(1024, int(round(box[1] * sy))))
    ymax = max(0, min(1024, int(round(box[3] * sy))))
    box_np = np.array([xmin, ymin, xmax, ymax])
    masks, scores, _ = _PREDICTOR.predict(
        box=box_np, multimask_output=False, return_logits=True
    )
    mask_2d = masks[0].astype(np.float32)
    soft_mask = _resize_to_128(mask_2d)
    if np.unique(soft_mask).size <= 2:
        raise RuntimeError("binary MedSAM2 mask returned; continuous logits required")
    return soft_mask


def main() -> int:
    import time
    img = np.zeros((128, 128), dtype=np.float32)
    img[40:60, 50:80] = 1.0
    box = (40, 40, 80, 80)
    t0 = time.time()
    out = predict_prior(img, box=box,
                        ckpt_path=Path(r"D:/MIAProject/SAM_KEM_QWEN3_MEDSAM2_DELIVERY/checkpoints/medsam2_decoder_brainweb.pt"))
    print(f"inference time: {time.time()-t0:.2f}s")
    print(f"output shape: {out.shape}, min={out.min():.4f}, max={out.max():.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
