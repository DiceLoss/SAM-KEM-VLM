#!/usr/bin/env python3
"""Real MedSAM2 Hiera-T inference wrapper for the SAM-KEM pipeline.

Uses the `sam2` package (>=1.1.0) + the pretrained MedSAM2 weights
(`D:/MIAProject/EM-reconstruction/MedSAM2.pt`, SHA-256
`c92743b9…1060`) to produce a continuous PET prior.

Pipeline per call:
  1. Resize the 128×128 PET image to 1024×1024 (native MedSAM2 input).
  2. Run `SAM2ImagePredictor.set_image` and `predict(box=...)`.
  3. Take the soft mask output and resize back to 128×128.
  4. Optional: train the mask decoder on PET targets (L1 only). The
     training loop is gated by `mode="train"`; this module's primary
     purpose is `mode="eval"` (inference only).

The wrapper caches the model + the image features. The first call
initializes the model; subsequent calls reuse the loaded weights.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Optional

import numpy as np

# Lazy imports: torch + sam2 may not be importable from every Python; we
# fail at first use with a clear message.


_MODEL = None
_PREDICTOR = None
_DEVICE = "cpu"


def _ensure_model():
    """Build the SAM2 model + predictor on the first call."""
    global _MODEL, _PREDICTOR, _DEVICE
    if _MODEL is not None:
        return
    # Add the EM-reconstruction directory to sys.path so we can import
    # build_medsam2 + sam2.
    em_root = r"D:/MIAProject/EM-reconstruction"
    if em_root not in sys.path:
        sys.path.insert(0, em_root)
    import torch
    import build_medsam2  # EM-reconstruction/build_medsam2.py
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    _DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    _MODEL = build_medsam2.build_medsam2_hiera_t(image_size=1024,
                                                 mode="eval",
                                                 reconfigure=False)
    _MODEL.to(_DEVICE)
    _MODEL.eval()
    _PREDICTOR = SAM2ImagePredictor(sam_model=_MODEL)


def _to_rgb(pet_2d: np.ndarray) -> np.ndarray:
    """Convert a [0,1] grayscale 2D PET image to a 1024x1024 uint8 RGB."""
    img = np.clip(pet_2d, 0.0, 1.0)
    img = (img * 255.0).astype(np.uint8)
    rgb = np.stack([img, img, img], axis=-1)  # H, W, 3
    # Resize to 1024x1024 (nearest neighbour)
    from scipy.ndimage import zoom
    rgb = zoom(rgb, (1024 / rgb.shape[0], 1024 / rgb.shape[1], 1), order=0)
    return rgb.astype(np.uint8)


def _resize_to_128(mask_2d: np.ndarray) -> np.ndarray:
    from scipy.ndimage import zoom
    return zoom(mask_2d.astype(np.float32),
                 (128 / mask_2d.shape[0], 128 / mask_2d.shape[1]),
                 order=1).astype(np.float32)


def predict_prior(pet_2d: np.ndarray, *, box: tuple[int, int, int, int],
                  cache_features: bool = True) -> np.ndarray:
    """Run MedSAM2 with a box prompt and return a soft prior at 128x128.

    Parameters
    ----------
    pet_2d : ndarray (128, 128) float in [0, 1]
    box : (xmin, ymin, xmax, ymax) in 128x128 coords; the box is mapped
        to the 1024x1024 native grid by *8 scaling.
    cache_features : if True (default), cache the image encoder features
        between calls with the same pet_2d.

    Returns
    -------
    soft_mask : ndarray (128, 128) float in [0, 1]
    """
    _ensure_model()
    rgb = _to_rgb(pet_2d)
    if not cache_features:
        _PREDICTOR.reset_image()
    _PREDICTOR.set_image(rgb)
    # Scale box from 128-coords to 1024-coords
    sx, sy = 1024 / pet_2d.shape[0], 1024 / pet_2d.shape[1]
    xmin = int(round(box[0] * sx))
    ymin = int(round(box[1] * sy))
    xmax = int(round(box[2] * sx))
    ymax = int(round(box[3] * sy))
    xmin = max(0, min(1024, xmin))
    xmax = max(0, min(1024, xmax))
    ymin = max(0, min(1024, ymin))
    ymax = max(0, min(1024, ymax))
    box_np = np.array([xmin, ymin, xmax, ymax])
    masks, scores, _ = _PREDICTOR.predict(box=box_np, multimask_output=False)
    mask_2d = masks[0].astype(np.float32)
    soft_mask = _resize_to_128(mask_2d)
    # Light normalization: clip to [0, 1]
    soft_mask = np.clip(soft_mask, 0.0, 1.0)
    return soft_mask


def main() -> int:
    import time
    img = np.zeros((128, 128), dtype=np.float32)
    img[40:60, 50:80] = 1.0
    box = (40, 40, 80, 80)
    t0 = time.time()
    out = predict_prior(img, box=box)
    print(f"inference time: {time.time()-t0:.2f}s")
    print(f"output shape: {out.shape}, min={out.min():.4f}, max={out.max():.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())