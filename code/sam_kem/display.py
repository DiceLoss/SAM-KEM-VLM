"""Deterministic 128x128 display rendering for VLM prompts only."""

from __future__ import annotations

from pathlib import Path
import hashlib
import json

import numpy as np

from .noise import array_sha256


def render_prompt_png(array_path: Path, output_path: Path) -> Path:
  from PIL import Image

  source_array = np.load(array_path).squeeze()
  array = source_array.astype(np.float64)
  if array.shape != (128, 128) or not np.all(np.isfinite(array)):
    raise ValueError("prompt array must be a finite 128x128 image")
  low, high = float(array.min()), float(array.max())
  display = np.zeros_like(array) if high <= low else (array - low) / (high - low)
  pixels = np.round(display * 255).astype(np.uint8)
  metadata_path = output_path.with_suffix(".json")
  if output_path.exists() or metadata_path.exists():
    raise FileExistsError(f"refusing to overwrite prompt rendering: {output_path}")
  output_path.parent.mkdir(parents=True, exist_ok=True)
  Image.fromarray(pixels, mode="L").convert("RGB").save(output_path, format="PNG", optimize=False)
  metadata_path.write_text(json.dumps({
    "schema_version": 1, "source_array_path": str(array_path),
    "source_mlem10_sha256": array_sha256(source_array),
    "display_transform": "per_image_minmax_round_to_uint8_rgb_only_for_vlm",
    "display_shape": [128, 128, 3],
    "png_sha256": hashlib.sha256(output_path.read_bytes()).hexdigest(),
  }, indent=2), encoding="utf-8")
  return output_path
