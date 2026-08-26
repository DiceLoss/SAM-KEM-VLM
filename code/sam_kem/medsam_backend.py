"""Strict MedSAM2 construction from one declared upstream checkout."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from typing import Any


MODEL_CONFIGS = {
  "sam2.1_hiera_t": "configs/sam2.1/sam2.1_hiera_t.yaml",
}


def build_medsam2(
  *, source_checkout: Path, architecture: str, checkpoint: Path,
  device: Any, mode: str,
) -> Any:
  """Build exactly the configured architecture and checkpoint, with no fallback."""
  source = source_checkout.resolve()
  checkpoint = checkpoint.resolve()
  if architecture not in MODEL_CONFIGS:
    raise ValueError(f"unsupported MedSAM2 architecture: {architecture}")
  if not (source / "sam2" / "build_sam.py").is_file():
    raise FileNotFoundError(f"declared SAM2 checkout is invalid: {source}")
  if not checkpoint.is_file():
    raise FileNotFoundError(f"declared MedSAM2 checkpoint is missing: {checkpoint}")
  source_text = str(source)
  if source_text not in sys.path:
    sys.path.insert(0, source_text)
  module = importlib.import_module("sam2.build_sam")
  module_path = Path(module.__file__).resolve()
  if source not in module_path.parents:
    raise RuntimeError(
      f"sam2 was imported from {module_path}, not the declared checkout {source}"
    )
  return module.build_sam2(
    MODEL_CONFIGS[architecture], str(checkpoint), device=device, mode=mode,
  )
