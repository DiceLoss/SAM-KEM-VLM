"""Fail-fast CUDA/MedSAM2 forward-backward smoke test before long training."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np
from sam_kem.boxes import Box
from sam_kem.config import load_yaml_strict
from sam_kem.medsam import (
  SINGLE_MASK_L1_INACTIVE_DECODER_PREFIXES,
  freeze_except_mask_decoder,
  pet_to_rgb,
  sha256_file,
)
from sam_kem.medsam_backend import MODEL_CONFIGS, build_medsam2
from sam_kem.medsam_training import _forward_logits


def preflight(root: Path, config_path: Path) -> dict:
  import torch
  import torch.nn.functional as F

  cfg = load_yaml_strict(config_path)
  if not torch.cuda.is_available():
    raise RuntimeError("CUDA is unavailable; do not start decoder training")
  source = root / cfg["medsam2"]["source_checkout_path"]
  source_head = subprocess.run(
    ["git", "rev-parse", "HEAD"], cwd=source, check=True,
    capture_output=True, text=True,
  ).stdout.strip()
  if source_head != cfg["medsam2"]["source_commit"]:
    raise RuntimeError(f"MedSAM2 checkout mismatch: {source_head}")
  official_config = source / "sam2" / MODEL_CONFIGS[cfg["medsam2"]["architecture"]]
  if not official_config.is_file():
    raise FileNotFoundError(f"official MedSAM2 config missing: {official_config}")
  checkpoint = root / cfg["medsam2"]["pretrained_checkpoint_path"]
  actual_sha = sha256_file(checkpoint)
  if actual_sha != cfg["medsam2"]["pretrained_checkpoint_sha256"]:
    raise RuntimeError(f"pretrained checkpoint SHA mismatch: {actual_sha}")
  device = torch.device("cuda")
  model = build_medsam2(
    source_checkout=source, architecture=cfg["medsam2"]["architecture"],
    checkpoint=checkpoint, device=device, mode="train",
  )
  from sam2.sam2_image_predictor import SAM2ImagePredictor
  predictor = SAM2ImagePredictor(model)
  freeze_except_mask_decoder(model)
  model.eval(); model.sam_mask_decoder.train(); model.to(device)
  image = np.linspace(0, 1, 128 * 128, dtype=np.float32).reshape(128, 128)
  logits = _forward_logits(
    model, predictor, pet_to_rgb(image), Box(24, 24, 104, 104),
    source_shape=image.shape, device=device,
  )
  prediction = F.interpolate(logits, size=(128, 128), mode="bilinear", align_corners=False)
  loss = F.l1_loss(prediction, torch.zeros_like(prediction))
  loss.backward()
  decoder_parameters = [
    (name, parameter) for name, parameter in model.named_parameters()
    if parameter.requires_grad
  ]
  missing_gradients = [name for name, parameter in decoder_parameters if parameter.grad is None]
  nonfinite_gradients = [
    name for name, parameter in decoder_parameters
    if parameter.grad is not None and not torch.isfinite(parameter.grad).all()
  ]
  if not decoder_parameters or missing_gradients or nonfinite_gradients:
    raise RuntimeError(
      "single-mask L1 decoder backward failed: "
      f"missing_gradients={missing_gradients}, nonfinite_gradients={nonfinite_gradients}"
    )
  if any(
    parameter.grad is not None for name, parameter in model.named_parameters()
    if not name.startswith("sam_mask_decoder.")
  ):
    raise RuntimeError("a frozen encoder received gradients")
  return {
    "status": "PASS", "cuda_device": torch.cuda.get_device_name(0),
    "torch_version": torch.__version__, "cuda_runtime": torch.version.cuda,
    "medsam2_source_commit": source_head, "medsam2_config": str(official_config),
    "pretrained_checkpoint_sha256": actual_sha,
    "model_image_size": int(model.image_size),
    "decoder_trainable_parameters": int(sum(
      parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )),
    "inactive_decoder_head_prefixes": list(SINGLE_MASK_L1_INACTIVE_DECODER_PREFIXES),
    "trainable_decoder_parameter_tensors": len(decoder_parameters),
    "smoke_loss": float(loss.detach().cpu()),
  }


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--root", type=Path, default=Path("."))
  parser.add_argument("--config", type=Path, default=Path("code/configs/experiment.yaml"))
  args = parser.parse_args()
  print(json.dumps(preflight(args.root.resolve(), args.config.resolve()), indent=2))
