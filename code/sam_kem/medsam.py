"""MedSAM2 loading and continuous decoder-output inference."""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .boxes import Box
from .medsam_backend import build_medsam2

SINGLE_MASK_L1_INACTIVE_DECODER_PREFIXES = (
  "sam_mask_decoder.iou_prediction_head.",
  "sam_mask_decoder.pred_obj_score_head.",
  "sam_mask_decoder.output_hypernetworks_mlps.1.",
  "sam_mask_decoder.output_hypernetworks_mlps.2.",
  "sam_mask_decoder.output_hypernetworks_mlps.3.",
)


def is_single_mask_l1_trainable_parameter(name: str) -> bool:
  """Return whether a parameter contributes to the selected single-mask logit."""
  return name.startswith("sam_mask_decoder.") and not name.startswith(
    SINGLE_MASK_L1_INACTIVE_DECODER_PREFIXES
  )


def sha256_file(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as handle:
    for chunk in iter(lambda: handle.read(chunk_size), b""):
      digest.update(chunk)
  return digest.hexdigest()


@dataclass(frozen=True)
class CheckpointSpec:
  dataset: str
  path: Path
  sha256: str
  normalization_id: str = ""
  run_id: str = ""

  def verify(self) -> None:
    if not self.path.is_file():
      raise FileNotFoundError(f"checkpoint missing for {self.dataset}: {self.path}")
    actual = sha256_file(self.path)
    if actual != self.sha256:
      raise ValueError(
        f"checkpoint SHA mismatch for {self.dataset}: expected {self.sha256}, actual {actual}"
      )


class PredictorRegistry:
  """Cache predictors by verified checkpoint identity, not process globally."""

  def __init__(self, factory: Callable[[CheckpointSpec], Any]):
    self._factory = factory
    self._predictors: dict[tuple[str, str, str], Any] = {}

  def get(self, checkpoint: CheckpointSpec) -> Any:
    checkpoint.verify()
    key = (checkpoint.dataset, checkpoint.sha256, checkpoint.run_id)
    if key not in self._predictors:
      self._predictors[key] = self._factory(checkpoint)
    return self._predictors[key]


def pet_to_rgb(pet: np.ndarray, *, display_size: int = 1024) -> np.ndarray:
  """Create a display image only; this scaling is not reconstruction scaling."""
  array = np.asarray(pet, dtype=np.float32)
  if array.ndim != 2:
    raise ValueError("PET input must be 2-D")
  low, high = float(array.min()), float(array.max())
  display = np.zeros_like(array) if high <= low else (array - low) / (high - low)
  from scipy.ndimage import zoom
  rgb = np.repeat((display[..., None] * 255).astype(np.uint8), 3, axis=2)
  factors = (display_size / array.shape[0], display_size / array.shape[1], 1)
  return zoom(rgb, factors, order=1).astype(np.uint8)


def predict_continuous_prior(
  predictor: Any,
  pet: np.ndarray,
  box: Box,
  *,
  output_shape: tuple[int, int],
  display_size: int = 1024,
) -> np.ndarray:
  """Return low-resolution decoder logits resized exactly as during training."""
  image = pet_to_rgb(pet, display_size=display_size)
  predictor.set_image(image)
  height, width = np.asarray(pet).shape
  scale_x, scale_y = display_size / width, display_size / height
  prompt = np.array([
    box.xmin * scale_x, box.ymin * scale_y,
    box.xmax * scale_x, box.ymax * scale_y,
  ], dtype=np.float32)
  _, _, low_resolution_logits = predictor.predict(
    box=prompt,
    multimask_output=False,
    return_logits=True,
  )
  raw = np.asarray(low_resolution_logits[0], dtype=np.float32)
  if np.all((raw == 0.0) | (raw == 1.0)):
    raise RuntimeError("MedSAM2 returned a binary mask; continuous decoder output is required")
  # Match torch.nn.functional.interpolate(..., align_corners=False) without
  # importing torch in lightweight audit/test environments.
  from scipy.ndimage import map_coordinates
  output_y = (np.arange(output_shape[0]) + 0.5) * raw.shape[0] / output_shape[0] - 0.5
  output_x = (np.arange(output_shape[1]) + 0.5) * raw.shape[1] / output_shape[1] - 0.5
  yy, xx = np.meshgrid(
    np.clip(output_y, 0, raw.shape[0] - 1),
    np.clip(output_x, 0, raw.shape[1] - 1), indexing="ij",
  )
  return map_coordinates(
    raw, [yy, xx], order=1, mode="nearest", prefilter=False,
  ).astype(np.float32)


def freeze_except_mask_decoder(model: Any) -> None:
  """Train the single-mask L1 path; freeze encoders and disconnected decoder heads."""
  trainable = 0
  for name, parameter in model.named_parameters():
    parameter.requires_grad = is_single_mask_l1_trainable_parameter(name)
    trainable += int(parameter.requires_grad) * parameter.numel()
  if trainable == 0:
    raise RuntimeError("no mask-decoder parameters were selected for training")


def predictor_factory(
  *, medsam_source: Path, pretrained_checkpoint: Path,
) -> Callable[[CheckpointSpec], Any]:
  """Return a factory that loads the exact base and dataset decoder state."""
  def build(spec: CheckpointSpec) -> Any:
    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_medsam2(
      source_checkout=medsam_source,
      architecture="sam2.1_hiera_t", checkpoint=pretrained_checkpoint,
      device=device, mode="eval",
    )
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    state = torch.load(spec.path, map_location=device, weights_only=True)
    if (
      state.get("schema_version") != 1
      or state.get("artifact_kind") != "inference_mask_decoder"
      or state.get("dataset") != spec.dataset
      or state.get("normalization_id") != spec.normalization_id
      or state.get("run_id") != spec.run_id
      or state.get("pretrained_checkpoint_sha256") != sha256_file(pretrained_checkpoint)
    ):
      raise ValueError(f"checkpoint payload identity mismatch for {spec.dataset}")
    model.sam_mask_decoder.load_state_dict(state["mask_decoder_state"], strict=True)
    model.to(device).eval()
    return SAM2ImagePredictor(model)
  return build


def build_released_predictor(
  *, medsam_source: Path, pretrained_checkpoint: Path,
) -> Any:
  """Build the released, non-PET-fine-tuned model for a real ablation."""
  import torch

  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  model = build_medsam2(
    source_checkout=medsam_source,
    architecture="sam2.1_hiera_t", checkpoint=pretrained_checkpoint,
    device=device, mode="eval",
  )
  from sam2.sam2_image_predictor import SAM2ImagePredictor
  model.to(device).eval()
  return SAM2ImagePredictor(model)
