"""MedSAM2 loading and continuous decoder-output inference."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np

from .boxes import Box


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
    self._predictors: dict[tuple[str, str], Any] = {}

  def get(self, checkpoint: CheckpointSpec) -> Any:
    checkpoint.verify()
    key = (checkpoint.dataset, checkpoint.sha256)
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
  """Return the raw single-mask decoder output requested with logits enabled."""
  image = pet_to_rgb(pet, display_size=display_size)
  predictor.set_image(image)
  height, width = np.asarray(pet).shape
  scale_x, scale_y = display_size / width, display_size / height
  prompt = np.array([
    box.xmin * scale_x, box.ymin * scale_y,
    box.xmax * scale_x, box.ymax * scale_y,
  ], dtype=np.float32)
  masks, scores, logits = predictor.predict(
    box=prompt,
    multimask_output=False,
    return_logits=True,
  )
  raw = np.asarray(masks[0], dtype=np.float32)
  if np.all((raw == 0.0) | (raw == 1.0)):
    raise RuntimeError("MedSAM2 returned a binary mask; continuous decoder output is required")
  from scipy.ndimage import zoom
  factors = (output_shape[0] / raw.shape[0], output_shape[1] / raw.shape[1])
  return zoom(raw, factors, order=1).astype(np.float32)


def freeze_except_mask_decoder(model: Any) -> None:
  """Freeze every parameter except those owned by the mask decoder."""
  trainable = 0
  for name, parameter in model.named_parameters():
    parameter.requires_grad = "mask_decoder" in name
    trainable += int(parameter.requires_grad) * parameter.numel()
  if trainable == 0:
    raise RuntimeError("no mask-decoder parameters were selected for training")


def predictor_factory(
  *, repo_root: Path, medsam_source: Path, pretrained_checkpoint: Path,
) -> Callable[[CheckpointSpec], Any]:
  """Return a factory that loads the exact base and dataset decoder state."""
  def build(spec: CheckpointSpec) -> Any:
    import sys
    import torch

    sys.path.insert(0, str(medsam_source))
    original = repo_root / "code/src/original"; sys.path.insert(0, str(original))
    import build_medsam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_medsam2.build_medsam2_hiera_t(
      image_size=1024, checkpoint=str(pretrained_checkpoint), mode="eval", reconfigure=False,
    )
    state = torch.load(spec.path, map_location=device)
    if (
      state.get("schema_version") != 1
      or state.get("artifact_kind") != "inference_mask_decoder"
      or state.get("dataset") != spec.dataset
      or state.get("normalization_id") != spec.normalization_id
      or state.get("pretrained_checkpoint_sha256") != sha256_file(pretrained_checkpoint)
    ):
      raise ValueError(f"checkpoint payload identity mismatch for {spec.dataset}")
    model.sam_mask_decoder.load_state_dict(state["mask_decoder_state"], strict=True)
    model.to(device).eval()
    return SAM2ImagePredictor(model)
  return build


def build_released_predictor(
  *, repo_root: Path, medsam_source: Path, pretrained_checkpoint: Path,
) -> Any:
  """Build the released, non-PET-fine-tuned model for a real ablation."""
  import sys
  import torch

  sys.path.insert(0, str(medsam_source))
  original = repo_root / "code/src/original"; sys.path.insert(0, str(original))
  import build_medsam2
  from sam2.sam2_image_predictor import SAM2ImagePredictor

  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  model = build_medsam2.build_medsam2_hiera_t(
    image_size=1024, checkpoint=str(pretrained_checkpoint), mode="eval", reconfigure=False,
  )
  model.to(device).eval()
  return SAM2ImagePredictor(model)
