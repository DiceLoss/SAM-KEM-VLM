"""Canonical activity scaling and forward projection for PET simulation."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .artifacts import atomic_save_numpy, atomic_write_json, file_sha256
from .config import load_contract
from .noise import array_sha256
from .operator_factory import build_discrete_radon


@dataclass(frozen=True)
class ProjectionContext:
  cfg: dict
  contract: object
  operator: object

  @classmethod
  def build(cls, root: Path, config_path: Path) -> "ProjectionContext":
    cfg, contract = load_contract(config_path)
    operator = build_discrete_radon(
      repo_root=root, size=int(cfg["forward_model"]["image_size"]),
      angles=contract.angles, detector_bins=contract.detector_bins,
      operator_id=contract.operator_id,
      expected_implementation_sha256=cfg["forward_model"]["implementation_sha256"],
    )
    return cls(cfg, contract, operator)


def prepare_noise_free_projection(
  root: Path,
  config_path: Path,
  *,
  dataset: str,
  raw_activity_path: Path,
  normalized_target_path: Path,
  projection_path: Path,
  context: ProjectionContext | None = None,
) -> tuple[Path, Path]:
  active = context or ProjectionContext.build(root, config_path)
  cfg, contract, operator = active.cfg, active.contract, active.operator
  if dataset not in contract.scale_constants:
    raise ValueError(f"unknown dataset: {dataset}")
  raw = np.load(raw_activity_path)
  ingestion_path = raw_activity_path.with_suffix(".json")
  if not ingestion_path.is_file():
    raise FileNotFoundError(f"canonical activity provenance is missing: {ingestion_path}")
  ingestion = json.loads(ingestion_path.read_text(encoding="utf-8"))
  if (
    ingestion.get("schema_version") != 1
    or ingestion.get("artifact_kind") != "canonical_activity"
    or ingestion.get("dataset") != dataset
    or ingestion.get("activity_sha256") != array_sha256(raw)
  ):
    raise ValueError(f"canonical activity provenance mismatch: {ingestion_path}")
  if raw.shape != (128, 128):
    raise ValueError(
      f"activity must already be registered/resampled to 128x128; got {raw.shape}"
    )
  if not np.all(np.isfinite(raw)) or np.any(raw < 0) or float(raw.sum()) <= 0:
    raise ValueError("activity must be finite, non-negative and non-empty")
  scale_C = contract.scale_constants[dataset]
  offset = contract.normalization_offsets[dataset]
  target = (np.asarray(raw, dtype=np.float64) - offset) / scale_C
  if contract.normalization_clip:
    target = np.clip(target, *cfg["normalization"]["expected_target_range"])
  elif np.any(target < 0):
    raise ValueError("normalization produced negative activity without declared clipping")
  target_upper = float(cfg["normalization"]["expected_target_range"][1])
  # Project the exact float32 target that is published and later scored.
  # Projecting a transient float64 value would create a subtle target/operator
  # mismatch even though both arrays originated from the same raw image.
  target_float32 = target.astype(np.float32)
  projection = operator.forward(target_float32)
  if projection.shape != operator.projection_shape or np.any(projection < 0):
    raise RuntimeError("forward operator returned an invalid projection")
  expected_projection_sum = contract.angles * float(target_float32.sum(dtype=np.float64))
  mass_error = abs(float(projection.sum()) - expected_projection_sum) / expected_projection_sum
  if mass_error > 1e-12:
    raise RuntimeError(f"forward projection failed mass-conservation check: {mass_error:.3e}")
  metadata_path = projection_path.with_suffix(".json")
  for path in (normalized_target_path, projection_path, metadata_path):
    if path.exists():
      raise FileExistsError(f"refusing to overwrite PET simulation artifact: {path}")
  projection_float64 = projection.astype(np.float64)
  atomic_save_numpy(normalized_target_path, target_float32)
  try:
    atomic_save_numpy(projection_path, projection_float64)
  except BaseException:
    normalized_target_path.unlink(missing_ok=True)
    raise
  metadata = {
    "schema_version": 2, "artifact_kind": "noise_free_pet_projection",
    "dataset": dataset,
    "raw_activity_path": str(raw_activity_path),
    "raw_activity_sha256": array_sha256(raw),
    "ingestion_metadata_path": str(ingestion_path),
    "ingestion_metadata_sha256": file_sha256(ingestion_path),
    "normalization_id": contract.normalization_id,
    "normalization_offset": offset,
    "normalization_scale_C": scale_C,
    "normalization_clip": contract.normalization_clip,
    "normalization": cfg["normalization"]["rule"],
    "support_policy": "complete_square_no_content_crop_or_circular_mask",
    "normalized_target_min": float(target.min()),
    "normalized_target_max": float(target.max()),
    "normalized_target_above_reference_fraction": float(np.mean(target > target_upper)),
    "field_of_view": cfg["forward_model"]["field_of_view"],
    "detector_bins": contract.detector_bins,
    "target_sha256": array_sha256(target_float32),
    "operator_id": contract.operator_id,
    "operator_implementation_sha256": cfg["forward_model"]["implementation_sha256"],
    "projection_shape": list(projection.shape),
    "noise_free_projection_sha256": array_sha256(projection_float64),
    "projection_mass_relative_error": mass_error,
    "noise_model": "none_at_this_stage",
  }
  try:
    atomic_write_json(metadata_path, metadata)
  except BaseException:
    normalized_target_path.unlink(missing_ok=True)
    projection_path.unlink(missing_ok=True)
    raise
  return normalized_target_path, projection_path
