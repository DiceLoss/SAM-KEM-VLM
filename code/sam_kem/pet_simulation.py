"""Canonical activity scaling and forward projection for PET simulation."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .config import load_contract
from .noise import array_sha256
from .operator_factory import build_discrete_radon


def prepare_noise_free_projection(
  root: Path,
  config_path: Path,
  *,
  dataset: str,
  raw_activity_path: Path,
  normalized_target_path: Path,
  projection_path: Path,
) -> tuple[Path, Path]:
  cfg, contract = load_contract(config_path)
  if dataset not in contract.scale_constants:
    raise ValueError(f"unknown dataset: {dataset}")
  raw = np.load(raw_activity_path)
  if raw.shape != (128, 128):
    raise ValueError(
      f"activity must already be registered/resampled to 128x128; got {raw.shape}"
    )
  if not np.all(np.isfinite(raw)) or np.any(raw < 0) or float(raw.sum()) <= 0:
    raise ValueError("activity must be finite, non-negative and non-empty")
  image_size = int(cfg["forward_model"]["image_size"])
  centre = (image_size - 1) / 2.0
  yy, xx = np.indices(raw.shape, dtype=np.float64)
  outside = ((xx - centre) ** 2 + (yy - centre) ** 2) > centre ** 2
  outside_mass_fraction = float(raw[outside].sum() / raw.sum())
  fov_tolerance = float(cfg["forward_model"]["out_of_fov_activity_mass_tolerance"])
  if outside_mass_fraction > fov_tolerance:
    raise ValueError(
      "activity outside the declared circular FOV is "
      f"{outside_mass_fraction:.6g}, exceeding {fov_tolerance:.6g}; "
      "crop/mask by a documented preprocessing protocol before simulation"
    )
  scale_C = contract.scale_constants[dataset]
  offset = contract.normalization_offsets[dataset]
  in_fov_activity = np.asarray(raw, dtype=np.float64).copy()
  in_fov_activity[outside] = 0.0
  target = (in_fov_activity - offset) / scale_C
  if contract.normalization_clip:
    target = np.clip(target, *cfg["normalization"]["expected_target_range"])
  elif np.any(target < 0):
    raise ValueError("normalization produced negative activity without declared clipping")
  target_upper = float(cfg["normalization"]["expected_target_range"][1])
  target_tolerance = float(cfg["normalization"]["target_range_tolerance"])
  if float(target.max()) > target_upper + target_tolerance:
    raise ValueError(
      f"normalization scale={scale_C:g} gives target max={float(target.max()):.6g}, outside "
      f"the declared PSNR/data range [0,{target_upper:g}]; verify normalization provenance"
    )
  operator = build_discrete_radon(
    repo_root=root, size=int(cfg["forward_model"]["image_size"]),
    angles=contract.angles,
    operator_id=contract.operator_id,
    expected_implementation_sha256=cfg["forward_model"]["implementation_sha256"],
  )
  projection = operator.forward(target)
  if projection.shape != operator.projection_shape or np.any(projection < 0):
    raise RuntimeError("forward operator returned an invalid projection")
  expected_projection_sum = contract.angles * float(target.sum())
  mass_error = abs(float(projection.sum()) - expected_projection_sum) / expected_projection_sum
  if mass_error > 1e-12:
    raise RuntimeError(f"forward projection failed mass-conservation check: {mass_error:.3e}")
  metadata_path = projection_path.with_suffix(".json")
  for path in (normalized_target_path, projection_path, metadata_path):
    if path.exists():
      raise FileExistsError(f"refusing to overwrite PET simulation artifact: {path}")
  normalized_target_path.parent.mkdir(parents=True, exist_ok=True)
  projection_path.parent.mkdir(parents=True, exist_ok=True)
  np.save(normalized_target_path, target.astype(np.float32))
  np.save(projection_path, projection.astype(np.float64))
  metadata = {
    "schema_version": 1, "dataset": dataset,
    "raw_activity_path": str(raw_activity_path),
    "raw_activity_sha256": array_sha256(raw),
    "normalization_id": contract.normalization_id,
    "normalization_offset": offset,
    "normalization_scale_C": scale_C,
    "normalization_clip": contract.normalization_clip,
    "normalization": cfg["normalization"]["rule"],
    "circular_fov_mask_applied": True,
    "normalized_target_min": float(target.min()),
    "normalized_target_max": float(target.max()),
    "field_of_view": cfg["forward_model"]["field_of_view"],
    "out_of_fov_activity_mass_fraction": outside_mass_fraction,
    "out_of_fov_activity_mass_tolerance": fov_tolerance,
    "target_sha256": array_sha256(target.astype(np.float32)),
    "operator_id": contract.operator_id,
    "operator_implementation_sha256": cfg["forward_model"]["implementation_sha256"],
    "projection_shape": list(projection.shape),
    "noise_free_projection_sha256": array_sha256(projection.astype(np.float64)),
    "projection_mass_relative_error": mass_error,
    "noise_model": "none_at_this_stage",
  }
  metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
  return normalized_target_path, projection_path
