"""Validation of the small source-traceability PET sample archives."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np


def file_sha256(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as handle:
    for block in iter(lambda: handle.read(1024 * 1024), b""):
      digest.update(block)
  return digest.hexdigest()


def circular_mask(size: int) -> np.ndarray:
  yy, xx = np.indices((size, size), dtype=np.float64)
  centre = (size - 1) / 2.0
  return (xx - centre) ** 2 + (yy - centre) ** 2 <= centre ** 2


def largest_centred_square_in_circle(size: int) -> int:
  """Largest same-parity square whose pixel centres fit in the FOV circle."""
  candidate = int(np.floor(1.0 + (size - 1) / np.sqrt(2.0)))
  if (size - candidate) % 2:
    candidate -= 1
  return candidate


def _prefixes(files: list[str], suffix: str) -> list[str]:
  return sorted(key[: -len(suffix)] for key in files if key.endswith(suffix))


def _scalar(archive: np.lib.npyio.NpzFile, key: str):
  value = archive[key]
  if value.shape != ():
    raise ValueError(f"{key} must be a scalar")
  return value.item()


def _validate_activity(value: np.ndarray, key: str, shape: tuple[int, int]) -> None:
  if value.shape != shape or value.dtype != np.float32:
    raise ValueError(f"{key} must be float32 with shape {shape}; got {value.dtype} {value.shape}")
  if not np.all(np.isfinite(value)) or np.any(value < 0) or float(value.sum()) <= 0:
    raise ValueError(f"{key} must be finite, non-negative, and non-empty")


def audit_traceability_archives(root: Path) -> dict:
  sample_dir = root / "data/raw_samples"
  manifest_path = sample_dir / "TRACEABILITY_MANIFEST.json"
  manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
  result: dict = {
    "status": "traceability_only_not_paper_results",
    "manifest": str(manifest_path.relative_to(root)),
    "manifest_sha256": file_sha256(manifest_path),
    "locked_test_present": False,
    "datasets": {},
    "unresolved": [
      "source-file and per-array SHA-256 values are absent from the initial committed records",
      "HECKTOR source NIfTI/DICOM metadata is absent, so the SUV declaration is provider-supplied",
      "HECKTOR voxel spacing, affine/orientation, and scanner transaxial FOV are absent",
      "central traceability slices are insufficient to fit a dataset-wide normalization constant",
    ],
  }

  bw_path = root / Path(manifest["brainweb"]["file"])
  bw_file_hash = file_sha256(bw_path)
  if bw_file_hash != manifest["brainweb"].get("file_sha256"):
    raise ValueError("BrainWeb traceability archive SHA-256 does not match its manifest")
  with np.load(bw_path, allow_pickle=False) as archive:
    prefixes = _prefixes(archive.files, "__raw256")
    cases = []
    target_circle = circular_mask(128)
    for prefix in prefixes:
      raw = archive[f"{prefix}__raw256"]
      target = archive[f"{prefix}__target128"]
      _validate_activity(raw, f"{prefix}__raw256", (256, 256))
      _validate_activity(target, f"{prefix}__target128", (128, 128))
      expected = raw.reshape(128, 2, 128, 2).mean(axis=(1, 3), dtype=np.float32)
      split = str(_scalar(archive, f"{prefix}__split"))
      if split == "test_locked":
        result["locked_test_present"] = True
      cases.append({
        "case": prefix, "split": split,
        "slice_idx": int(_scalar(archive, f"{prefix}__slice_idx")),
        "raw_max": float(raw.max()), "target_max": float(target.max()),
        "target_outside_circular_fov_mass_fraction": float(
          target[~target_circle].sum(dtype=np.float64) / target.sum(dtype=np.float64)
        ),
        "mean_pool_max_abs_error": float(np.max(np.abs(target - expected))),
        "mean_pool_mass_ratio": float(4.0 * target.sum(dtype=np.float64) / raw.sum(dtype=np.float64)),
      })
  result["datasets"]["brainweb"] = {
    "archive": str(bw_path.relative_to(root)), "archive_sha256": bw_file_hash,
    "activity_unit": manifest["brainweb"].get("activity_unit", "not declared"),
    "cases": cases,
    "train_target_max": max(row["target_max"] for row in cases if row["split"] == "train"),
    "validation_target_max": max(
      row["target_max"] for row in cases if row["split"] == "validation"
    ),
  }

  he_path = root / Path(manifest["hecktor"]["file"])
  he_file_hash = file_sha256(he_path)
  if he_file_hash != manifest["hecktor"].get("file_sha256"):
    raise ValueError("HECKTOR traceability archive SHA-256 does not match its manifest")
  with np.load(he_path, allow_pickle=False) as archive:
    prefixes = _prefixes(archive.files, "__pet_input")
    cases = []
    native_circle = circular_mask(144)
    for prefix in prefixes:
      pet = archive[f"{prefix}__pet_input"]
      ct = archive[f"{prefix}__ct"]
      _validate_activity(pet, f"{prefix}__pet_input", (144, 144))
      if ct.shape != (144, 144) or ct.dtype != np.float32 or not np.all(np.isfinite(ct)):
        raise ValueError(f"{prefix}__ct must be finite float32 with shape (144, 144)")
      subject_id = int(prefix.split("__", 1)[0])
      axis0 = int(_scalar(archive, f"{prefix}__npz_axis0"))
      split = str(_scalar(archive, f"{prefix}__split"))
      if split == "test_locked":
        result["locked_test_present"] = True
      cases.append({
        "case": prefix, "split": split,
        "slice_idx": int(_scalar(archive, f"{prefix}__slice_idx")),
        "npz_axis0": axis0, "subject_minus_one_mapping_valid": axis0 == subject_id - 1,
        "pet_min": float(pet.min()), "pet_max": float(pet.max()),
        "pet_mean": float(pet.mean()),
        "native_inscribed_circle_outside_mass_fraction": float(
          pet[~native_circle].sum(dtype=np.float64) / pet.sum(dtype=np.float64)
        ),
      })
  outside = [row["native_inscribed_circle_outside_mass_fraction"] for row in cases]
  result["datasets"]["hecktor"] = {
    "archive": str(he_path.relative_to(root)), "archive_sha256": he_file_hash,
    "activity_unit": manifest["hecktor"].get("pet_unit", "not declared"),
    "cases": cases,
    "all_subject_minus_one_mappings_valid": all(
      row["subject_minus_one_mapping_valid"] for row in cases
    ),
    "outside_circle_mass_fraction_min": min(outside),
    "outside_circle_mass_fraction_mean": float(np.mean(outside)),
    "outside_circle_mass_fraction_max": max(outside),
    "largest_square_side_fitting_128_circular_fov": largest_centred_square_in_circle(128),
    "train_pet_max": max(row["pet_max"] for row in cases if row["split"] == "train"),
    "validation_pet_max": max(row["pet_max"] for row in cases if row["split"] == "validation"),
  }
  if result["locked_test_present"]:
    raise ValueError("traceability archives must never contain locked-test cases")
  return result
