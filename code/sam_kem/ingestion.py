"""Manifest-driven ingestion of activity images into the canonical PET grid."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.ndimage import map_coordinates

from .artifacts import atomic_save_numpy, atomic_write_json, file_sha256
from .noise import array_sha256
from .splits import load_subject_splits
SOURCE_COLUMNS = (
  "dataset", "subject_id", "slice_id", "split", "source_path", "source_sha256",
  "source_kind", "source_key", "source_subject_index", "source_slice_axis",
  "source_slice_index", "source_channel_index", "activity_unit", "spatial_transform",
  "source_spacing_y_mm", "source_spacing_x_mm",
)
ALLOWED_SPLITS = {"train", "validation", "test_locked"}
ALLOWED_TRANSFORMS = {
  "mean_pool_2x2_full_square",
  "identity_128_full_square",
  "resize_full_square_to_128",
}


@dataclass(frozen=True)
class SourceCase:
  dataset: str
  subject_id: str
  slice_id: str
  split: str
  source_path: Path
  source_sha256: str
  source_kind: str
  source_key: str
  source_subject_index: int | None
  source_slice_axis: int | None
  source_slice_index: int | None
  source_channel_index: int | None
  activity_unit: str
  spatial_transform: str
  source_spacing_y_mm: float | None
  source_spacing_x_mm: float | None


def _optional_int(value: str) -> int | None:
  return None if value.strip() == "" else int(value)


def _optional_float(value: str) -> float | None:
  return None if value.strip() == "" else float(value)


def load_source_manifest(root: Path, path: Path) -> list[SourceCase]:
  with path.open("r", encoding="utf-8", newline="") as handle:
    reader = csv.DictReader(handle)
    missing = set(SOURCE_COLUMNS) - set(reader.fieldnames or ())
    if missing:
      raise ValueError(f"source manifest missing columns: {sorted(missing)}")
    rows = list(reader)
  if not rows:
    raise ValueError("source manifest is empty")
  cases = []
  for row in rows:
    if row["split"] not in ALLOWED_SPLITS:
      raise ValueError(f"invalid split {row['split']!r}")
    if row["spatial_transform"] not in ALLOWED_TRANSFORMS:
      raise ValueError(f"unapproved spatial transform {row['spatial_transform']!r}")
    source = (root / row["source_path"]).resolve()
    cases.append(SourceCase(
      dataset=row["dataset"], subject_id=row["subject_id"], slice_id=row["slice_id"],
      split=row["split"], source_path=source, source_sha256=row["source_sha256"],
      source_kind=row["source_kind"], source_key=row["source_key"],
      source_subject_index=_optional_int(row["source_subject_index"]),
      source_slice_axis=_optional_int(row["source_slice_axis"]),
      source_slice_index=_optional_int(row["source_slice_index"]),
      source_channel_index=_optional_int(row["source_channel_index"]),
      activity_unit=row["activity_unit"], spatial_transform=row["spatial_transform"],
      source_spacing_y_mm=_optional_float(row["source_spacing_y_mm"]),
      source_spacing_x_mm=_optional_float(row["source_spacing_x_mm"]),
    ))
  identities = [(case.dataset, case.subject_id, case.slice_id) for case in cases]
  if len(identities) != len(set(identities)):
    raise ValueError("source manifest contains duplicate dataset/subject/slice identities")
  return cases


def _take_slice(volume: np.ndarray, axis: int, index: int) -> np.ndarray:
  if axis < 0 or axis >= volume.ndim or index < 0 or index >= volume.shape[axis]:
    raise IndexError(f"slice axis/index {axis}/{index} is invalid for shape {volume.shape}")
  return np.take(volume, index, axis=axis)


def load_source_activity(case: SourceCase) -> tuple[np.ndarray, dict]:
  if not case.source_path.is_file():
    raise FileNotFoundError(f"activity source is missing: {case.source_path}")
  actual_source_sha = file_sha256(case.source_path)
  if actual_source_sha != case.source_sha256:
    raise ValueError(f"source SHA-256 mismatch: {case.source_path}")
  metadata: dict = {"source_file_sha256": actual_source_sha}
  if case.source_kind == "npy_slice":
    value = np.load(case.source_path, allow_pickle=False)
  elif case.source_kind == "npz_hecktor_input":
    required = (
      case.source_key, case.source_subject_index, case.source_slice_index,
      case.source_channel_index,
    )
    if any(item is None or item == "" for item in required):
      raise ValueError("npz_hecktor_input requires key, subject, slice, and channel indices")
    with np.load(case.source_path, allow_pickle=False) as archive:
      array = archive[case.source_key]
      if array.ndim != 5:
        raise ValueError(f"HECKTOR input must be 5-D; got {array.shape}")
      value = array[
        case.source_subject_index, case.source_slice_index, :, :, case.source_channel_index,
      ]
  elif case.source_kind == "nifti_volume":
    if case.source_slice_axis is None or case.source_slice_index is None:
      raise ValueError("nifti_volume requires source_slice_axis and source_slice_index")
    try:
      import nibabel as nib
    except ImportError as error:
      raise RuntimeError("install the project 'data' extra to read NIfTI sources") from error
    image = nib.load(str(case.source_path))
    volume = np.asanyarray(image.dataobj)
    value = _take_slice(volume, case.source_slice_axis, case.source_slice_index)
    zooms = image.header.get_zooms()
    in_plane_axes = [axis for axis in range(3) if axis != case.source_slice_axis]
    metadata["nifti_affine"] = np.asarray(image.affine).tolist()
    metadata["nifti_zooms"] = [float(item) for item in zooms]
    metadata["nifti_in_plane_spacing_mm"] = [float(zooms[axis]) for axis in in_plane_axes]
  else:
    raise ValueError(f"unsupported source_kind {case.source_kind!r}")
  activity = np.asarray(value, dtype=np.float32)
  if activity.ndim != 2 or not np.all(np.isfinite(activity)):
    raise ValueError(f"source activity must be a finite 2-D slice; got {activity.shape}")
  if np.any(activity < 0) or float(activity.sum()) <= 0:
    raise ValueError("source activity must be non-negative and non-empty")
  metadata["source_array_sha256"] = array_sha256(activity)
  metadata["source_shape"] = list(activity.shape)
  metadata["source_dtype_after_float32_cast"] = activity.dtype.str
  return activity, metadata


def mean_pool_2x2(activity: np.ndarray) -> np.ndarray:
  value = np.asarray(activity, dtype=np.float64)
  if value.shape != (256, 256):
    raise ValueError(f"2x2 mean pooling requires 256x256 input; got {value.shape}")
  return value.reshape(128, 2, 128, 2).mean(axis=(1, 3))


def resize_full_square(activity: np.ndarray, *, output_size: int = 128) -> np.ndarray:
  """Bilinearly resize a complete square using pixel-centre coordinates.

  PET voxel values are treated as activity concentration, so the interpolation
  preserves the numeric scale rather than forcing conservation of the array sum.
  No anatomical crop, scanner-FOV mask, or content-dependent recentering is used.
  """
  value = np.asarray(activity, dtype=np.float64)
  if value.ndim != 2 or value.shape[0] != value.shape[1] or output_size < 2:
    raise ValueError("full-square resize requires a square 2-D image and output_size >= 2")
  source_coordinate = (
    (np.arange(output_size, dtype=np.float64) + 0.5) * value.shape[0] / output_size
    - 0.5
  )
  source_rows, source_columns = np.meshgrid(
    source_coordinate, source_coordinate, indexing="ij",
  )
  return map_coordinates(
    value, [source_rows, source_columns], order=1, mode="nearest",
    prefilter=False,
  )


def transform_activity(
  activity: np.ndarray, case: SourceCase,
) -> np.ndarray:
  if case.spatial_transform == "mean_pool_2x2_full_square":
    transformed = mean_pool_2x2(activity)
  elif case.spatial_transform == "identity_128_full_square":
    if activity.shape != (128, 128):
      raise ValueError(f"identity transform requires 128x128 input; got {activity.shape}")
    transformed = np.asarray(activity, dtype=np.float64).copy()
  else:
    transformed = resize_full_square(activity, output_size=128)
  transformed = np.asarray(transformed, dtype=np.float32)
  if transformed.shape != (128, 128) or float(transformed.sum()) <= 0:
    raise ValueError("spatial transform did not produce a non-empty canonical 128x128 image")
  return transformed


def ingest_case(
  root: Path, case: SourceCase, output_root: Path,
) -> Path:
  source, source_metadata = load_source_activity(case)
  activity = transform_activity(source, case)
  case_dir = output_root / case.dataset / case.subject_id / case.slice_id
  output = case_dir / "raw_activity.npy"
  sidecar = output.with_suffix(".json")
  if output.exists() or sidecar.exists():
    raise FileExistsError(f"refusing to overwrite canonical activity: {output}")
  atomic_save_numpy(output, activity)
  try:
    atomic_write_json(sidecar, {
      "schema_version": 1, "artifact_kind": "canonical_activity",
      "dataset": case.dataset, "subject_id": case.subject_id, "slice_id": case.slice_id,
      "split": case.split, "activity_unit": case.activity_unit,
      "source_path": str(case.source_path), "source_kind": case.source_kind,
      "source_key": case.source_key,
      "source_subject_index": case.source_subject_index,
      "source_slice_axis": case.source_slice_axis,
      "source_slice_index": case.source_slice_index,
      "source_channel_index": case.source_channel_index,
      "spatial_transform": case.spatial_transform,
      "source_spacing_y_mm": case.source_spacing_y_mm,
      "source_spacing_x_mm": case.source_spacing_x_mm,
      "support_policy": "complete_square_no_content_crop_or_circular_mask",
      "activity_shape": list(activity.shape), "activity_dtype": activity.dtype.str,
      "activity_min": float(activity.min()), "activity_max": float(activity.max()),
      "activity_sha256": array_sha256(activity), **source_metadata,
    })
  except BaseException:
    output.unlink(missing_ok=True)
    raise
  return output


def ingest_manifest(
  root: Path, manifest_path: Path, split_path: Path, output_root: Path, *, splits: set[str],
) -> list[Path]:
  if not splits or not splits <= ALLOWED_SPLITS:
    raise ValueError(f"invalid requested splits: {sorted(splits)}")
  assignments = {
    (row["dataset"], row["subject_id"]): row["split"]
    for row in load_subject_splits(split_path)
  }
  all_cases = load_source_manifest(root, manifest_path)
  for case in all_cases:
    identity = (case.dataset, case.subject_id)
    if identity not in assignments:
      raise ValueError(f"source case has no canonical subject split: {identity}")
    if assignments[identity] != case.split:
      raise ValueError(
        f"source manifest split mismatch for {identity}: "
        f"{case.split} != {assignments[identity]}"
      )
  cases = [case for case in all_cases if case.split in splits]
  if not cases:
    raise ValueError("no source cases match the requested splits")
  outputs = []
  for case in cases:
    outputs.append(ingest_case(root, case, output_root))
  return outputs
