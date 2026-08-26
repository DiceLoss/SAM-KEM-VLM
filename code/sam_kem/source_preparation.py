"""Deterministic preparation of development PET planes under subject-level splits."""

from __future__ import annotations

import json
import shutil
import tempfile
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from .artifacts import (
  atomic_csv_writer, atomic_save_numpy, atomic_write_json, file_sha256, finalize_csv,
)
from .config import load_yaml_strict
from .ingestion import SOURCE_COLUMNS
from .noise import array_sha256
from .splits import load_configured_subject_splits


def _manifest_path(root: Path, path: Path) -> str:
  resolved = path.resolve()
  try:
    return resolved.relative_to(root.resolve()).as_posix()
  except ValueError:
    return str(resolved)


def _extract_npz_member(archive_path: Path, key: str, destination: Path) -> Path:
  """Stream one NPY member out of an NPZ without loading unrelated arrays."""
  with zipfile.ZipFile(archive_path) as archive:
    matches = [name for name in archive.namelist() if Path(name).name == f"{key}.npy"]
    if len(matches) != 1:
      raise ValueError(f"expected exactly one {key!r} member in {archive_path}; got {matches}")
    with archive.open(matches[0]) as source, destination.open("wb") as target:
      shutil.copyfileobj(source, target, length=16 * 1024 * 1024)
  return destination


def _copy_hecktor_pet_slice(input_array: np.ndarray, subject_index: int) -> np.ndarray:
  """Return an owned PET slice that never keeps the archive mmap alive."""
  pet = np.array(
    input_array[subject_index, 72, :, :, 1], dtype=np.float32, copy=True,
  )
  if pet.shape != (144, 144) or np.any(pet < 0) or not np.all(np.isfinite(pet)):
    raise ValueError(f"invalid HECKTOR PET slice for subject index {subject_index}")
  return pet


def _usable_brainweb_slice_indices(volume: np.ndarray) -> list[int]:
  """Select every finite, nonnegative axial plane with positive activity mass."""
  value = np.asarray(volume, dtype=np.float32)
  if value.ndim != 3 or value.shape[:2] != (256, 256):
    raise ValueError(f"unexpected BrainWeb source shape: {value.shape}")
  if not np.all(np.isfinite(value)) or np.any(value < 0):
    raise ValueError("BrainWeb volume contains non-finite or negative activity")
  masses = value.sum(axis=(0, 1), dtype=np.float64)
  selected = np.flatnonzero(masses > 0.0).astype(int).tolist()
  if not selected:
    raise ValueError("BrainWeb volume has no positive-activity axial plane")
  return selected


def prepare_development_sources(
  root: Path,
  config_path: Path,
  *,
  brainweb_root: Path,
  hecktor_npz: Path,
  hecktor_unit: str,
  output_root: Path,
  manifest_path: Path,
) -> Path:
  """Create a SHA-bound source manifest without accessing HE segmentation labels."""
  root = root.resolve()
  cfg = load_yaml_strict(config_path)
  if cfg.get("status") != "development_only_no_locked_test" or cfg.get("locked") is not False:
    raise ValueError("source preparation requires the explicit unlocked development-only state")
  splits = load_configured_subject_splits(root, cfg)
  if any(row["split"] == "test_locked" for row in splits):
    raise ValueError("development source preparation refuses a split containing test_locked rows")
  if not hecktor_unit.strip():
    raise ValueError("hecktor_unit must record the verified DICOM-export PET value unit")
  if manifest_path.exists() or manifest_path.with_suffix(manifest_path.suffix + ".manifest.json").exists():
    raise FileExistsError(f"refusing to overwrite source manifest: {manifest_path}")
  geometry_path = root / "inputs/clean_v1/hecktor_geometry.json"
  geometry = json.loads(geometry_path.read_text(encoding="utf-8"))
  expected_npz_sha = geometry["hecktor_npz_layout"]["sha256"]
  actual_npz_sha = file_sha256(hecktor_npz)
  if actual_npz_sha != expected_npz_sha:
    raise ValueError(f"HECKTOR.npz SHA mismatch: {actual_npz_sha} != {expected_npz_sha}")

  output_root.mkdir(parents=True, exist_ok=True)
  rows: list[dict[str, object]] = []
  brainweb_rows = [row for row in splits if row["dataset"] == "brainweb"]
  for split_row in brainweb_rows:
    subject = split_row["subject_id"]
    source = brainweb_root / subject / "sim_0" / "true_pet.nii.gz"
    if not source.is_file():
      raise FileNotFoundError(source)
    try:
      import nibabel as nib
    except ImportError as error:
      raise RuntimeError("install the project 'data' extra to inspect BrainWeb NIfTI") from error
    image = nib.load(str(source))
    volume = np.asarray(image.dataobj, dtype=np.float32)
    usable_indices = _usable_brainweb_slice_indices(volume)
    zooms = image.header.get_zooms()
    upstream_sha = file_sha256(source)
    for slice_index in usable_indices:
      selected = np.array(volume[:, :, slice_index], dtype=np.float32, copy=True)
      extracted = output_root / "brainweb" / subject / f"z{slice_index:03d}_pet.npy"
      if extracted.exists() or extracted.with_suffix(".json").exists():
        raise FileExistsError(f"refusing to overwrite extracted source: {extracted}")
      atomic_save_numpy(extracted, selected)
      atomic_write_json(extracted.with_suffix(".json"), {
        "schema_version": 1, "artifact_kind": "brainweb_simulation_pet_slice",
        "dataset": "brainweb", "subject_id": subject,
        "slice_id": f"z{slice_index:03d}",
        "activity_unit": "BrainWeb_simulation_activity",
        "array_sha256": array_sha256(selected),
        "source_file_sha256": file_sha256(extracted),
        "upstream_container_path": str(source.resolve()),
        "upstream_container_sha256": upstream_sha,
        "upstream_selector": f"dataobj[:,:,{slice_index}]",
        "selection_rule": "all_finite_nonnegative_axial_planes_with_positive_activity_mass",
      })
      rows.append({
        "dataset": "brainweb", "subject_id": subject,
        "slice_id": f"z{slice_index:03d}", "split": split_row["split"],
        "source_path": _manifest_path(root, extracted),
        "source_sha256": file_sha256(extracted),
        "source_kind": "npy_slice", "source_key": "", "source_subject_index": "",
        "source_slice_axis": 2, "source_slice_index": slice_index,
        "source_channel_index": "", "activity_unit": "BrainWeb_simulation_activity",
        "spatial_transform": "mean_pool_2x2_full_square",
        "source_spacing_y_mm": float(zooms[0]), "source_spacing_x_mm": float(zooms[1]),
        "upstream_container_path": str(source.resolve()),
        "upstream_container_sha256": upstream_sha,
        "upstream_selector": f"dataobj[:,:,{slice_index}]",
      })
    del volume

  # Manually manage the temporary directory so we can release the mmap
  # handle and the underlying input.npy file BEFORE rmtree runs. On
  # Windows the read-only mmap keeps a file lock that blocks os.unlink
  # inside TemporaryDirectory.__exit__; on Linux this is unnecessary but
  # the explicit close is also harmless. The streaming extraction itself
  # is unchanged: only HECKTOR.npz 'input' is read, no other member.
  temporary = Path(tempfile.mkdtemp(prefix="hecktor_input_", dir=output_root.parent))
  input_array = None
  try:
    input_npy = _extract_npz_member(hecktor_npz, "input", temporary / "input.npy")
    input_array = np.load(input_npy, allow_pickle=False, mmap_mode="r")
    if input_array.shape != (224, 144, 144, 144, 2):
      raise ValueError(f"unexpected HECKTOR input shape: {input_array.shape}")
    for split_row in (row for row in splits if row["dataset"] == "hecktor"):
      subject = split_row["subject_id"]
      subject_index = int(subject) - 1
      slice_index = 72
      channel_index = 1
      pet = _copy_hecktor_pet_slice(input_array, subject_index)
      source = output_root / "hecktor" / subject / "z072_pet.npy"
      if source.exists() or source.with_suffix(".json").exists():
        raise FileExistsError(f"refusing to overwrite extracted source: {source}")
      atomic_save_numpy(source, pet)
      atomic_write_json(source.with_suffix(".json"), {
        "schema_version": 1, "artifact_kind": "hecktor_dicom_export_pet_slice",
        "dataset": "hecktor", "subject_id": subject, "slice_id": "z072",
        "activity_unit": hecktor_unit, "array_sha256": array_sha256(pet),
        "source_file_sha256": file_sha256(source),
        "upstream_container_path": str(hecktor_npz.resolve()),
        "upstream_container_sha256": actual_npz_sha,
        "upstream_selector": f"input[{subject_index},72,:,:,1]",
        "researcher_source_statement": "reconstructed clinical PET exported directly from DICOM",
      })
      rows.append({
        "dataset": "hecktor", "subject_id": subject, "slice_id": "z072",
        "split": split_row["split"], "source_path": _manifest_path(root, source),
        "source_sha256": file_sha256(source), "source_kind": "npy_slice",
        "source_key": "", "source_subject_index": subject_index,
        "source_slice_axis": "", "source_slice_index": slice_index,
        "source_channel_index": channel_index, "activity_unit": hecktor_unit,
        "spatial_transform": "resize_full_square_to_128",
        "source_spacing_y_mm": "", "source_spacing_x_mm": "",
        "upstream_container_path": str(hecktor_npz.resolve()),
        "upstream_container_sha256": actual_npz_sha,
        "upstream_selector": f"input[{subject_index},72,:,:,1]",
      })
  finally:
    if input_array is not None:
      memory_map = getattr(input_array, "_mmap", None)
      if memory_map is not None:
        memory_map.close()
      del input_array
    # Scratch cleanup is part of success: never silently leave a many-GB copy.
    shutil.rmtree(temporary)

  actual_plane_counts: dict[str, Counter] = defaultdict(Counter)
  for row in rows:
    actual_plane_counts[str(row["dataset"])][str(row["split"])] += 1
  expected_plane_counts = cfg["data"].get("expected_source_plane_counts", {})
  mismatches = {}
  for dataset, expected in expected_plane_counts.items():
    actual = actual_plane_counts.get(dataset, Counter())
    split_names = set(actual) | set(expected)
    if any(int(actual.get(name, 0)) != int(expected.get(name, 0)) for name in split_names):
      mismatches[dataset] = {"actual": dict(actual), "expected": expected}
  if mismatches:
    raise ValueError(f"source-plane counts differ from the pinned protocol: {mismatches}")

  with atomic_csv_writer(manifest_path, list(SOURCE_COLUMNS)) as writer:
    writer.writerows(sorted(rows, key=lambda row: (str(row["dataset"]), str(row["subject_id"]))))
  finalize_csv(
    manifest_path, schema_name="source_activity_manifest_v2",
    primary_key=["dataset", "subject_id", "slice_id"],
  )
  return manifest_path
