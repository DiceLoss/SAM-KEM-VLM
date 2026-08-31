"""Read-only inventory of candidate anatomy sources for comparison baselines."""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import numpy as np

from .artifacts import atomic_write_json, file_sha256, verify_csv_artifact


def brainweb_subject_root(activity_path: Path) -> Path:
  """Resolve `<subject>` from `<subject>/sim_0/true_pet.nii.gz`."""
  if activity_path.name.lower() != "true_pet.nii.gz" or activity_path.parent.name != "sim_0":
    raise ValueError(f"unexpected BrainWeb activity layout: {activity_path}")
  return activity_path.parent.parent


def summarize_ct_slices(slices: list[np.ndarray]) -> dict:
  if not slices:
    raise ValueError("CT inventory cannot be empty")
  shapes = sorted({tuple(np.asarray(array).shape) for array in slices})
  nonfinite = sum(int(np.size(array) - np.isfinite(array).sum()) for array in slices)
  minima = [float(np.nanmin(array)) for array in slices]
  maxima = [float(np.nanmax(array)) for array in slices]
  return {
    "records": len(slices),
    "unique_shapes": [list(shape) for shape in shapes],
    "nonfinite_values": nonfinite,
    "slice_min_range": [min(minima), max(minima)],
    "slice_max_range": [min(maxima), max(maxima)],
    "overall_min": min(minima),
    "overall_max": max(maxima),
  }


def audit_anatomy_sources(
  root: Path, source_manifest_path: Path, output_path: Path,
) -> dict:
  """Inventory candidates without selecting, transforming or writing anatomy arrays."""
  root = root.resolve()
  source_manifest_path = source_manifest_path.resolve()
  output_path = output_path.resolve()
  if output_path.exists():
    raise FileExistsError(f"refusing to overwrite anatomy inventory: {output_path}")
  record = verify_csv_artifact(source_manifest_path)
  if record.get("schema_name") != "source_activity_manifest_v2":
    raise ValueError("anatomy inventory requires the canonical source activity manifest")
  with source_manifest_path.open("r", encoding="utf-8", newline="") as handle:
    rows = list(csv.DictReader(handle))
  if not rows:
    raise ValueError("source activity manifest is empty")

  brainweb_rows = defaultdict(list)
  hecktor_rows = []
  for row in rows:
    if row["dataset"] == "brainweb":
      brainweb_rows[row["subject_id"]].append(row)
    elif row["dataset"] == "hecktor":
      hecktor_rows.append(row)
    else:
      raise ValueError(f"unexpected dataset in source manifest: {row['dataset']}")

  import nibabel as nib

  brainweb_subjects = {}
  for subject_id, subject_rows in sorted(brainweb_rows.items()):
    upstream_paths = {Path(row["upstream_container_path"]) for row in subject_rows}
    upstream_shas = {row["upstream_container_sha256"] for row in subject_rows}
    if len(upstream_paths) != 1 or len(upstream_shas) != 1:
      raise ValueError(f"BrainWeb subject has inconsistent activity provenance: {subject_id}")
    activity_path = next(iter(upstream_paths))
    if not activity_path.is_file():
      raise FileNotFoundError(activity_path)
    if file_sha256(activity_path) != next(iter(upstream_shas)):
      raise ValueError(f"BrainWeb activity source SHA mismatch: {activity_path}")
    subject_root = brainweb_subject_root(activity_path)
    candidates = sorted(set([
      *subject_root.rglob("*.nii"), *subject_root.rglob("*.nii.gz"),
    ]))
    candidate_records = []
    activity_image = nib.load(str(activity_path))
    for candidate in candidates:
      image = nib.load(str(candidate))
      affine = np.asarray(image.affine, dtype=np.float64)
      activity_affine = np.asarray(activity_image.affine, dtype=np.float64)
      candidate_records.append({
        "path": candidate.as_posix(),
        "relative_to_subject": candidate.relative_to(subject_root).as_posix(),
        "sha256": file_sha256(candidate),
        "byte_size": candidate.stat().st_size,
        "shape": list(image.shape),
        "zooms": [float(value) for value in image.header.get_zooms()],
        "dtype": str(image.get_data_dtype()),
        "affine": affine.tolist(),
        "axis_codes": list(nib.aff2axcodes(affine)),
        "voxel_grid_matches_activity_source": bool(
          image.shape == activity_image.shape
          and np.allclose(affine, activity_affine, rtol=0.0, atol=1e-8)
        ),
        "is_activity_source": candidate.resolve() == activity_path.resolve(),
      })
    brainweb_subjects[subject_id] = {
      "split": subject_rows[0]["split"],
      "activity_path": activity_path.as_posix(),
      "subject_root": subject_root.as_posix(),
      "selected_slice_indices": sorted(int(row["source_slice_index"]) for row in subject_rows),
      "candidate_count": len(candidate_records),
      "candidates": candidate_records,
      "mr_selection_status": "pending_researcher_review_no_automatic_filename_guess",
    }

  container_paths = {Path(row["upstream_container_path"]) for row in hecktor_rows}
  container_shas = {row["upstream_container_sha256"] for row in hecktor_rows}
  if len(container_paths) != 1 or len(container_shas) != 1:
    raise ValueError("HECKTOR rows do not share one pinned NPZ container")
  container_path = next(iter(container_paths))
  if not container_path.is_file() or file_sha256(container_path) != next(iter(container_shas)):
    raise ValueError("HECKTOR NPZ container is missing or has the wrong SHA")
  with np.load(container_path, allow_pickle=False) as archive:
    if "input" not in archive.files:
      raise KeyError("HECKTOR container lacks input array")
    input_array = archive["input"]
    ct_slices = []
    subject_records = []
    for row in sorted(hecktor_rows, key=lambda item: int(item["subject_id"])):
      subject_index = int(row["source_subject_index"])
      slice_index = int(row["source_slice_index"])
      ct = np.asarray(input_array[subject_index, slice_index, :, :, 0])
      ct_slices.append(ct)
      subject_records.append({
        "subject_id": row["subject_id"], "split": row["split"],
        "source_subject_index": subject_index, "source_slice_index": slice_index,
        "selector": f"input[{subject_index},{slice_index},:,:,0]",
        "shape": list(ct.shape), "dtype": str(ct.dtype),
        "min": float(np.nanmin(ct)), "max": float(np.nanmax(ct)),
        "nonfinite_values": int(ct.size - np.isfinite(ct).sum()),
      })
    hecktor = {
      "container_path": container_path.as_posix(),
      "container_sha256": next(iter(container_shas)),
      "container_keys": list(archive.files),
      "input_shape": list(input_array.shape),
      "input_dtype": str(input_array.dtype),
      "ct_channel_index": 0,
      "pet_channel_index": 1,
      "ct_unit": "HU",
      "ct_selection_status": "declared_input_channel_0_pending_transform_manifest",
      "summary": summarize_ct_slices(ct_slices),
      "subjects": subject_records,
    }

  result = {
    "artifact_kind": "anatomy_source_inventory",
    "schema_version": 2,
    "status": "complete_pending_brainweb_mr_selection",
    "read_only": True,
    "source_manifest_path": source_manifest_path.relative_to(root).as_posix(),
    "source_manifest_sha256": file_sha256(source_manifest_path),
    "brainweb_subject_count": len(brainweb_subjects),
    "brainweb": brainweb_subjects,
    "hecktor_subject_count": len(hecktor_rows),
    "hecktor": hecktor,
    "proposed_method_anatomy_use": "prohibited",
    "allowed_use": "MR_KEM_CT_KEM_and_Bowsher_KEM_comparison_baselines_only",
  }
  atomic_write_json(output_path, result)
  return result
