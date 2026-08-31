"""Freeze, ingest, and audit anatomy used only by named comparison baselines."""

from __future__ import annotations

import csv
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .artifacts import (
  atomic_csv_writer,
  atomic_save_numpy,
  atomic_write_json,
  file_sha256,
  finalize_csv,
  verify_csv_artifact,
)
from .ingestion import mean_pool_2x2, resize_full_square
from .noise import array_sha256


ANATOMY_FIELDS = [
  "dataset", "subject_id", "slice_id", "split", "modality", "unit",
  "source_path", "source_sha256", "source_kind", "source_key",
  "source_subject_index", "source_slice_axis", "source_slice_index",
  "source_channel_index", "source_selector", "spatial_transform",
  "source_spacing_y_mm", "source_spacing_x_mm", "selection_rule",
  "allowed_methods", "proposed_method_anatomy_use",
]
ALLOWED_SPLITS = {"train", "validation", "test_locked"}
SELECTION_RULE = "brainweb_root_t1_nifti_and_hecktor_declared_ct_channel_0_v1"
ALLOWED_METHODS = "MR-KEM|CT-KEM|Bowsher-KEM"


@dataclass(frozen=True)
class AnatomyCase:
  dataset: str
  subject_id: str
  slice_id: str
  split: str
  modality: str
  unit: str
  source_path: Path
  source_sha256: str
  source_kind: str
  source_key: str
  source_subject_index: int | None
  source_slice_axis: int | None
  source_slice_index: int
  source_channel_index: int | None
  source_selector: str
  spatial_transform: str
  source_spacing_y_mm: float
  source_spacing_x_mm: float
  selection_rule: str


def _optional_int(value: str) -> int | None:
  return None if value == "" else int(value)


def _resolve_source(path: str, root: Path) -> Path:
  source = Path(path)
  return source.resolve() if source.is_absolute() else (root / source).resolve()


def _read_csv(path: Path) -> list[dict[str, str]]:
  with path.open("r", encoding="utf-8", newline="") as handle:
    return list(csv.DictReader(handle))


def freeze_anatomy_source_manifest(
  root: Path,
  inventory_path: Path,
  source_activity_manifest: Path,
  output_path: Path,
  *,
  expected_inventory_sha256: str,
) -> Path:
  """Apply the researcher-declared T1/CT choice to every canonical source slice."""
  root = root.resolve()
  inventory_path = inventory_path.resolve()
  source_activity_manifest = source_activity_manifest.resolve()
  output_path = output_path.resolve()
  if file_sha256(inventory_path) != expected_inventory_sha256:
    raise ValueError("anatomy inventory SHA-256 differs from the reviewed artifact")
  inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
  if (
    inventory.get("artifact_kind") != "anatomy_source_inventory"
    or inventory.get("schema_version") != 2
    or inventory.get("status") != "complete_pending_brainweb_mr_selection"
    or inventory.get("read_only") is not True
    or inventory.get("proposed_method_anatomy_use") != "prohibited"
  ):
    raise ValueError("anatomy inventory is not the reviewed read-only gate")
  source_record = verify_csv_artifact(source_activity_manifest)
  if source_record.get("schema_name") != "source_activity_manifest_v2":
    raise ValueError("anatomy selection requires source_activity_manifest_v2")
  activity_rows = _read_csv(source_activity_manifest)

  brainweb_selection: dict[str, dict] = {}
  for subject_id, subject in inventory["brainweb"].items():
    matches = [
      candidate for candidate in subject["candidates"]
      if candidate["relative_to_subject"] == "t1.nii.gz"
    ]
    if len(matches) != 1:
      raise ValueError(f"BrainWeb {subject_id} must have exactly one root t1.nii.gz")
    selected = matches[0]
    if (
      selected.get("is_activity_source") is not False
      or selected.get("shape") != [256, 256, 258]
      or selected.get("zooms") != [1.0, 1.0, 1.0]
      or selected.get("dtype") != "float32"
      or selected.get("voxel_grid_matches_activity_source") is not True
      or len(selected.get("affine", [])) != 4
      or len(selected.get("axis_codes", [])) != 3
    ):
      raise ValueError(f"BrainWeb T1 geometry/type mismatch: {subject_id}")
    brainweb_selection[subject_id] = selected

  hecktor = inventory["hecktor"]
  if (
    hecktor.get("ct_channel_index") != 0
    or hecktor.get("pet_channel_index") != 1
    or hecktor.get("ct_unit") != "HU"
    or hecktor.get("input_shape") != [224, 144, 144, 144, 2]
    or hecktor.get("summary", {}).get("nonfinite_values") != 0
  ):
    raise ValueError("HECKTOR CT declaration differs from the reviewed inventory")
  hecktor_subjects = {
    row["subject_id"]: row for row in hecktor["subjects"]
  }

  records = []
  for row in activity_rows:
    common = {
      "dataset": row["dataset"], "subject_id": row["subject_id"],
      "slice_id": row["slice_id"], "split": row["split"],
      "selection_rule": SELECTION_RULE, "allowed_methods": ALLOWED_METHODS,
      "proposed_method_anatomy_use": "prohibited",
    }
    if row["dataset"] == "brainweb":
      selected = brainweb_selection[row["subject_id"]]
      index = int(row["source_slice_index"])
      record = {
        **common, "modality": "T1_MR", "unit": "arbitrary_MR_intensity",
        "source_path": selected["path"], "source_sha256": selected["sha256"],
        "source_kind": "nifti_volume", "source_key": "",
        "source_subject_index": "", "source_slice_axis": 2,
        "source_slice_index": index, "source_channel_index": "",
        "source_selector": f"dataobj[:,:,{index}]",
        "spatial_transform": "mean_pool_2x2_full_square",
        "source_spacing_y_mm": 1.0, "source_spacing_x_mm": 1.0,
      }
    elif row["dataset"] == "hecktor":
      subject = hecktor_subjects[row["subject_id"]]
      if (
        int(row["source_subject_index"]) != int(subject["source_subject_index"])
        or int(row["source_slice_index"]) != int(subject["source_slice_index"])
      ):
        raise ValueError(f"HECKTOR PET/CT subject-slice mismatch: {row['subject_id']}")
      record = {
        **common, "modality": "CT", "unit": "HU",
        "source_path": hecktor["container_path"],
        "source_sha256": hecktor["container_sha256"],
        "source_kind": "npz_hecktor_input", "source_key": "input",
        "source_subject_index": subject["source_subject_index"],
        "source_slice_axis": 1, "source_slice_index": subject["source_slice_index"],
        "source_channel_index": 0, "source_selector": subject["selector"],
        "spatial_transform": "resize_full_square_to_128",
        "source_spacing_y_mm": "", "source_spacing_x_mm": "",
      }
    else:
      raise ValueError(f"unexpected dataset: {row['dataset']}")
    records.append(record)

  keys = {(row["dataset"], row["subject_id"], row["slice_id"]) for row in records}
  counts = Counter(row["dataset"] for row in records)
  if len(keys) != len(records) or counts != {"brainweb": 3511, "hecktor": 224}:
    raise ValueError(f"unexpected anatomy selection cardinality: {dict(counts)}")
  with atomic_csv_writer(output_path, ANATOMY_FIELDS) as writer:
    writer.writerows(records)
  sidecar = finalize_csv(
    output_path, schema_name="anatomy_source_manifest_v1",
    primary_key=["dataset", "subject_id", "slice_id"],
  )
  sidecar_record = json.loads(sidecar.read_text(encoding="utf-8"))
  sidecar_record.update({
    "inventory_path": inventory_path.relative_to(root).as_posix(),
    "inventory_sha256": expected_inventory_sha256,
    "source_activity_manifest_path": source_activity_manifest.relative_to(root).as_posix(),
    "source_activity_manifest_sha256": file_sha256(source_activity_manifest),
    "selection_rule": SELECTION_RULE,
    "dataset_counts": dict(counts),
    "proposed_method_anatomy_use": "prohibited",
    "allowed_methods": ALLOWED_METHODS.split("|"),
  })
  atomic_write_json(sidecar, sidecar_record, replace=True)
  return output_path


def load_anatomy_source_manifest(root: Path, path: Path) -> tuple[list[AnatomyCase], str]:
  record = verify_csv_artifact(path)
  if (
    record.get("schema_name") != "anatomy_source_manifest_v1"
    or record.get("selection_rule") != SELECTION_RULE
    or record.get("proposed_method_anatomy_use") != "prohibited"
  ):
    raise ValueError("unreviewed anatomy source manifest")
  rows = _read_csv(path)
  cases = []
  for row in rows:
    if row["split"] not in ALLOWED_SPLITS:
      raise ValueError(f"invalid anatomy split: {row['split']}")
    if row["allowed_methods"] != ALLOWED_METHODS or row["proposed_method_anatomy_use"] != "prohibited":
      raise ValueError("anatomy use policy mismatch")
    cases.append(AnatomyCase(
      dataset=row["dataset"], subject_id=row["subject_id"], slice_id=row["slice_id"],
      split=row["split"], modality=row["modality"], unit=row["unit"],
      source_path=_resolve_source(row["source_path"], root),
      source_sha256=row["source_sha256"], source_kind=row["source_kind"],
      source_key=row["source_key"],
      source_subject_index=_optional_int(row["source_subject_index"]),
      source_slice_axis=_optional_int(row["source_slice_axis"]),
      source_slice_index=int(row["source_slice_index"]),
      source_channel_index=_optional_int(row["source_channel_index"]),
      source_selector=row["source_selector"], spatial_transform=row["spatial_transform"],
      source_spacing_y_mm=float(row["source_spacing_y_mm"] or "nan"),
      source_spacing_x_mm=float(row["source_spacing_x_mm"] or "nan"),
      selection_rule=row["selection_rule"],
    ))
  return cases, file_sha256(path)


def _load_source_group(cases: list[AnatomyCase]) -> object:
  first = cases[0]
  if not first.source_path.is_file() or file_sha256(first.source_path) != first.source_sha256:
    raise ValueError(f"anatomy source missing or SHA mismatch: {first.source_path}")
  if first.source_kind == "nifti_volume":
    try:
      import nibabel as nib
    except ImportError as error:
      raise RuntimeError("install the project 'data' extra to read BrainWeb T1") from error
    return nib.load(str(first.source_path))
  if first.source_kind == "npz_hecktor_input":
    # NpzFile.__getitem__ decompresses an entry on every access. Materialize the
    # shared 5-D container once so 224 subjects do not trigger 224 decompressions.
    with np.load(first.source_path, allow_pickle=False) as archive:
      if first.source_key not in archive.files:
        raise KeyError(f"HECKTOR container lacks {first.source_key!r}")
      return archive[first.source_key]
  raise ValueError(f"unsupported anatomy source kind: {first.source_kind}")


def _extract_source(case: AnatomyCase, loaded: object) -> np.ndarray:
  if case.source_kind == "nifti_volume":
    value = np.asanyarray(loaded.dataobj[:, :, case.source_slice_index])
  else:
    value = loaded[
      case.source_subject_index, case.source_slice_index, :, :, case.source_channel_index,
    ]
  value = np.asarray(value, dtype=np.float32)
  if value.ndim != 2 or not np.all(np.isfinite(value)) or float(np.ptp(value)) <= 0:
    raise ValueError(f"anatomy source must be finite, 2-D and nonconstant: {case}")
  return value


def _transform_anatomy(value: np.ndarray, case: AnatomyCase) -> np.ndarray:
  if case.spatial_transform == "mean_pool_2x2_full_square":
    transformed = mean_pool_2x2(value)
  elif case.spatial_transform == "resize_full_square_to_128":
    transformed = resize_full_square(value, output_size=128)
  else:
    raise ValueError(f"unapproved anatomy transform: {case.spatial_transform}")
  transformed = np.asarray(transformed, dtype=np.float32)
  if transformed.shape != (128, 128) or not np.all(np.isfinite(transformed)) or float(np.ptp(transformed)) <= 0:
    raise ValueError(f"anatomy transform produced an invalid feature: {case}")
  return transformed


def _expected_metadata(case: AnatomyCase, manifest_sha256: str, source: np.ndarray, anatomy: np.ndarray) -> dict:
  return {
    "schema_version": 1, "artifact_kind": "canonical_comparison_anatomy",
    "dataset": case.dataset, "subject_id": case.subject_id, "slice_id": case.slice_id,
    "split": case.split, "modality": case.modality, "unit": case.unit,
    "source_path": str(case.source_path), "source_sha256": case.source_sha256,
    "source_kind": case.source_kind, "source_key": case.source_key,
    "source_subject_index": case.source_subject_index,
    "source_slice_axis": case.source_slice_axis,
    "source_slice_index": case.source_slice_index,
    "source_channel_index": case.source_channel_index,
    "source_selector": case.source_selector,
    "source_array_sha256": array_sha256(source), "source_shape": list(source.shape),
    "spatial_transform": case.spatial_transform,
    "support_policy": "complete_square_no_content_crop_recenter_window_or_mask",
    "normalization_at_storage": "none_raw_modality_values_float32",
    "kernel_feature_normalization": "per_image_minmax_at_reconstruction_only",
    "selection_rule": case.selection_rule,
    "anatomy_source_manifest_sha256": manifest_sha256,
    "anatomy_shape": list(anatomy.shape), "anatomy_dtype": anatomy.dtype.str,
    "anatomy_min": float(anatomy.min()), "anatomy_max": float(anatomy.max()),
    "anatomy_sha256": array_sha256(anatomy),
    "allowed_methods": ALLOWED_METHODS.split("|"),
    "proposed_method_anatomy_use": "prohibited",
  }


def _verify_existing(output: Path, sidecar: Path, expected: dict) -> bool:
  if not output.is_file() or not sidecar.is_file():
    return False
  anatomy = np.load(output, allow_pickle=False)
  record = json.loads(sidecar.read_text(encoding="utf-8"))
  return (
    anatomy.shape == (128, 128) and np.all(np.isfinite(anatomy))
    and float(np.ptp(anatomy)) > 0
    and array_sha256(anatomy) == expected["anatomy_sha256"]
    and record == expected
  )


def ingest_anatomy_manifest(
  root: Path, manifest_path: Path, arrays_root: Path, *, splits: set[str],
  resume_verified: bool = False,
) -> dict:
  """Write raw-modality anatomy arrays with exact source and transform provenance."""
  if not splits or not splits <= ALLOWED_SPLITS or "test_locked" in splits:
    raise ValueError("this development anatomy gate accepts train/validation only")
  root = root.resolve()
  arrays_root = arrays_root.resolve()
  cases, manifest_sha = load_anatomy_source_manifest(root, manifest_path.resolve())
  selected = [case for case in cases if case.split in splits]
  groups: dict[tuple[Path, str], list[AnatomyCase]] = {}
  for case in selected:
    groups.setdefault((case.source_path, case.source_sha256), []).append(case)
  written = reused = 0
  for group_cases in groups.values():
    loaded = _load_source_group(group_cases)
    try:
      for case in group_cases:
        source = _extract_source(case, loaded)
        anatomy = _transform_anatomy(source, case)
        output = arrays_root / case.dataset / case.subject_id / case.slice_id / "anatomy.npy"
        sidecar = output.with_suffix(".json")
        expected = _expected_metadata(case, manifest_sha, source, anatomy)
        if output.exists() or sidecar.exists():
          if resume_verified and _verify_existing(output, sidecar, expected):
            reused += 1
            continue
          raise FileExistsError(f"refusing to overwrite/unverified anatomy: {output}")
        atomic_save_numpy(output, anatomy)
        try:
          atomic_write_json(sidecar, expected)
        except BaseException:
          output.unlink(missing_ok=True)
          raise
        written += 1
    finally:
      close = getattr(loaded, "close", None)
      if close is not None:
        close()
  return {
    "artifact_kind": "canonical_comparison_anatomy_batch",
    "status": "complete", "selected": len(selected), "written": written,
    "reused_verified": reused, "splits": sorted(splits),
    "dataset_counts": dict(Counter(case.dataset for case in selected)),
    "anatomy_source_manifest_sha256": manifest_sha,
    "selection_rule": SELECTION_RULE,
    "proposed_method_anatomy_use": "prohibited",
  }


def audit_ingested_anatomy(
  root: Path, manifest_path: Path, arrays_root: Path, output_path: Path,
  *, splits: set[str],
) -> dict:
  root = root.resolve()
  arrays_root = arrays_root.resolve()
  cases, manifest_sha = load_anatomy_source_manifest(root, manifest_path.resolve())
  selected = [case for case in cases if case.split in splits]
  modality_counts = Counter()
  negative_counts = Counter()
  for case in selected:
    output = arrays_root / case.dataset / case.subject_id / case.slice_id / "anatomy.npy"
    sidecar = output.with_suffix(".json")
    if not output.is_file() or not sidecar.is_file():
      raise FileNotFoundError(f"missing canonical anatomy pair: {output}")
    anatomy = np.load(output, allow_pickle=False)
    record = json.loads(sidecar.read_text(encoding="utf-8"))
    if (
      anatomy.shape != (128, 128) or not np.all(np.isfinite(anatomy))
      or float(np.ptp(anatomy)) <= 0
      or record.get("anatomy_sha256") != array_sha256(anatomy)
      or record.get("anatomy_source_manifest_sha256") != manifest_sha
      or record.get("selection_rule") != SELECTION_RULE
      or record.get("proposed_method_anatomy_use") != "prohibited"
      or record.get("dataset") != case.dataset
      or record.get("subject_id") != case.subject_id
      or record.get("slice_id") != case.slice_id
      or record.get("split") != case.split
    ):
      raise ValueError(f"canonical anatomy provenance mismatch: {output}")
    modality_counts[case.modality] += 1
    if float(anatomy.min()) < 0:
      negative_counts[case.modality] += 1
  result = {
    "artifact_kind": "canonical_comparison_anatomy_audit", "schema_version": 1,
    "status": "PASS", "records": len(selected), "splits": sorted(splits),
    "dataset_counts": dict(Counter(case.dataset for case in selected)),
    "modality_counts": dict(modality_counts),
    "records_with_negative_pixels": dict(negative_counts),
    "shape": [128, 128], "finite_and_nonconstant": True,
    "anatomy_source_manifest_path": manifest_path.resolve().relative_to(root).as_posix(),
    "anatomy_source_manifest_sha256": manifest_sha,
    "selection_rule": SELECTION_RULE,
    "storage_policy": "raw_modality_values_float32_no_windowing_or_per_image_normalization",
    "kernel_feature_normalization": "per_image_minmax_at_reconstruction_only",
    "allowed_methods": ALLOWED_METHODS.split("|"),
    "proposed_method_anatomy_use": "prohibited",
  }
  atomic_write_json(output_path.resolve(), result)
  return result
