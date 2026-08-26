import csv
import json
from pathlib import Path

import numpy as np
import pytest

from sam_kem.artifacts import file_sha256
from sam_kem.ingestion import (
  SOURCE_COLUMNS, SourceCase, ingest_manifest, physical_resample_to_circular_fov,
  transform_activity,
)
from sam_kem.noise import array_sha256
from sam_kem.traceability import circular_mask


def _write_manifest(path: Path, row: dict) -> None:
  with path.open("w", encoding="utf-8", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=SOURCE_COLUMNS)
    writer.writeheader(); writer.writerow(row)


def test_manifest_ingestion_mean_pools_without_normalizing(tmp_path):
  source = np.zeros((256, 256), dtype=np.float32)
  source[80:176, 80:176] = 4.0
  source_path = tmp_path / "source.npy"
  np.save(source_path, source, allow_pickle=False)
  manifest = tmp_path / "sources.csv"
  split_path = tmp_path / "splits.csv"
  split_path.write_text("dataset,subject_id,split\nbrainweb,subject01,train\n", encoding="utf-8")
  _write_manifest(manifest, {
    "dataset": "brainweb", "subject_id": "subject01", "slice_id": "z120",
    "split": "train", "source_path": "source.npy",
    "source_sha256": file_sha256(source_path), "source_kind": "npy_slice",
    "source_key": "", "source_subject_index": "", "source_slice_axis": "",
    "source_slice_index": "", "source_channel_index": "",
    "activity_unit": "simulation_activity", "spatial_transform": "mean_pool_2x2_circular_fov",
    "source_spacing_y_mm": "", "source_spacing_x_mm": "",
    "scanner_fov_diameter_mm": "",
    "scanner_axis_row_index": "", "scanner_axis_column_index": "",
  })
  outputs = ingest_manifest(
    tmp_path, manifest, split_path, tmp_path / "arrays", splits={"train"},
    outside_mass_tolerance=1e-12,
  )
  assert len(outputs) == 1
  actual = np.load(outputs[0], allow_pickle=False)
  expected = source.reshape(128, 2, 128, 2).mean(axis=(1, 3), dtype=np.float32)
  np.testing.assert_array_equal(actual, expected)
  assert actual.max() == 4.0
  sidecar = json.loads(outputs[0].with_suffix(".json").read_text(encoding="utf-8"))
  assert sidecar["activity_sha256"] == array_sha256(actual)
  assert sidecar["spatial_transform"] == "mean_pool_2x2_circular_fov"


def test_physical_resampling_is_identity_on_matching_grids():
  activity = np.zeros((128, 128), dtype=np.float64)
  activity[circular_mask(128)] = 2.0
  transformed, outside = physical_resample_to_circular_fov(
    activity, source_spacing_y_mm=2.0, source_spacing_x_mm=2.0,
    scanner_fov_diameter_mm=256.0,
  )
  assert outside == 0.0
  np.testing.assert_allclose(transformed, activity, rtol=0.0, atol=1e-12)


def test_physical_transform_refuses_missing_geometry():
  case = SourceCase(
    "hecktor", "1", "z72", "train", Path("unused"), "0" * 64,
    "npy_slice", "", None, None, None, None, "SUV",
    "physical_resample_circular_fov", None, None, None,
    None, None,
  )
  with pytest.raises(ValueError, match="requires spacing, scanner FOV"):
    transform_activity(np.ones((144, 144)), case, outside_mass_tolerance=1e-6)


def test_ingestion_refuses_activity_outside_declared_fov():
  activity = np.zeros((128, 128), dtype=np.float64)
  activity[:8, :8] = 1.0
  case = SourceCase(
    "hecktor", "1", "z72", "train", Path("unused"), "0" * 64,
    "npy_slice", "", None, None, None, None, "SUV",
    "identity_128_circular_fov", None, None, None,
    None, None,
  )
  with pytest.raises(ValueError, match="outside the declared scanner FOV"):
    transform_activity(activity, case, outside_mass_tolerance=1e-6)
