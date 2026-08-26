import csv
import json
from pathlib import Path

import numpy as np

from sam_kem.artifacts import file_sha256
from sam_kem.ingestion import (
  SOURCE_COLUMNS, SourceCase, ingest_manifest, resize_full_square, transform_activity,
)
from sam_kem.noise import array_sha256


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
    "activity_unit": "simulation_activity", "spatial_transform": "mean_pool_2x2_full_square",
    "source_spacing_y_mm": "", "source_spacing_x_mm": "",
  })
  outputs = ingest_manifest(
    tmp_path, manifest, split_path, tmp_path / "arrays", splits={"train"},
  )
  assert len(outputs) == 1
  actual = np.load(outputs[0], allow_pickle=False)
  expected = source.reshape(128, 2, 128, 2).mean(axis=(1, 3), dtype=np.float32)
  np.testing.assert_array_equal(actual, expected)
  assert actual.max() == 4.0
  sidecar = json.loads(outputs[0].with_suffix(".json").read_text(encoding="utf-8"))
  assert sidecar["activity_sha256"] == array_sha256(actual)
  assert sidecar["spatial_transform"] == "mean_pool_2x2_full_square"
  assert sidecar["support_policy"] == "complete_square_no_content_crop_or_circular_mask"


def test_full_square_resize_is_identity_on_matching_grids():
  activity = np.arange(128 * 128, dtype=np.float64).reshape(128, 128)
  transformed = resize_full_square(activity)
  np.testing.assert_allclose(transformed, activity, rtol=0.0, atol=1e-12)


def test_hecktor_resize_preserves_complete_constant_square_without_geometry():
  case = SourceCase(
    "hecktor", "1", "z72", "train", Path("unused"), "0" * 64,
    "npy_slice", "", None, None, None, None, "SUV",
    "resize_full_square_to_128", None, None,
  )
  transformed = transform_activity(np.ones((144, 144)), case)
  assert transformed.shape == (128, 128)
  np.testing.assert_array_equal(transformed, np.ones((128, 128), dtype=np.float32))


def test_ingestion_preserves_corner_activity():
  activity = np.zeros((128, 128), dtype=np.float64)
  activity[:8, :8] = 1.0
  case = SourceCase(
    "hecktor", "1", "z72", "train", Path("unused"), "0" * 64,
    "npy_slice", "", None, None, None, None, "SUV",
    "identity_128_full_square", None, None,
  )
  transformed = transform_activity(activity, case)
  np.testing.assert_array_equal(transformed, activity.astype(np.float32))
