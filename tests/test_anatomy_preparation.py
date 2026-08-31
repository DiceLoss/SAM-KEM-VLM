import csv
import json
from pathlib import Path

import numpy as np

from sam_kem.anatomy_preparation import (
  ALLOWED_METHODS,
  ANATOMY_FIELDS,
  SELECTION_RULE,
  audit_ingested_anatomy,
  freeze_anatomy_source_manifest,
  ingest_anatomy_manifest,
)
from sam_kem.artifacts import file_sha256, finalize_csv
from sam_kem.ingestion import SOURCE_COLUMNS
from sam_kem.manifest import build_case_manifest


def _write_csv(path: Path, fields, rows, schema, primary_key):
  path.parent.mkdir(parents=True, exist_ok=True)
  with path.open("w", encoding="utf-8", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
  finalize_csv(path, schema_name=schema, primary_key=primary_key)


def test_ingest_and_audit_ct_preserve_raw_negative_hu(tmp_path):
  source = tmp_path / "hecktor.npz"
  value = np.zeros((1, 1, 2, 2, 2), dtype=np.float64)
  value[0, 0, :, :, 0] = [[-1000.0, -100.0], [100.0, 900.0]]
  np.savez(source, input=value)
  manifest = tmp_path / "anatomy.csv"
  rows = [{
    "dataset": "hecktor", "subject_id": "1", "slice_id": "z000",
    "split": "validation", "modality": "CT", "unit": "HU",
    "source_path": str(source), "source_sha256": file_sha256(source),
    "source_kind": "npz_hecktor_input", "source_key": "input",
    "source_subject_index": 0, "source_slice_axis": 1, "source_slice_index": 0,
    "source_channel_index": 0, "source_selector": "input[0,0,:,:,0]",
    "spatial_transform": "resize_full_square_to_128",
    "source_spacing_y_mm": "", "source_spacing_x_mm": "",
    "selection_rule": SELECTION_RULE, "allowed_methods": ALLOWED_METHODS,
    "proposed_method_anatomy_use": "prohibited",
  }]
  _write_csv(
    manifest, ANATOMY_FIELDS, rows, "anatomy_source_manifest_v1",
    ["dataset", "subject_id", "slice_id"],
  )
  sidecar = manifest.with_suffix(".csv.manifest.json")
  record = json.loads(sidecar.read_text(encoding="utf-8"))
  record.update({
    "selection_rule": SELECTION_RULE,
    "proposed_method_anatomy_use": "prohibited",
  })
  sidecar.write_text(json.dumps(record), encoding="utf-8")

  arrays = tmp_path / "arrays"
  result = ingest_anatomy_manifest(
    tmp_path, manifest, arrays, splits={"validation"},
  )
  assert result["written"] == 1 and result["reused_verified"] == 0
  anatomy = np.load(arrays / "hecktor/1/z000/anatomy.npy", allow_pickle=False)
  assert anatomy.shape == (128, 128)
  assert anatomy.min() < 0 < anatomy.max()
  resumed = ingest_anatomy_manifest(
    tmp_path, manifest, arrays, splits={"validation"}, resume_verified=True,
  )
  assert resumed["written"] == 0 and resumed["reused_verified"] == 1
  audit = audit_ingested_anatomy(
    tmp_path, manifest, arrays, tmp_path / "audit.json", splits={"validation"},
  )
  assert audit["status"] == "PASS"
  assert audit["records_with_negative_pixels"] == {"CT": 1}


def test_freeze_selects_only_root_t1_and_declared_ct_channel(tmp_path):
  subject_counts = [
    175, 173, 176, 181, 173, 175, 176, 172, 175, 176,
    174, 176, 174, 178, 178, 173, 173, 175, 181, 177,
  ]
  assert sum(subject_counts) == 3511
  activity_rows = []
  brainweb = {}
  for number, count in enumerate(subject_counts, start=4):
    subject = f"subject{number:02d}"
    t1 = tmp_path / subject / "t1.nii.gz"
    brainweb[subject] = {
      "candidates": [{
        "path": str(t1), "relative_to_subject": "t1.nii.gz",
        "sha256": f"{number:064x}", "shape": [256, 256, 258],
        "zooms": [1.0, 1.0, 1.0], "dtype": "float32",
        "is_activity_source": False, "affine": np.eye(4).tolist(),
        "axis_codes": ["R", "A", "S"],
        "voxel_grid_matches_activity_source": True,
      }],
    }
    for index in range(count):
      activity_rows.append({
        **{field: "" for field in SOURCE_COLUMNS},
        "dataset": "brainweb", "subject_id": subject, "slice_id": f"z{index:03d}",
        "split": "train", "source_slice_index": str(index),
      })
  hecktor_subjects = []
  for number in range(1, 225):
    activity_rows.append({
      **{field: "" for field in SOURCE_COLUMNS},
      "dataset": "hecktor", "subject_id": str(number), "slice_id": "z072",
      "split": "validation", "source_subject_index": str(number - 1),
      "source_slice_index": "72",
    })
    hecktor_subjects.append({
      "subject_id": str(number), "source_subject_index": number - 1,
      "source_slice_index": 72, "selector": f"input[{number - 1},72,:,:,0]",
    })
  source_manifest = tmp_path / "source.csv"
  _write_csv(
    source_manifest, SOURCE_COLUMNS, activity_rows, "source_activity_manifest_v2",
    ["dataset", "subject_id", "slice_id"],
  )
  inventory = tmp_path / "inventory.json"
  inventory.write_text(json.dumps({
    "artifact_kind": "anatomy_source_inventory",
    "schema_version": 2,
    "status": "complete_pending_brainweb_mr_selection", "read_only": True,
    "proposed_method_anatomy_use": "prohibited", "brainweb": brainweb,
    "hecktor": {
      "ct_channel_index": 0, "pet_channel_index": 1, "ct_unit": "HU",
      "input_shape": [224, 144, 144, 144, 2],
      "summary": {"nonfinite_values": 0},
      "container_path": str(tmp_path / "HECKTOR.npz"),
      "container_sha256": "a" * 64, "subjects": hecktor_subjects,
    },
  }), encoding="utf-8")
  output = tmp_path / "anatomy_sources.csv"
  freeze_anatomy_source_manifest(
    tmp_path, inventory, source_manifest, output,
    expected_inventory_sha256=file_sha256(inventory),
  )
  rows = list(csv.DictReader(output.open(encoding="utf-8", newline="")))
  assert len(rows) == 3735
  assert sum(row["dataset"] == "brainweb" for row in rows) == 3511
  assert sum(row["dataset"] == "hecktor" for row in rows) == 224
  assert {row["source_path"] for row in rows if row["dataset"] == "brainweb"} == {
    str(tmp_path / f"subject{number:02d}" / "t1.nii.gz") for number in range(4, 24)
  }
  assert {row["source_channel_index"] for row in rows if row["dataset"] == "hecktor"} == {"0"}
  assert {row["proposed_method_anatomy_use"] for row in rows} == {"prohibited"}


def test_case_manifest_records_single_requested_split_in_sidecar(tmp_path, monkeypatch):
  import sam_kem.manifest as manifest_module

  arrays = tmp_path / "arrays"
  (arrays / "brainweb/subject04").mkdir(parents=True)
  monkeypatch.setattr(
    manifest_module, "load_contract", lambda _path: ({}, object()),
  )
  monkeypatch.setattr(
    manifest_module, "load_configured_subject_splits",
    lambda _root, _cfg: [{
      "dataset": "brainweb", "subject_id": "subject04", "split": "validation",
    }],
  )
  output = tmp_path / "validation.csv"
  build_case_manifest(
    tmp_path, tmp_path / "config.yaml", arrays, output, splits={"validation"},
  )
  record = json.loads(output.with_suffix(".csv.manifest.json").read_text(encoding="utf-8"))
  assert record["split"] == "validation"
