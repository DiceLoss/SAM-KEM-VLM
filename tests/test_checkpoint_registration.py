import json
import shutil

import pytest
from sam_kem.artifacts import file_sha256
from sam_kem.checkpoint_registration import (
  register_decoder_archive,
  validate_registered_decoder_configuration,
)
from sam_kem.config import load_yaml_strict


def test_registration_preserves_training_config_and_binds_download(tmp_path):
  root = tmp_path / "repository"
  root.mkdir()
  training_config = root / "code/configs/generated/training.yaml"
  training_config.parent.mkdir(parents=True)
  training_config.write_text(
    "run_id: test_run\n"
    "medsam2:\n"
    "  checkpoint_paths:\n"
    "    brainweb: checkpoints/test/brainweb.pt\n"
    "    hecktor: checkpoints/test/hecktor.pt\n"
    "  checkpoint_sha256:\n"
    "    brainweb: null\n"
    "    hecktor: null\n"
    "  checkpoint_archive_uri:\n"
    "    brainweb: null\n"
    "    hecktor: null\n"
    "outputs:\n"
    "  artifact_policy:\n"
    "    immutable_archive_uri: null\n",
    encoding="utf-8",
  )
  original_config_bytes = training_config.read_bytes()
  training_config_sha = file_sha256(training_config)
  datasets = {}
  for dataset in ("brainweb", "hecktor"):
    checkpoint = root / f"checkpoints/test/{dataset}.pt"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    checkpoint.write_bytes(f"{dataset}-checkpoint".encode())
    checkpoint.with_suffix(".json").write_text(json.dumps({
      "checkpoint_sha256": file_sha256(checkpoint),
      "config_sha256": training_config_sha,
    }) + "\n", encoding="utf-8")
    datasets[dataset] = {"checkpoint_sha256": file_sha256(checkpoint)}

  local_archive = root / "delivery/bundle.zip"
  local_archive.parent.mkdir()
  local_archive.write_bytes(b"verified archive bytes")
  downloaded_archive = tmp_path / "downloaded-bundle.zip"
  shutil.copyfile(local_archive, downloaded_archive)
  build = root / "logs/build.json"
  verify = root / "logs/verify.json"
  build.parent.mkdir()
  archive_sha = file_sha256(local_archive)
  manifest_sha = "c" * 64
  build.write_text(json.dumps({
    "status": "local_bundle_complete_pending_immutable_upload",
    "archive_path": "delivery/bundle.zip",
    "archive_sha256": archive_sha,
    "archive_byte_size": local_archive.stat().st_size,
    "manifest_sha256": manifest_sha,
    "datasets": datasets,
  }) + "\n", encoding="utf-8")
  verify.write_text(json.dumps({
    "status": "complete",
    "archive_sha256": archive_sha,
    "archive_byte_size": local_archive.stat().st_size,
    "manifest_sha256": manifest_sha,
  }) + "\n", encoding="utf-8")

  output_config = root / "code/configs/generated/registered.yaml"
  registration = root / "logs/registration.json"
  result = register_decoder_archive(
    root, training_config, build, verify, downloaded_archive,
    archive_uri="https://archive.example/records/1/files/bundle.zip",
    archive_version="record-1-object-7",
    output_config_path=output_config,
    registration_record_path=registration,
  )

  assert training_config.read_bytes() == original_config_bytes
  assert result["status"] == "registered_for_development_validation"
  assert result["full_result_archive_uri"] is None
  registered = load_yaml_strict(output_config)
  assert registered["medsam2"]["training_config_sha256"] == training_config_sha
  assert registered["medsam2"]["checkpoint_sha256"] == {
    dataset: datasets[dataset]["checkpoint_sha256"]
    for dataset in ("brainweb", "hecktor")
  }
  assert registered["outputs"]["artifact_policy"]["immutable_archive_uri"] is None
  assert validate_registered_decoder_configuration(root, output_config)["status"] == "complete"


def test_registration_rejects_local_bundle_as_download(tmp_path):
  root = tmp_path / "repository"
  root.mkdir()
  config = root / "training.yaml"
  config.write_text("run_id: test\n", encoding="utf-8")
  build = root / "build.json"
  verify = root / "verify.json"
  archive = root / "bundle.zip"
  archive.write_bytes(b"archive")
  shared = {
    "archive_sha256": file_sha256(archive),
    "archive_byte_size": archive.stat().st_size,
    "manifest_sha256": "a" * 64,
  }
  build.write_text(json.dumps({
    **shared, "status": "local_bundle_complete_pending_immutable_upload",
    "archive_path": "bundle.zip", "datasets": {},
  }), encoding="utf-8")
  verify.write_text(json.dumps({**shared, "status": "complete"}), encoding="utf-8")
  with pytest.raises(ValueError, match="separately downloaded"):
    register_decoder_archive(
      root, config, build, verify, archive,
      archive_uri="https://archive.example/bundle.zip", archive_version="v1",
      output_config_path=root / "registered.yaml",
      registration_record_path=root / "registration.json",
    )
