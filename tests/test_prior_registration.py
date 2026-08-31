import json
import shutil

from sam_kem.artifacts import file_sha256
from sam_kem.config import load_yaml_strict
from sam_kem.prior_registration import register_validation_prior_archive

from test_prior_delivery import _write_test_bundle


def test_prior_registration_preserves_source_config_and_binds_download(
  tmp_path, monkeypatch,
):
  root = tmp_path / "repository"
  root.mkdir()
  source_config = root / "code/configs/generated/decoder_registered.yaml"
  source_config.parent.mkdir(parents=True)
  source_config.write_text(
    "run_id: run\n"
    "outputs:\n"
    "  artifact_policy:\n"
    "    immutable_archive_uri: null\n",
    encoding="utf-8",
  )
  original = source_config.read_bytes()
  monkeypatch.setattr(
    "sam_kem.prior_registration.validate_registered_decoder_configuration",
    lambda *_args, **_kwargs: {"status": "complete"},
  )
  local_archive = root / "delivery/priors.zip"
  local_archive.parent.mkdir()
  _write_test_bundle(local_archive)
  downloaded = tmp_path / "downloaded.zip"
  shutil.copyfile(local_archive, downloaded)
  verification = __import__(
    "sam_kem.prior_delivery", fromlist=["verify_validation_prior_bundle"]
  ).verify_validation_prior_bundle(local_archive)
  build = root / "logs/build.json"
  verify = root / "logs/verify.json"
  build.parent.mkdir()
  shared = {
    "archive_sha256": file_sha256(local_archive),
    "archive_byte_size": local_archive.stat().st_size,
    "manifest_sha256": verification["manifest_sha256"],
    "prior_records": 8892,
  }
  build.write_text(json.dumps({
    **shared,
    "status": "local_bundle_complete_pending_lfs_publication",
    "archive_path": "delivery/priors.zip",
    "prior_manifest_sha256": "b" * 64,
  }), encoding="utf-8")
  verify.write_text(json.dumps({**shared, "status": "complete"}), encoding="utf-8")
  output_config = root / "code/configs/generated/validation_registered.yaml"
  registration = root / "logs/registration.json"

  result = register_validation_prior_archive(
    root, source_config, build, verify, downloaded,
    archive_uri="https://archive.example/commit/1/priors.zip",
    archive_version="commit-1+lfs-oid-sha256",
    output_config_path=output_config,
    registration_record_path=registration,
  )

  assert source_config.read_bytes() == original
  assert result["status"] == "registered_for_development_validation_sweep"
  registered = load_yaml_strict(output_config)
  assert registered["validation_prior_archive"]["sha256"] == shared["archive_sha256"]
  assert registered["outputs"]["artifact_policy"]["immutable_archive_uri"] is None
