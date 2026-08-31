import hashlib
import json
import zipfile

import pytest

from sam_kem.artifacts import file_sha256
from sam_kem.prior_delivery import verify_validation_prior_bundle


def _write_test_bundle(path, *, corrupt=False):
  metadata = {
    "artifact_kind": "validation_prior_bundle_metadata",
    "split": "validation", "prior_records": 8892,
  }
  metadata_bytes = (json.dumps(metadata) + "\n").encode()
  data = b"prior" if not corrupt else b"changed"
  records = {
    "arrays/prior.npy": hashlib.sha256(b"prior").hexdigest(),
    "environment/VALIDATION_PRIOR_BUNDLE_METADATA.json": hashlib.sha256(
      metadata_bytes
    ).hexdigest(),
  }
  manifest = "".join(
    f"{checksum}  {name}\n" for name, checksum in records.items()
  ).encode()
  with zipfile.ZipFile(path, "w") as archive:
    archive.writestr("arrays/prior.npy", data)
    archive.writestr("environment/VALIDATION_PRIOR_BUNDLE_METADATA.json", metadata_bytes)
    archive.writestr("MANIFEST.sha256", manifest)


def test_validation_prior_bundle_stream_verification(tmp_path):
  archive = tmp_path / "priors.zip"
  _write_test_bundle(archive)
  result = verify_validation_prior_bundle(archive)
  assert result["status"] == "complete"
  assert result["archive_sha256"] == file_sha256(archive)
  assert result["verified_member_count"] == 2


def test_validation_prior_bundle_rejects_member_checksum_change(tmp_path):
  archive = tmp_path / "priors.zip"
  _write_test_bundle(archive, corrupt=True)
  with pytest.raises(ValueError, match="member checksum mismatch"):
    verify_validation_prior_bundle(archive)
