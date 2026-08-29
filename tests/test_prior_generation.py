import json

import numpy as np
import pytest

from sam_kem.noise import array_sha256
from sam_kem.prior_generation import _verify_existing_prior


def test_resume_verifies_prior_array_and_metadata_identity(tmp_path):
  prior_path = tmp_path / "sam_prior.npy"
  metadata_path = tmp_path / "sam_prior.json"
  prior = np.linspace(-0.5, 1.5, 128 * 128, dtype=np.float32).reshape(128, 128)
  np.save(prior_path, prior, allow_pickle=False)
  expected = {
    "run_id": "run", "dataset": "brainweb",
    "decoder_checkpoint_sha256": "a" * 64,
  }
  metadata_path.write_text(json.dumps({
    **expected, "prior_sha256": array_sha256(prior),
  }), encoding="utf-8")

  verified, metadata = _verify_existing_prior(
    prior_path, metadata_path, expected=expected,
  )
  assert np.array_equal(verified, prior)
  assert metadata["decoder_checkpoint_sha256"] == "a" * 64


def test_resume_rejects_changed_prior_array(tmp_path):
  prior_path = tmp_path / "sam_prior.npy"
  metadata_path = tmp_path / "sam_prior.json"
  original = np.zeros((128, 128), dtype=np.float32)
  np.save(prior_path, original, allow_pickle=False)
  metadata_path.write_text(json.dumps({
    "run_id": "run", "prior_sha256": array_sha256(original),
  }), encoding="utf-8")
  changed = original.copy()
  changed[0, 0] = 1.0
  np.save(prior_path, changed, allow_pickle=False)

  with pytest.raises(ValueError, match="array checksum mismatch"):
    _verify_existing_prior(
      prior_path, metadata_path, expected={"run_id": "run"},
    )
