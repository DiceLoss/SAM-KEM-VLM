import json
from pathlib import Path

import numpy as np
import yaml

from sam_kem.noise import array_sha256
from sam_kem.pet_simulation import prepare_noise_free_projection


ROOT = Path(__file__).resolve().parents[1]


def test_projection_requires_and_binds_canonical_ingestion(tmp_path):
  cfg = yaml.safe_load((ROOT / "code/configs/experiment.yaml").read_text(encoding="utf-8"))
  cfg["normalization"]["scale_constants"] = {"brainweb": 6.0, "hecktor": 35.0}
  cfg["data"]["split_sha256"] = "a" * 64
  cfg["normalization"]["scale_constant_sources"] = {
    "brainweb": "unit_test", "hecktor": "unit_test",
  }
  config_path = tmp_path / "experiment.yaml"
  config_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
  raw = np.full((128, 128), 3.0, dtype=np.float32)
  raw_path = tmp_path / "raw_activity.npy"
  np.save(raw_path, raw, allow_pickle=False)
  raw_path.with_suffix(".json").write_text(json.dumps({
    "schema_version": 1, "artifact_kind": "canonical_activity",
    "dataset": "brainweb", "activity_sha256": array_sha256(raw),
  }), encoding="utf-8")
  target_path = tmp_path / "target.npy"
  projection_path = tmp_path / "projection.npy"
  prepare_noise_free_projection(
    ROOT, config_path, dataset="brainweb", raw_activity_path=raw_path,
    normalized_target_path=target_path, projection_path=projection_path,
  )
  target = np.load(target_path, allow_pickle=False)
  projection = np.load(projection_path, allow_pickle=False)
  np.testing.assert_array_equal(target, raw / 6.0)
  assert projection.shape == (180, 182)
  sidecar = json.loads(projection_path.with_suffix(".json").read_text(encoding="utf-8"))
  assert sidecar["schema_version"] == 2
  assert sidecar["normalization_scale_C"] == 6.0
  assert sidecar["target_sha256"] == array_sha256(target)
  assert sidecar["noise_free_projection_sha256"] == array_sha256(projection)
  assert sidecar["support_policy"] == "complete_square_no_content_crop_or_circular_mask"


def test_fixed_scale_records_but_does_not_clip_reference_range_excursions(tmp_path):
  cfg = yaml.safe_load((ROOT / "code/configs/experiment.yaml").read_text(encoding="utf-8"))
  cfg["normalization"]["scale_constants"] = {"brainweb": 1.0, "hecktor": 1.0}
  cfg["data"]["split_sha256"] = "a" * 64
  cfg["normalization"]["scale_constant_sources"] = {
    "brainweb": "unit_test", "hecktor": "unit_test",
  }
  config_path = tmp_path / "experiment.yaml"
  config_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
  raw = np.full((128, 128), 2.0, dtype=np.float32)
  raw_path = tmp_path / "raw_activity.npy"
  np.save(raw_path, raw, allow_pickle=False)
  raw_path.with_suffix(".json").write_text(json.dumps({
    "schema_version": 1, "artifact_kind": "canonical_activity",
    "dataset": "brainweb", "activity_sha256": array_sha256(raw),
  }), encoding="utf-8")
  target_path = tmp_path / "target.npy"
  projection_path = tmp_path / "projection.npy"
  prepare_noise_free_projection(
    ROOT, config_path, dataset="brainweb", raw_activity_path=raw_path,
    normalized_target_path=target_path, projection_path=projection_path,
  )
  np.testing.assert_array_equal(np.load(target_path), raw)
  record = json.loads(projection_path.with_suffix(".json").read_text(encoding="utf-8"))
  assert record["normalized_target_above_reference_fraction"] == 1.0
