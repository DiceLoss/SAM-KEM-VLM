import json
from pathlib import Path

import numpy as np
import yaml

from sam_kem.case_preparation import prepare_cases
from sam_kem.noise import array_sha256


ROOT = Path(__file__).resolve().parents[1]


def test_batch_preparation_creates_one_matched_case_and_verified_resume(tmp_path):
  cfg = yaml.safe_load((ROOT / "code/configs/experiment.yaml").read_text(encoding="utf-8"))
  cfg["acquisition"]["count_levels"] = [200000]
  cfg["acquisition"]["realizations"] = [0]
  cfg["medsam2"]["training_count_levels"] = [200000]
  cfg["normalization"]["scale_constants"] = {"brainweb": 6.0, "hecktor": 35.0}
  cfg["data"]["split_sha256"] = "a" * 64
  cfg["normalization"]["scale_constant_sources"] = {
    "brainweb": "unit_test", "hecktor": "unit_test",
  }
  config_path = tmp_path / "experiment.yaml"
  config_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
  case_dir = tmp_path / "arrays/brainweb/subject04/z125"
  case_dir.mkdir(parents=True)
  activity = np.full((128, 128), 3.0, dtype=np.float32)
  raw_path = case_dir / "raw_activity.npy"
  np.save(raw_path, activity, allow_pickle=False)
  raw_path.with_suffix(".json").write_text(json.dumps({
    "schema_version": 1, "artifact_kind": "canonical_activity",
    "dataset": "brainweb", "subject_id": "subject04", "slice_id": "z125",
    "split": "train", "activity_sha256": array_sha256(activity),
  }), encoding="utf-8")
  summary = prepare_cases(
    ROOT, config_path, tmp_path / "arrays", splits={"train"},
  )
  assert summary == {"activity_cases": 1, "poisson_cases": 1, "reused_groups": 0}
  sinogram = np.load(case_dir / "c200000/r0/sinogram.npy", allow_pickle=False)
  assert sinogram.shape == (180, 182) and np.issubdtype(sinogram.dtype, np.integer)
  assert (case_dir / "c200000/r0/mlem10.npy").is_file()
  resumed = prepare_cases(
    ROOT, config_path, tmp_path / "arrays", splits={"train"},
    resume_verified=True,
  )
  assert resumed["reused_groups"] == 3
