from pathlib import Path

import pytest

from sam_kem.boxes import Box, BoxCache, BoxKey, BoxRecord, centroid_fallback
from sam_kem.config import ConfigurationError, load_yaml_strict
from sam_kem.reproducibility import CaseKey, case_seed


def test_duplicate_yaml_keys_are_rejected(tmp_path: Path):
  path = tmp_path / "bad.yaml"
  path.write_text("sam:\n  a: 1\nsam:\n  a: 2\n", encoding="utf-8")
  with pytest.raises(ConfigurationError, match="duplicate YAML key"):
    load_yaml_strict(path)


def test_seed_uses_complete_case_identity():
  first = CaseKey("brainweb", "subject1", "slice1", 100000, 0)
  variants = [
    CaseKey("hecktor", "subject1", "slice1", 100000, 0),
    CaseKey("brainweb", "subject2", "slice1", 100000, 0),
    CaseKey("brainweb", "subject1", "slice2", 100000, 0),
    CaseKey("brainweb", "subject1", "slice1", 200000, 0),
    CaseKey("brainweb", "subject1", "slice1", 100000, 1),
  ]
  seeds = {case_seed(first, experiment_seed=7)} | {case_seed(key, experiment_seed=7) for key in variants}
  assert len(seeds) == 6


def test_centroid_fallback_and_zero_mass_rule():
  image = __import__("numpy").zeros((128, 128), dtype=float)
  assert centroid_fallback(image, box_width=32, box_height=32).__dict__ == {
    "xmin": 48, "ymin": 48, "xmax": 80, "ymax": 80,
  }
  image[10, 20] = 3
  box = centroid_fallback(image, box_width=32, box_height=32)
  assert box.xmin == 4 and box.ymin == 0


def test_box_cache_roundtrip_keeps_anchor_and_request_provenance(tmp_path):
  record = BoxRecord(
    BoxKey("brainweb", "s1", "z1", 200000), Box(1, 2, 20, 30),
    "qwen3-vl-plus", "request-1", "a" * 64, "b" * 64, "c" * 64,
    "qwen3-vl-plus",
  )
  path = tmp_path / "boxes.csv"
  BoxCache([record]).write(path)
  assert BoxCache.read(path).get(record.key) == record
