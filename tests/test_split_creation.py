import csv
from pathlib import Path

import yaml

from sam_kem.artifacts import file_sha256
from sam_kem.audit import _matches_expected_split_counts
from sam_kem.split_creation import UNTOUCHED, _stratified_select, create_subject_split
from sam_kem.splits import load_configured_subject_splits, split_counts


def test_split_is_subject_level_balanced_and_test_is_unexposed(tmp_path: Path):
  inventory = tmp_path / "inventory.csv"
  with inventory.open("w", encoding="utf-8", newline="") as handle:
    writer = csv.DictWriter(
      handle,
      fieldnames=("dataset", "subject_id", "stratum", "previous_exposure_status"),
    )
    writer.writeheader()
    for index in range(10):
      writer.writerow({
        "dataset": "brainweb", "subject_id": f"s{index:02d}",
        "stratum": "a" if index % 2 == 0 else "b",
        "previous_exposure_status": UNTOUCHED if index >= 4 else "development_allowed",
      })
  config = tmp_path / "config.yaml"
  config.write_text(yaml.safe_dump({
    "run_id": "clean_v1",
    "reproducibility": {"experiment_seed": 7},
    "data": {
      "split_file": "split.csv", "split_id": "clean_v1_subject_holdout",
      "split_sha256": None,
      "locked_test_required_exposure_status": UNTOUCHED,
      "expected_subject_counts": {
        "brainweb": {"train": 6, "validation": 2, "test_locked": 2},
      },
    },
  }), encoding="utf-8")
  output = tmp_path / "split.csv"
  create_subject_split(tmp_path, config, inventory, output)
  cfg = yaml.safe_load(config.read_text(encoding="utf-8"))
  cfg["data"]["split_sha256"] = file_sha256(output)
  rows = load_configured_subject_splits(tmp_path, cfg)
  assert split_counts(rows)["brainweb"] == {
    "train": 6, "validation": 2, "test_locked": 2,
  }
  assert all(
    row["previous_exposure_status"] == UNTOUCHED
    for row in rows if row["split"] == "test_locked"
  )


def test_unique_subject_strata_fall_back_to_seeded_global_selection():
  rows = [
    {"dataset": "hecktor", "subject_id": f"s{index:02d}", "stratum": f"subject_{index:02d}"}
    for index in range(20)
  ]
  selected = _stratified_select(rows, 5, seed=17)
  assert selected == _stratified_select(rows, 5, seed=17)
  assert {row["subject_id"] for row in selected} != {f"s{index:02d}" for index in range(5)}


def test_audit_treats_absent_zero_count_split_as_zero():
  assert _matches_expected_split_counts(
    {"train": 16, "validation": 4},
    {"train": 16, "validation": 4, "test_locked": 0},
  )
  assert not _matches_expected_split_counts(
    {"train": 16, "validation": 4, "test_locked": 1},
    {"train": 16, "validation": 4, "test_locked": 0},
  )
