from pathlib import Path

import numpy as np
import pytest
from sam_kem.artifacts import file_sha256
from sam_kem.medsam_training import _balanced_order, _training_input_bindings
from sam_kem.training import (
  TrainingSample,
  balanced_sample_weights,
  validate_training_samples,
)


def sample(subject, count):
  return TrainingSample("brainweb", subject, "s1", count, Path("x"), Path("y"), (0, 0, 1, 1))


def test_four_count_training_and_balancing():
  samples = [sample("a", 100000), sample("a", 100000), sample("b", 200000), sample("b", 500000), sample("b", 1000000)]
  selected = validate_training_samples(
    samples, dataset="brainweb", train_subjects={"a", "b"},
    count_levels=(100000, 200000, 500000, 1000000),
  )
  weights = balanced_sample_weights(selected)
  assert weights[0] == weights[1] == 0.5
  assert np.all(weights[2:] == 1.0)


def test_validation_subject_is_rejected():
  with pytest.raises(ValueError, match="non-train subject"):
    validate_training_samples(
      [sample("validation_subject", 100000)], dataset="brainweb",
      train_subjects={"train_subject"}, count_levels=(100000,),
    )


def test_decoder_schedule_balances_count_then_subject():
  rows = [
    {
      "count_level": str(count), "subject_id": subject,
      "slice_id": slice_id, "realization_id": "0",
    }
    for count in (100, 200)
    for subject in ("a", "b", "c")
    for slice_id in ("z1", "z2")
  ]
  order = _balanced_order(rows, (100, 200), seed=7, total_steps=24)
  counts = {count: 0 for count in (100, 200)}
  subjects = {subject: 0 for subject in ("a", "b", "c")}
  cases = set()
  for row in order:
    counts[int(row["count_level"])] += 1
    subjects[row["subject_id"]] += 1
    cases.add((row["count_level"], row["subject_id"], row["slice_id"]))
  assert counts == {100: 12, 200: 12}
  assert subjects == {"a": 8, "b": 8, "c": 8}
  assert len(cases) == 12


def test_training_input_bindings_cover_manifest_and_box_cache(tmp_path):
  manifest = tmp_path / "cases.csv"
  cache = tmp_path / "boxes.csv"
  files = (
    manifest, manifest.with_suffix(".csv.manifest.json"),
    cache, cache.with_suffix(".csv.manifest.json"),
  )
  for index, path in enumerate(files):
    path.write_text(f"artifact-{index}\n", encoding="utf-8")
  bindings = _training_input_bindings(manifest, cache)
  assert bindings == {
    "training_manifest_sha256": file_sha256(files[0]),
    "training_manifest_manifest_sha256": file_sha256(files[1]),
    "training_box_cache_sha256": file_sha256(files[2]),
    "training_box_cache_manifest_sha256": file_sha256(files[3]),
  }
