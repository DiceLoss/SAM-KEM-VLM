import csv
from collections import Counter
from pathlib import Path

import numpy as np

from sam_kem.source_preparation import (
  _copy_hecktor_pet_slice, _extract_npz_member, _usable_brainweb_slice_indices,
)


ROOT = Path(__file__).resolve().parents[1]


def test_npz_input_is_streamed_without_selecting_other_members(tmp_path: Path):
  archive = tmp_path / "source.npz"
  expected = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
  np.savez(archive, input=expected, unrelated_label=np.ones((50, 50), dtype=np.uint8))
  extracted = _extract_npz_member(archive, "input", tmp_path / "input.npy")
  np.testing.assert_array_equal(np.load(extracted, mmap_mode="r"), expected)


def test_extracted_hecktor_slice_owns_its_memory():
  source = np.zeros((1, 144, 144, 144, 2), dtype=np.float32)
  source[0, 72, :, :, 1] = 2.0
  pet = _copy_hecktor_pet_slice(source, 0)
  assert not np.shares_memory(pet, source)
  np.testing.assert_array_equal(pet, np.full((144, 144), 2.0, dtype=np.float32))


def test_brainweb_uses_all_and_only_positive_activity_planes():
  volume = np.zeros((256, 256, 8), dtype=np.float32)
  volume[10, 20, 1] = 1.0
  volume[30, 40, 3:6] = 2.0
  assert _usable_brainweb_slice_indices(volume) == [1, 3, 4, 5]


def test_pinned_brainweb_plane_counts_match_committed_trace_volumes():
  with (ROOT / "inputs/clean_v1/subject_split.csv").open(
    encoding="utf-8", newline="",
  ) as handle:
    split = {
      row["subject_id"]: row["split"] for row in csv.DictReader(handle)
      if row["dataset"] == "brainweb"
    }
  counts = Counter()
  archives = sorted((ROOT / "data/raw_pet").glob("raw_pet_brainweb_subject*.npz"))
  assert len(archives) == 20
  for archive in archives:
    with np.load(archive, allow_pickle=False) as content:
      keys = [key for key in content.files if key.endswith("__true_pet")]
      assert len(keys) == 1
      subject = keys[0].split("__", 1)[0]
      counts[split[subject]] += len(_usable_brainweb_slice_indices(content[keys[0]]))
  assert counts == {"train": 2815, "validation": 696}
