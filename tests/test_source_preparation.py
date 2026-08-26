from pathlib import Path

import numpy as np

from sam_kem.source_preparation import _extract_npz_member


def test_npz_input_is_streamed_without_selecting_other_members(tmp_path: Path):
  archive = tmp_path / "source.npz"
  expected = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
  np.savez(archive, input=expected, unrelated_label=np.ones((50, 50), dtype=np.uint8))
  extracted = _extract_npz_member(archive, "input", tmp_path / "input.npy")
  np.testing.assert_array_equal(np.load(extracted, mmap_mode="r"), expected)
