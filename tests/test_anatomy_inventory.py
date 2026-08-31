from pathlib import Path

import numpy as np
import pytest

from sam_kem.anatomy_inventory import brainweb_subject_root, summarize_ct_slices


def test_brainweb_subject_root_requires_canonical_layout():
  assert brainweb_subject_root(
    Path("/data/subject04/sim_0/true_pet.nii.gz")
  ) == Path("/data/subject04")
  with pytest.raises(ValueError, match="unexpected BrainWeb activity layout"):
    brainweb_subject_root(Path("/data/subject04/pet.nii.gz"))


def test_ct_inventory_preserves_negative_hu_and_reports_shape():
  first = np.array([[-1000.0, 0.0], [100.0, 500.0]], dtype=np.float32)
  second = np.array([[-900.0, -10.0], [200.0, 1200.0]], dtype=np.float32)
  summary = summarize_ct_slices([first, second])
  assert summary["records"] == 2
  assert summary["unique_shapes"] == [[2, 2]]
  assert summary["overall_min"] == -1000.0
  assert summary["overall_max"] == 1200.0
  assert summary["nonfinite_values"] == 0
