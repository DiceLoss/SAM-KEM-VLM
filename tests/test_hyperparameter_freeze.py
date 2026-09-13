from types import SimpleNamespace

import pytest

from sam_kem.hyperparameter_freeze import FROZEN_FIELDS, _validated_rows


def _rows():
  methods = {
    "brainweb": ("MLEM", "OSEM", "Bowsher-KEM", "SelfKEM", "MR-KEM", "SAM"),
    "hecktor": ("MLEM", "OSEM", "Bowsher-KEM", "SelfKEM", "CT-KEM", "SAM"),
  }
  return [{
    "run_id": "run", "dataset": dataset, "count_level": count,
    "method": method, "sigma": 0.1, "radius": 1, "neighbours": 0,
    "iterations": 20, "alpha": 0.0,
    "validation_subject_mean_nrmse": 0.2,
    "validation_subject_mean_ssim": 0.9,
  } for dataset, dataset_methods in methods.items()
    for count in (100_000, 200_000, 500_000, 1_000_000)
    for method in dataset_methods]


def _protocol():
  return SimpleNamespace(
    methods=("MLEM", "OSEM", "Bowsher-KEM", "SelfKEM", "MR-KEM", "CT-KEM", "SAM"),
  )


def test_validated_rows_accepts_complete_reviewed_domain():
  rows = _validated_rows(
    {"selected_hyperparameters": _rows()}, _protocol(), "run",
    (100_000, 200_000, 500_000, 1_000_000),
  )
  assert len(rows) == 48
  assert list(rows[0]) == FROZEN_FIELDS


def test_validated_rows_rejects_duplicate_identity():
  rows = _rows()
  rows[-1] = dict(rows[0])
  with pytest.raises(ValueError, match="duplicate frozen selection"):
    _validated_rows(
      {"selected_hyperparameters": rows}, _protocol(), "run",
      (100_000, 200_000, 500_000, 1_000_000),
    )
