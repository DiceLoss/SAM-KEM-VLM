import pytest

from sam_kem.prior_preflight import select_validation_smoke_rows


def _row(dataset, subject, slice_id, *, split="validation", count=200000, realization=0):
  return {
    "dataset": dataset,
    "subject_id": subject,
    "slice_id": slice_id,
    "split": split,
    "count_level": str(count),
    "realization_id": str(realization),
  }


def test_validation_prior_smoke_selection_is_deterministic_per_dataset():
  rows = [
    _row("brainweb", "subject44", "z100"),
    _row("hecktor", "108", "z072"),
    _row("brainweb", "subject04", "z039"),
    _row("brainweb", "subject04", "z038"),
    _row("hecktor", "10", "z072"),
    _row("hecktor", "1", "z072", realization=1),
    _row("brainweb", "subject01", "z001", split="train"),
  ]
  selected = select_validation_smoke_rows(
    rows, datasets=("brainweb", "hecktor"), count_level=200000,
    realization_id=0,
  )
  assert [
    (row["dataset"], row["subject_id"], row["slice_id"]) for row in selected
  ] == [
    ("brainweb", "subject04", "z038"),
    ("hecktor", "10", "z072"),
  ]


def test_validation_prior_smoke_selection_rejects_missing_dataset():
  with pytest.raises(ValueError, match="no validation smoke candidate for hecktor"):
    select_validation_smoke_rows(
      [_row("brainweb", "subject04", "z038")],
      datasets=("brainweb", "hecktor"), count_level=200000,
      realization_id=0,
    )
