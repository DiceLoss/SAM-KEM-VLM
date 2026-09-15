import csv
from types import SimpleNamespace

import pytest

from sam_kem.hyperparameter_freeze import FROZEN_FIELDS
from sam_kem.sweep_evaluation import SWEEP_FIELDS
from sam_kem.validation_only_reporting import (
  _selected_parameter_map,
  _stream_selected_rows,
)


def _selection_rows():
  methods = {
    "brainweb": ("MLEM", "OSEM", "Bowsher-KEM", "SelfKEM", "MR-KEM", "SAM"),
    "hecktor": ("MLEM", "OSEM", "Bowsher-KEM", "SelfKEM", "CT-KEM", "SAM"),
  }
  return [{
    "run_id": "run", "dataset": dataset, "count_level": count,
    "method": method, "sigma": 0.1, "radius": 1, "neighbours": 2,
    "iterations": 20, "alpha": 0.3,
    "validation_subject_mean_nrmse": 0.2,
    "validation_subject_mean_ssim": 0.9,
  } for dataset, dataset_methods in methods.items()
    for count in (100_000, 200_000, 500_000, 1_000_000)
    for method in dataset_methods]


def _write_csv(path, fields, rows):
  with path.open("w", encoding="utf-8", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)


def _sweep_row(subject="s1", realization="0"):
  return {
    "run_id": "run", "dataset": "brainweb", "subject_id": subject,
    "slice_id": "z001", "realization_id": realization, "count_level": "100000",
    "method": "MLEM", "split": "validation", "sigma": "0.1", "radius": "1",
    "neighbours": "2", "iterations": "20", "osem_subsets": "0",
    "effective_em_updates": "20", "alpha": "0.3", "nrmse": "0.2", "ssim": "0.9",
  }


def test_selected_parameter_map_requires_exact_frozen_order(tmp_path):
  rows = _selection_rows()
  frozen = tmp_path / "frozen.csv"
  _write_csv(frozen, FROZEN_FIELDS, rows)
  protocol = SimpleNamespace(
    methods=("MLEM", "OSEM", "Bowsher-KEM", "SelfKEM", "MR-KEM", "CT-KEM", "SAM"),
  )
  selected = _selected_parameter_map(
    {"selected_hyperparameters": rows}, frozen, protocol, "run",
    (100_000, 200_000, 500_000, 1_000_000),
  )
  assert len(selected) == 48
  assert selected[("brainweb", 100_000, "MLEM")] == (0.1, 1, 2, 20, 0.3)

  _write_csv(frozen, FROZEN_FIELDS, list(reversed(rows)))
  with pytest.raises(ValueError, match="row order"):
    _selected_parameter_map(
      {"selected_hyperparameters": rows}, frozen, protocol, "run",
      (100_000, 200_000, 500_000, 1_000_000),
    )


def test_stream_selected_rows_aggregates_subject_then_dataset(tmp_path):
  source = tmp_path / "source.csv"
  rows = [_sweep_row(realization="0"), _sweep_row(realization="1")]
  unselected = dict(_sweep_row(subject="s2"), iterations="40")
  _write_csv(source, SWEEP_FIELDS, [*rows, unselected])
  result = _stream_selected_rows(
    (source,), {("brainweb", 100_000, "MLEM"): (0.1, 1, 2, 20, 0.3)}, "run",
  )
  streamed, selected, by_source, subjects, summaries = result
  assert (streamed, selected) == (3, 2)
  assert by_source == {"original": 2}
  assert len(subjects) == 1
  assert subjects[0]["observation_count"] == 2
  assert subjects[0]["selection_and_reporting_same_split"] is True
  assert len(summaries) == 1
  assert summaries[0]["case_row_count"] == 2
  assert summaries[0]["subject_mean_nrmse"] == pytest.approx(0.2)


def test_stream_selected_rows_rejects_cross_source_duplicates(tmp_path):
  first = tmp_path / "first.csv"
  second = tmp_path / "second.csv"
  _write_csv(first, SWEEP_FIELDS, [_sweep_row()])
  _write_csv(second, SWEEP_FIELDS, [_sweep_row()])
  with pytest.raises(ValueError, match="multiple reviewed sources"):
    _stream_selected_rows(
      (first, second),
      {("brainweb", 100_000, "MLEM"): (0.1, 1, 2, 20, 0.3)}, "run",
    )
