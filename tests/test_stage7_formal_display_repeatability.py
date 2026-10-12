import csv
import sys
from pathlib import Path

import pytest

EXPERIMENTS = Path(__file__).resolve().parents[1] / "code" / "experiments"
sys.path.insert(0, str(EXPERIMENTS))

from run_stage7_formal_display_repeatability import _rows, call_plan


def test_call_plan_freezes_exact_budget_and_families():
    assert call_plan() == (
        ("display", "grayscale_only", 0),
        ("display", "jet_only", 0),
        ("display", "grayscale_plus_jet", 0),
        ("repeatability", "grayscale_plus_jet", 1),
        ("repeatability", "grayscale_plus_jet", 2),
    )


def test_selection_requires_twenty_unique_cases(tmp_path):
    path = tmp_path / "cases.csv"
    fields = ("dataset", "subject_id", "slice_id", "count_level", "realization_id")
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for index in range(20):
            writer.writerow({
                "dataset": "brainweb", "subject_id": f"s{index}",
                "slice_id": "z001", "count_level": 100000,
                "realization_id": 0,
            })
    assert len(_rows(path)) == 20
    with path.open("a", encoding="utf-8") as stream:
        stream.write("brainweb,s0,z001,100000,0\n")
    with pytest.raises(ValueError, match="20 unique"):
        _rows(path)
