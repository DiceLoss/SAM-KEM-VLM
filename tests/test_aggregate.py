import csv
from pathlib import Path

import pytest

from sam_kem.aggregate import aggregate_cross_count


FIELDS = ["dataset", "subject_id", "slice_id", "realization_id", "count_level", "method", "psnr_db", "ssim", "nrmse", "mae"]


def test_duplicate_case_rows_are_rejected(tmp_path: Path):
  source = tmp_path / "cases.csv"
  row = dict(zip(FIELDS, ["brainweb", "s1", "z1", "0", "100000", "MLEM", "20", ".5", ".3", ".1"]))
  with source.open("w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=FIELDS); writer.writeheader(); writer.writerows([row, row])
  with pytest.raises(ValueError, match="duplicate"):
    aggregate_cross_count(source, tmp_path / "out.csv")

