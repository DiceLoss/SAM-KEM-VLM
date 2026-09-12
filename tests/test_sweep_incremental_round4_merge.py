from pathlib import Path

import pytest
from sam_kem.sweep_evaluation import SWEEP_FIELDS
from sam_kem.sweep_incremental_round4_merge import stream_merge_round4_fragments


def _fragment(path: Path, value: str) -> None:
  row = [value] * len(SWEEP_FIELDS)
  path.write_text(
    ",".join(SWEEP_FIELDS) + "\n" + ",".join(row) + "\n",
    encoding="utf-8",
  )


def test_round4_merge_enforces_reviewed_row_count(tmp_path):
  first = tmp_path / "first.csv"
  second = tmp_path / "second.csv"
  _fragment(first, "a")
  _fragment(second, "b")
  output = tmp_path / "merged.csv"
  result = stream_merge_round4_fragments(
    [first, second], output, expected_rows=2,
  )
  assert result["row_count"] == 2
  assert output.read_text(encoding="utf-8").count("\n") == 3


def test_round4_merge_rejects_wrong_row_count(tmp_path):
  fragment = tmp_path / "fragment.csv"
  _fragment(fragment, "a")
  with pytest.raises(ValueError, match="has 1 rows, expected 2"):
    stream_merge_round4_fragments(
      [fragment], tmp_path / "merged.csv", expected_rows=2,
    )
