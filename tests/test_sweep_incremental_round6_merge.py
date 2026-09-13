from pathlib import Path

import pytest
from sam_kem.sweep_evaluation import SWEEP_FIELDS
from sam_kem.sweep_incremental_round6_merge import stream_merge_round6_fragments


def _fragment(path: Path, value: str | None) -> None:
  text = ",".join(SWEEP_FIELDS) + "\n"
  if value is not None:
    text += ",".join([value] * len(SWEEP_FIELDS)) + "\n"
  path.write_text(text, encoding="utf-8")


def test_round6_merge_enforces_reviewed_row_count(tmp_path):
  first = tmp_path / "first.csv"
  second = tmp_path / "second.csv"
  _fragment(first, "a")
  _fragment(second, "b")
  output = tmp_path / "merged.csv"
  result = stream_merge_round6_fragments(
    [first, second], output, expected_rows=2,
  )
  assert result["row_count"] == 2
  assert output.read_text(encoding="utf-8").count("\n") == 3


def test_round6_merge_preserves_header_with_zero_row_fragment(tmp_path):
  first = tmp_path / "first.csv"
  empty = tmp_path / "empty.csv"
  _fragment(first, "a")
  _fragment(empty, None)
  output = tmp_path / "merged.csv"
  result = stream_merge_round6_fragments(
    [first, empty], output, expected_rows=1,
  )
  lines = output.read_text(encoding="utf-8").splitlines()
  assert result["row_count"] == 1
  assert lines == [",".join(SWEEP_FIELDS), ",".join(["a"] * len(SWEEP_FIELDS))]


def test_round6_merge_rejects_wrong_row_count(tmp_path):
  fragment = tmp_path / "fragment.csv"
  _fragment(fragment, "a")
  with pytest.raises(ValueError, match="has 1 rows, expected 2"):
    stream_merge_round6_fragments(
      [fragment], tmp_path / "merged.csv", expected_rows=2,
    )
