from pathlib import Path

import pytest

from sam_kem.sweep_evaluation import SWEEP_FIELDS
from sam_kem.sweep_merge import stream_merge_csv_fragments


def _fragment(path: Path, rows: list[list[str]], fields: list[str] | None = None) -> None:
  columns = SWEEP_FIELDS if fields is None else fields
  path.write_text(
    ",".join(columns) + "\n" + "".join(",".join(row) + "\n" for row in rows),
    encoding="utf-8",
  )


def test_stream_merge_preserves_order_and_one_header(tmp_path):
  width = len(SWEEP_FIELDS)
  first = tmp_path / "first.csv"
  second = tmp_path / "second.csv"
  _fragment(first, [["a"] * width, ["b"] * width])
  _fragment(second, [["c"] * width])
  output = tmp_path / "merged.csv"
  result = stream_merge_csv_fragments([first, second], output)
  assert result == {"row_count": 3, "columns": SWEEP_FIELDS}
  lines = output.read_text(encoding="utf-8").splitlines()
  assert len(lines) == 4
  assert lines[0] == ",".join(SWEEP_FIELDS)
  assert lines[1:] == [",".join([value] * width) for value in ("a", "b", "c")]


def test_stream_merge_rejects_header_drift(tmp_path):
  width = len(SWEEP_FIELDS)
  first = tmp_path / "first.csv"
  second = tmp_path / "second.csv"
  _fragment(first, [["a"] * width])
  _fragment(second, [["b"] * (width - 1)], fields=SWEEP_FIELDS[:-1])
  with pytest.raises(ValueError, match="header differs"):
    stream_merge_csv_fragments([first, second], tmp_path / "merged.csv")
