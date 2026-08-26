import json

import numpy as np
import pytest

from sam_kem.artifacts import (
  atomic_csv_writer, atomic_save_numpy, finalize_csv, verify_csv_artifact,
  write_success_marker,
)


def test_atomic_csv_is_not_published_after_failed_write(tmp_path):
  output = tmp_path / "result.csv"
  with pytest.raises(RuntimeError):
    with atomic_csv_writer(output, ["id", "value"]) as writer:
      writer.writerow({"id": "1", "value": "x"})
      raise RuntimeError("interrupted")
  assert not output.exists()


def test_csv_manifest_detects_tampering_and_duplicate_keys(tmp_path):
  output = tmp_path / "result.csv"
  with atomic_csv_writer(output, ["id", "value"]) as writer:
    writer.writerow({"id": "1", "value": "x"})
  manifest = finalize_csv(
    output, schema_name="test_v1", primary_key=["id"], split="validation",
  )
  assert verify_csv_artifact(output)["row_count"] == 1
  output.write_text("id,value\n1,changed\n", encoding="utf-8")
  with pytest.raises(ValueError, match="checksum"):
    verify_csv_artifact(output)
  assert json.loads(manifest.read_text())["primary_key"] == ["id"]


def test_atomic_csv_bytes_use_platform_independent_lf(tmp_path):
  output = tmp_path / "result.csv"
  with atomic_csv_writer(output, ["id", "value"]) as writer:
    writer.writerow({"id": "1", "value": "x"})
  assert output.read_bytes() == b"id,value\n1,x\n"


def test_success_marker_requires_all_artifacts(tmp_path):
  present = tmp_path / "a.txt"; present.write_text("a", encoding="utf-8")
  with pytest.raises(FileNotFoundError):
    write_success_marker(tmp_path, [present, tmp_path / "missing.txt"])
  assert not (tmp_path / "_SUCCESS.json").exists()


def test_csv_manifest_rejects_duplicate_primary_keys(tmp_path):
  output = tmp_path / "duplicate.csv"
  with atomic_csv_writer(output, ["id", "value"]) as writer:
    writer.writerow({"id": "1", "value": "first"})
    writer.writerow({"id": "1", "value": "second"})
  with pytest.raises(ValueError, match="duplicate primary key"):
    finalize_csv(output, schema_name="test_v1", primary_key=["id"])
  assert not output.with_suffix(".csv.manifest.json").exists()


def test_atomic_numpy_refuses_overwrite(tmp_path):
  path = tmp_path / "array.npy"
  atomic_save_numpy(path, np.array([1.0], dtype=np.float32))
  np.testing.assert_array_equal(np.load(path, allow_pickle=False), [1.0])
  with pytest.raises(FileExistsError):
    atomic_save_numpy(path, np.array([2.0], dtype=np.float32))
