"""Atomic, checksum-bound scientific result and checkpoint artifacts."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import tempfile
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

import numpy as np


def file_sha256(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as handle:
    for chunk in iter(lambda: handle.read(chunk_size), b""):
      digest.update(chunk)
  return digest.hexdigest()


def _atomic_bytes(path: Path, payload: bytes, *, replace: bool = False) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  if path.exists() and not replace:
    raise FileExistsError(f"refusing to overwrite immutable artifact: {path}")
  descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
  temporary = Path(temporary_name)
  try:
    with os.fdopen(descriptor, "wb") as handle:
      handle.write(payload); handle.flush(); os.fsync(handle.fileno())
    os.replace(temporary, path)
  except BaseException:
    temporary.unlink(missing_ok=True)
    raise


def atomic_write_json(path: Path, value: dict, *, replace: bool = False) -> None:
  _atomic_bytes(
    path, (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8"),
    replace=replace,
  )


def atomic_write_text(path: Path, value: str, *, replace: bool = False) -> None:
  _atomic_bytes(path, value.encode("utf-8"), replace=replace)


def atomic_write_bytes(path: Path, value: bytes, *, replace: bool = False) -> None:
  _atomic_bytes(path, value, replace=replace)


def atomic_save_numpy(path: Path, value: np.ndarray, *, replace: bool = False) -> None:
  """Atomically publish one unpickled NumPy array."""
  path.parent.mkdir(parents=True, exist_ok=True)
  if path.exists() and not replace:
    raise FileExistsError(f"refusing to overwrite immutable array: {path}")
  descriptor, temporary_name = tempfile.mkstemp(
    prefix=f".{path.name}.", suffix=".npy", dir=path.parent,
  )
  temporary = Path(temporary_name)
  try:
    with os.fdopen(descriptor, "wb") as handle:
      np.save(handle, np.asarray(value), allow_pickle=False)
      handle.flush(); os.fsync(handle.fileno())
    os.replace(temporary, path)
  except BaseException:
    temporary.unlink(missing_ok=True)
    raise


@contextmanager
def atomic_csv_writer(path: Path, fieldnames: list[str]) -> Iterator[csv.DictWriter]:
  """Yield a DictWriter and publish the file only after a successful close."""
  path.parent.mkdir(parents=True, exist_ok=True)
  if path.exists() or path.with_suffix(path.suffix + ".manifest.json").exists():
    raise FileExistsError(f"refusing to overwrite immutable CSV artifact: {path}")
  descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
  temporary = Path(temporary_name)
  try:
    with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
      writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="raise")
      writer.writeheader()
      yield writer
      handle.flush(); os.fsync(handle.fileno())
    os.replace(temporary, path)
  except BaseException:
    temporary.unlink(missing_ok=True)
    raise


def finalize_csv(
  path: Path, *, schema_name: str, primary_key: list[str], split: str | None = None,
) -> Path:
  with path.open("r", encoding="utf-8", newline="") as handle:
    reader = csv.DictReader(handle)
    rows = list(reader)
    columns = list(reader.fieldnames or ())
  missing_key = set(primary_key) - set(columns)
  if missing_key:
    raise ValueError(f"CSV {path} lacks primary-key columns: {sorted(missing_key)}")
  keys = [tuple(row[name] for name in primary_key) for row in rows]
  if len(keys) != len(set(keys)):
    raise ValueError(f"CSV {path} contains duplicate primary keys")
  sidecar = path.with_suffix(path.suffix + ".manifest.json")
  atomic_write_json(sidecar, {
    "schema_version": 1, "schema_name": schema_name,
    "artifact_path": path.name, "artifact_sha256": file_sha256(path),
    "byte_size": path.stat().st_size, "row_count": len(rows),
    "columns": columns, "primary_key": primary_key, "split": split,
    "completed_utc": datetime.now(timezone.utc).isoformat(),
  })
  return sidecar


def verify_csv_artifact(path: Path) -> dict:
  sidecar = path.with_suffix(path.suffix + ".manifest.json")
  record = json.loads(sidecar.read_text(encoding="utf-8"))
  if record.get("artifact_sha256") != file_sha256(path):
    raise ValueError(f"CSV checksum mismatch: {path}")
  with path.open("r", encoding="utf-8", newline="") as handle:
    reader = csv.DictReader(handle); rows = list(reader)
  if list(reader.fieldnames or ()) != record.get("columns"):
    raise ValueError(f"CSV schema differs from sidecar: {path}")
  if len(rows) != int(record.get("row_count", -1)):
    raise ValueError(f"CSV row count differs from sidecar: {path}")
  keys = [tuple(row[name] for name in record["primary_key"]) for row in rows]
  if len(keys) != len(set(keys)):
    raise ValueError(f"CSV primary key is not unique: {path}")
  return record


def write_success_marker(output_dir: Path, artifacts: list[Path]) -> Path:
  records = []
  for artifact in artifacts:
    if not artifact.is_file():
      raise FileNotFoundError(f"cannot finalize incomplete run; missing {artifact}")
    records.append({
      "path": Path(os.path.relpath(artifact, output_dir)).as_posix(),
      "sha256": file_sha256(artifact), "byte_size": artifact.stat().st_size,
    })
  marker = output_dir / "_SUCCESS.json"
  atomic_write_json(marker, {
    "schema_version": 1, "status": "complete",
    "completed_utc": datetime.now(timezone.utc).isoformat(), "artifacts": records,
  })
  return marker


def verify_success_marker(output_dir: Path) -> dict:
  marker = output_dir / "_SUCCESS.json"
  record = json.loads(marker.read_text(encoding="utf-8"))
  if record.get("status") != "complete" or not record.get("artifacts"):
    raise ValueError(f"invalid run completion marker: {marker}")
  for artifact in record["artifacts"]:
    path = (output_dir / artifact["path"]).resolve()
    if (
      not path.is_file() or file_sha256(path) != artifact.get("sha256")
      or path.stat().st_size != int(artifact.get("byte_size", -1))
    ):
      raise ValueError(f"run artifact missing or changed after completion: {path}")
  return record


def atomic_torch_save(payload, path: Path, *, replace: bool = False) -> None:
  """Atomically publish a torch serialization; replace is only for rolling resume state."""
  import torch

  path.parent.mkdir(parents=True, exist_ok=True)
  if path.exists() and not replace:
    raise FileExistsError(f"refusing to overwrite immutable checkpoint: {path}")
  descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
  os.close(descriptor)
  temporary = Path(temporary_name)
  try:
    torch.save(payload, temporary)
    with temporary.open("rb") as handle:
      os.fsync(handle.fileno())
    os.replace(temporary, path)
  except BaseException:
    temporary.unlink(missing_ok=True)
    raise
