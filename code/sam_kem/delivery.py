"""Checksum-bound packaging of completed decoder training bundles."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

from .artifacts import atomic_write_json, file_sha256, verify_csv_artifact
from .config import load_yaml_strict


def _json(path: Path) -> dict:
  value = json.loads(path.read_text(encoding="utf-8"))
  if not isinstance(value, dict):
    raise ValueError(f"expected a JSON object: {path}")
  return value


def _checkpoint_group(root: Path, config_path: Path, cfg: dict, dataset: str) -> tuple[list[Path], dict]:
  checkpoint = root / str(cfg["medsam2"]["checkpoint_paths"][dataset])
  resume = checkpoint.with_suffix(".resume.pt")
  provenance_path = checkpoint.with_suffix(".json")
  metrics = checkpoint.with_suffix(".training.csv")
  metrics_manifest = metrics.with_suffix(metrics.suffix + ".manifest.json")
  paths = [checkpoint, resume, provenance_path, metrics, metrics_manifest]
  missing = [path for path in paths if not path.is_file()]
  if missing:
    raise FileNotFoundError(f"incomplete {dataset} checkpoint group: {missing}")
  provenance = _json(provenance_path)
  expected = {
    "artifact_kind": "checkpoint_provenance",
    "dataset": dataset,
    "run_id": cfg["run_id"],
    "config_sha256": file_sha256(config_path),
    "checkpoint_sha256": file_sha256(checkpoint),
    "resume_checkpoint_sha256": file_sha256(resume),
    "training_metrics_sha256": file_sha256(metrics),
    "training_metrics_manifest_sha256": file_sha256(metrics_manifest),
  }
  mismatches = {
    key: (provenance.get(key), value)
    for key, value in expected.items() if provenance.get(key) != value
  }
  if mismatches:
    raise ValueError(f"{dataset} checkpoint provenance mismatch: {mismatches}")
  verify_csv_artifact(metrics)
  return paths, provenance


def _zip_info(name: str) -> zipfile.ZipInfo:
  info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
  info.compress_type = zipfile.ZIP_STORED
  info.external_attr = 0o100644 << 16
  return info


def _write_file(archive: zipfile.ZipFile, source: Path, name: str) -> None:
  with source.open("rb") as input_handle, archive.open(_zip_info(name), "w") as output_handle:
    shutil.copyfileobj(input_handle, output_handle, length=1024 * 1024)


def build_decoder_checkpoint_bundle(
  root: Path, config_path: Path, output_path: Path, record_path: Path,
) -> dict:
  """Verify and atomically package both decoder groups without publishing a URI claim."""
  root = root.resolve()
  config_path = config_path.resolve()
  output_path = output_path.resolve()
  record_path = record_path.resolve()
  if output_path.exists() or record_path.exists():
    raise FileExistsError("refusing to overwrite an existing decoder delivery artifact")
  cfg = load_yaml_strict(config_path)
  run_id = str(cfg["run_id"])
  packaging_commit = subprocess.run(
    ["git", "rev-parse", "HEAD"], cwd=root, check=True,
    capture_output=True, text=True,
  ).stdout.strip()
  dirty = subprocess.run(
    ["git", "status", "--porcelain"], cwd=root, check=True,
    capture_output=True, text=True,
  ).stdout.strip()
  if dirty:
    raise RuntimeError(f"refusing to package from a dirty worktree: {dirty.splitlines()[:5]}")

  files = [config_path]
  dataset_records = {}
  for dataset in ("brainweb", "hecktor"):
    group, provenance = _checkpoint_group(root, config_path, cfg, dataset)
    files.extend(group)
    dataset_records[dataset] = {
      "checkpoint_sha256": provenance["checkpoint_sha256"],
      "resume_checkpoint_sha256": provenance["resume_checkpoint_sha256"],
      "training_code_git_commit": provenance["code_git_commit"],
    }

  handoff_path = root / "logs/decoder_training_handoff_audit.json"
  handoff = _json(handoff_path)
  if handoff.get("status") != "complete" or handoff.get("run_id") != run_id:
    raise ValueError("decoder handoff audit is absent or incomplete")
  for dataset, record in dataset_records.items():
    if handoff.get("datasets", {}).get(dataset, {}).get("checkpoint_sha256") != record["checkpoint_sha256"]:
      raise ValueError(f"handoff audit checkpoint mismatch: {dataset}")
  files.append(handoff_path)

  correction = root / "logs/decoder_training_commit_identity_correction.json"
  if correction.is_file():
    files.append(correction)
  protocol_dir = config_path.with_name(f"{config_path.stem}_protocols")
  files.extend(sorted(path for path in protocol_dir.rglob("*") if path.is_file()))
  gpu_logs = sorted((root / "logs").glob(f"gpu_{run_id}_*.log"))
  if not gpu_logs:
    raise FileNotFoundError(f"no GPU execution log found for {run_id}")
  files.extend(gpu_logs)

  unique_files = sorted(set(files), key=lambda path: path.relative_to(root).as_posix())
  file_records = [
    {
      "path": path.relative_to(root).as_posix(),
      "sha256": file_sha256(path),
      "byte_size": path.stat().st_size,
    }
    for path in unique_files
  ]
  metadata = {
    "artifact_kind": "decoder_checkpoint_bundle_metadata",
    "schema_version": 1,
    "run_id": run_id,
    "packaging_git_commit": packaging_commit,
    "config_path": config_path.relative_to(root).as_posix(),
    "config_sha256": file_sha256(config_path),
    "python": sys.version,
    "platform": platform.platform(),
    "datasets": dataset_records,
  }
  metadata_bytes = (json.dumps(metadata, indent=2, sort_keys=True) + "\n").encode("utf-8")
  metadata_record = {
    "path": "environment/BUNDLE_METADATA.json",
    "sha256": hashlib.sha256(metadata_bytes).hexdigest(),
    "byte_size": len(metadata_bytes),
  }
  manifest_records = [*file_records, metadata_record]
  manifest_bytes = "".join(
    f"{record['sha256']}  {record['path']}\n" for record in manifest_records
  ).encode("utf-8")

  output_path.parent.mkdir(parents=True, exist_ok=True)
  descriptor, temporary_name = tempfile.mkstemp(
    prefix=f".{output_path.name}.", dir=output_path.parent,
  )
  temporary = Path(temporary_name)
  handle = None
  try:
    handle = os.fdopen(descriptor, "w+b")
    with handle:
      with zipfile.ZipFile(handle, "w", allowZip64=True) as archive:
        for path in unique_files:
          _write_file(archive, path, path.relative_to(root).as_posix())
        archive.writestr(_zip_info(metadata_record["path"]), metadata_bytes)
        archive.writestr(_zip_info("MANIFEST.sha256"), manifest_bytes)
      handle.flush()
      os.fsync(handle.fileno())
    os.replace(temporary, output_path)
  except BaseException:
    if handle is None:
      os.close(descriptor)
    temporary.unlink(missing_ok=True)
    raise

  result = {
    "artifact_kind": "decoder_checkpoint_archive_build",
    "schema_version": 1,
    "status": "local_bundle_complete_pending_immutable_upload",
    "run_id": run_id,
    "archive_path": (
      output_path.relative_to(root).as_posix()
      if output_path.is_relative_to(root) else output_path.as_posix()
    ),
    "archive_sha256": file_sha256(output_path),
    "archive_byte_size": output_path.stat().st_size,
    "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
    "manifest_member_count": len(manifest_records),
    "packaging_git_commit": packaging_commit,
    "datasets": dataset_records,
    "immutable_archive_uri": None,
  }
  atomic_write_json(record_path, result)
  return result
