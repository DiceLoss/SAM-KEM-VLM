"""Deterministic, checksum-bound packaging of audited validation priors."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
import subprocess
import tempfile
import zipfile
from pathlib import Path

from .artifacts import atomic_write_json, file_sha256, verify_csv_artifact, verify_success_marker
from .checkpoint_registration import validate_registered_decoder_configuration
from .config import load_yaml_strict


def _json(path: Path) -> dict:
  value = json.loads(path.read_text(encoding="utf-8"))
  if not isinstance(value, dict):
    raise ValueError(f"expected JSON object: {path}")
  return value


def _zip_info(name: str) -> zipfile.ZipInfo:
  info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
  info.compress_type = zipfile.ZIP_DEFLATED
  info.external_attr = 0o100644 << 16
  return info


def _write_file(archive: zipfile.ZipFile, source: Path, name: str) -> None:
  with source.open("rb") as input_handle, archive.open(_zip_info(name), "w") as output_handle:
    shutil.copyfileobj(input_handle, output_handle, length=1024 * 1024)


def _relative_file(root: Path, path: Path) -> tuple[Path, str]:
  resolved = path.resolve()
  try:
    relative = resolved.relative_to(root).as_posix()
  except ValueError as error:
    raise ValueError(f"bundle member lies outside repository root: {resolved}") from error
  if not resolved.is_file():
    raise FileNotFoundError(resolved)
  return resolved, relative


def build_validation_prior_bundle(
  root: Path, config_path: Path, prior_dir: Path, audit_path: Path,
  diagnostics_path: Path, output_path: Path, record_path: Path,
) -> dict:
  """Package only a completed, audited validation-prior set."""
  root = root.resolve()
  config_path = config_path.resolve()
  prior_dir = prior_dir.resolve()
  audit_path = audit_path.resolve()
  diagnostics_path = diagnostics_path.resolve()
  output_path = output_path.resolve()
  record_path = record_path.resolve()
  if output_path.exists() or record_path.exists():
    raise FileExistsError("refusing to overwrite validation-prior delivery artifacts")
  dirty = subprocess.run(
    ["git", "status", "--porcelain"], cwd=root, check=True,
    capture_output=True, text=True,
  ).stdout.strip()
  if dirty:
    raise RuntimeError(f"refusing to package from a dirty worktree: {dirty.splitlines()[:8]}")
  packaging_commit = subprocess.run(
    ["git", "rev-parse", "HEAD"], cwd=root, check=True,
    capture_output=True, text=True,
  ).stdout.strip()
  cfg = load_yaml_strict(config_path)
  validate_registered_decoder_configuration(root, config_path, cfg)
  verify_success_marker(prior_dir)

  prior_manifest = prior_dir / "prior_manifest.csv"
  prior_manifest_record = verify_csv_artifact(prior_manifest)
  if (
    prior_manifest_record.get("schema_name") != "medsam2_prior_manifest_v1"
    or prior_manifest_record.get("split") != "validation"
  ):
    raise ValueError("validation-prior manifest schema/split mismatch")
  with prior_manifest.open("r", encoding="utf-8", newline="") as handle:
    prior_rows = list(csv.DictReader(handle))
  audit = _json(audit_path)
  diagnostics_record = verify_csv_artifact(diagnostics_path)
  if (
    audit.get("status") != "PASS"
    or audit.get("records") != len(prior_rows)
    or audit.get("prior_manifest_sha256") != file_sha256(prior_manifest)
    or audit.get("diagnostics_sha256") != file_sha256(diagnostics_path)
    or diagnostics_record.get("schema_name") != "validation_prior_diagnostics_v1"
    or diagnostics_record.get("row_count") != len(prior_rows)
  ):
    raise ValueError("validation-prior audit is incomplete or mismatched")

  member_paths: dict[str, Path] = {}
  for row in prior_rows:
    if row.get("split") != "validation" or row.get("prior_variant") != "sam_prior":
      raise ValueError("prior bundle contains a non-validation or non-primary prior row")
    for field, sha_field in (
      ("prior_path", "prior_file_sha256"),
      ("metadata_path", "metadata_file_sha256"),
    ):
      path, relative = _relative_file(root, root / row[field])
      if file_sha256(path) != row[sha_field]:
        raise ValueError(f"prior manifest member SHA mismatch: {path}")
      if relative in member_paths:
        raise ValueError(f"duplicate prior bundle member: {relative}")
      member_paths[relative] = path

  evidence = [
    config_path,
    root / str(cfg["prompts"]["training_box_cache"]),
    (root / str(cfg["prompts"]["training_box_cache"])).with_suffix(".csv.manifest.json"),
    root / str(cfg["outputs"]["pretraining_manifest"]),
    (root / str(cfg["outputs"]["pretraining_manifest"])).with_suffix(".csv.manifest.json"),
    prior_manifest,
    prior_manifest.with_suffix(".csv.manifest.json"),
    prior_dir / "generation_record.json",
    prior_dir / "_SUCCESS.json",
    audit_path,
    diagnostics_path,
    diagnostics_path.with_suffix(".csv.manifest.json"),
    root / "logs/decoder_checkpoint_archive_registration.json",
    root / "logs/decoder_checkpoint_lfs_remote_verify.json",
  ]
  for path in evidence:
    resolved, relative = _relative_file(root, path)
    if relative in member_paths and member_paths[relative] != resolved:
      raise ValueError(f"conflicting validation-prior evidence member: {relative}")
    member_paths[relative] = resolved

  file_records = [
    {
      "path": relative,
      "sha256": file_sha256(path),
      "byte_size": path.stat().st_size,
    }
    for relative, path in sorted(member_paths.items())
  ]
  metadata = {
    "artifact_kind": "validation_prior_bundle_metadata",
    "schema_version": 1,
    "run_id": cfg["run_id"],
    "split": "validation",
    "packaging_git_commit": packaging_commit,
    "registered_config_sha256": file_sha256(config_path),
    "prior_manifest_sha256": file_sha256(prior_manifest),
    "prior_records": len(prior_rows),
    "audit_sha256": file_sha256(audit_path),
    "diagnostics_sha256": file_sha256(diagnostics_path),
  }
  metadata_bytes = (json.dumps(metadata, indent=2, sort_keys=True) + "\n").encode("utf-8")
  metadata_record = {
    "path": "environment/VALIDATION_PRIOR_BUNDLE_METADATA.json",
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
      with zipfile.ZipFile(
        handle, "w", compression=zipfile.ZIP_DEFLATED,
        compresslevel=6, allowZip64=True,
      ) as archive:
        for relative, path in sorted(member_paths.items()):
          _write_file(archive, path, relative)
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
    "artifact_kind": "validation_prior_archive_build",
    "schema_version": 1,
    "status": "local_bundle_complete_pending_lfs_publication",
    "run_id": cfg["run_id"],
    "split": "validation",
    "archive_path": output_path.relative_to(root).as_posix(),
    "archive_sha256": file_sha256(output_path),
    "archive_byte_size": output_path.stat().st_size,
    "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
    "manifest_member_count": len(manifest_records),
    "prior_records": len(prior_rows),
    "packaging_git_commit": packaging_commit,
    "registered_config_sha256": file_sha256(config_path),
    "prior_manifest_sha256": file_sha256(prior_manifest),
    "audit_sha256": file_sha256(audit_path),
    "immutable_archive_uri": None,
  }
  atomic_write_json(record_path, result)
  return result


def verify_validation_prior_bundle(archive_path: Path) -> dict:
  """Stream-verify exact membership and every checksum in a prior bundle."""
  archive_path = archive_path.resolve()
  with zipfile.ZipFile(archive_path) as archive:
    names = archive.namelist()
    if len(names) != len(set(names)):
      raise ValueError("validation-prior bundle contains duplicate names")
    if "MANIFEST.sha256" not in names:
      raise ValueError("validation-prior bundle lacks MANIFEST.sha256")
    manifest_bytes = archive.read("MANIFEST.sha256")
    expected = {}
    for line in manifest_bytes.decode("utf-8").splitlines():
      checksum, separator, name = line.partition("  ")
      if not separator or len(checksum) != 64 or not name or name in expected:
        raise ValueError(f"invalid validation-prior manifest line: {line!r}")
      expected[name] = checksum
    members = set(names) - {"MANIFEST.sha256"}
    if members != set(expected):
      raise ValueError("validation-prior bundle manifest membership mismatch")
    for name, expected_sha in expected.items():
      digest = hashlib.sha256()
      with archive.open(name) as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
          digest.update(chunk)
      if digest.hexdigest() != expected_sha:
        raise ValueError(f"validation-prior bundle member checksum mismatch: {name}")
    metadata = json.loads(
      archive.read("environment/VALIDATION_PRIOR_BUNDLE_METADATA.json")
    )
    if metadata.get("split") != "validation" or metadata.get("prior_records") != 8892:
      raise ValueError("validation-prior bundle metadata has wrong scope/count")
  return {
    "artifact_kind": "validation_prior_archive_verification",
    "schema_version": 1,
    "status": "complete",
    "archive_path": archive_path.as_posix(),
    "archive_sha256": file_sha256(archive_path),
    "archive_byte_size": archive_path.stat().st_size,
    "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
    "verified_member_count": len(expected),
    "prior_records": int(metadata["prior_records"]),
    "duplicate_member_count": 0,
    "missing_member_count": 0,
    "checksum_mismatch_count": 0,
  }
