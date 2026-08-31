"""Register a versioned validation-prior archive for development sweeping."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from urllib.parse import urlparse

import yaml

from .artifacts import (
  atomic_write_json, atomic_write_text, file_sha256, verify_csv_artifact,
  verify_success_marker,
)
from .checkpoint_registration import validate_registered_decoder_configuration
from .config import load_yaml_strict
from .prior_delivery import verify_validation_prior_bundle


def _json(path: Path) -> dict:
  value = json.loads(path.read_text(encoding="utf-8"))
  if not isinstance(value, dict):
    raise ValueError(f"expected JSON object: {path}")
  return value


def _relative(path: Path, root: Path) -> str:
  try:
    return path.resolve().relative_to(root.resolve()).as_posix()
  except ValueError:
    return path.resolve().as_posix()


def _validate_uri(uri: str, version: str) -> None:
  parsed = urlparse(uri)
  if parsed.scheme.lower() in ("", "file"):
    raise ValueError("validation-prior archive URI must not be local or relative")
  if parsed.scheme.lower() in ("http", "https") and not parsed.netloc:
    raise ValueError("validation-prior HTTP URI must include a host")
  if not version.strip():
    raise ValueError("validation-prior archive version/LFS object identity is required")


def register_validation_prior_archive(
  root: Path, source_config_path: Path, build_record_path: Path,
  verify_record_path: Path, verified_download_path: Path, *, archive_uri: str,
  archive_version: str, output_config_path: Path, registration_record_path: Path,
) -> dict:
  """Create a downstream config without mutating decoder-registration evidence."""
  root = root.resolve()
  source_config_path = source_config_path.resolve()
  build_record_path = build_record_path.resolve()
  verify_record_path = verify_record_path.resolve()
  verified_download_path = verified_download_path.resolve()
  output_config_path = output_config_path.resolve()
  registration_record_path = registration_record_path.resolve()
  if output_config_path.exists() or registration_record_path.exists():
    raise FileExistsError("refusing to overwrite validation-prior registration artifacts")
  _validate_uri(archive_uri, archive_version)
  cfg = load_yaml_strict(source_config_path)
  validate_registered_decoder_configuration(root, source_config_path, cfg)
  build = _json(build_record_path)
  verification = _json(verify_record_path)
  if build.get("status") != "local_bundle_complete_pending_lfs_publication":
    raise ValueError("validation-prior build record is not ready for registration")
  if verification.get("status") != "complete":
    raise ValueError("validation-prior verification record is incomplete")
  shared = (
    "archive_sha256", "archive_byte_size", "manifest_sha256", "prior_records",
  )
  mismatches = {
    key: (build.get(key), verification.get(key))
    for key in shared if build.get(key) != verification.get(key)
  }
  if mismatches:
    raise ValueError(f"validation-prior build/verification mismatch: {mismatches}")
  local_archive = (root / str(build["archive_path"])).resolve()
  if verified_download_path == local_archive:
    raise ValueError("registration requires a separately downloaded prior archive")
  if not verified_download_path.is_file():
    raise FileNotFoundError(verified_download_path)
  if (
    file_sha256(verified_download_path) != build["archive_sha256"]
    or verified_download_path.stat().st_size != int(build["archive_byte_size"])
  ):
    raise ValueError("downloaded validation-prior archive differs from the built bundle")
  downloaded_verification = verify_validation_prior_bundle(verified_download_path)
  if (
    downloaded_verification["manifest_sha256"] != build["manifest_sha256"]
    or downloaded_verification["prior_records"] != build["prior_records"]
  ):
    raise ValueError("downloaded validation-prior archive has mismatched members")
  if cfg.get("validation_prior_archive") is not None:
    raise ValueError("source config already contains validation-prior registration")
  if cfg["outputs"]["artifact_policy"].get("immutable_archive_uri") is not None:
    raise ValueError("source config unexpectedly claims a full-result archive")

  output_cfg = copy.deepcopy(cfg)
  source_config_sha = file_sha256(source_config_path)
  output_cfg["validation_prior_archive"] = {
    "status": "registered_for_development_validation_sweep",
    "source_config_path": _relative(source_config_path, root),
    "source_config_sha256": source_config_sha,
    "uri": archive_uri,
    "version": archive_version,
    "sha256": build["archive_sha256"],
    "byte_size": int(build["archive_byte_size"]),
    "manifest_sha256": build["manifest_sha256"],
    "prior_records": int(build["prior_records"]),
    "prior_manifest_path": f"runs/{cfg['run_id']}/validation_priors/prior_manifest.csv",
    "prior_manifest_sha256": build["prior_manifest_sha256"],
    "success_marker_path": f"runs/{cfg['run_id']}/validation_priors/_SUCCESS.json",
    "evidence": {
      "build_record_path": _relative(build_record_path, root),
      "build_record_sha256": file_sha256(build_record_path),
      "verify_record_path": _relative(verify_record_path, root),
      "verify_record_sha256": file_sha256(verify_record_path),
    },
  }
  atomic_write_text(output_config_path, yaml.safe_dump(output_cfg, sort_keys=False))
  result = {
    "artifact_kind": "validation_prior_archive_registration",
    "schema_version": 1,
    "status": "registered_for_development_validation_sweep",
    "run_id": cfg["run_id"],
    "source_config_path": _relative(source_config_path, root),
    "source_config_sha256": source_config_sha,
    "registered_config_path": _relative(output_config_path, root),
    "registered_config_sha256": file_sha256(output_config_path),
    "archive_uri": archive_uri,
    "archive_version": archive_version,
    "archive_sha256": build["archive_sha256"],
    "archive_byte_size": int(build["archive_byte_size"]),
    "manifest_sha256": build["manifest_sha256"],
    "prior_records": int(build["prior_records"]),
    "verified_download_path": verified_download_path.as_posix(),
    "verified_download_sha256": file_sha256(verified_download_path),
    "full_result_archive_uri": None,
  }
  try:
    atomic_write_json(registration_record_path, result)
  except BaseException:
    output_config_path.unlink(missing_ok=True)
    raise
  return result


def validate_registered_validation_prior_configuration(
  root: Path, config_path: Path, cfg: dict | None = None,
) -> dict:
  """Verify the full local/remote identity chain required before sweeping."""
  root = root.resolve()
  config_path = config_path.resolve()
  cfg = load_yaml_strict(config_path) if cfg is None else cfg
  validate_registered_decoder_configuration(root, config_path, cfg)
  block = cfg.get("validation_prior_archive")
  if not isinstance(block, dict) or block.get("status") != (
    "registered_for_development_validation_sweep"
  ):
    raise ValueError("validation-prior archive is not registered for sweeping")
  source_config = root / str(block.get("source_config_path", ""))
  evidence = block.get("evidence", {})
  build_path = root / str(evidence.get("build_record_path", ""))
  verify_path = root / str(evidence.get("verify_record_path", ""))
  if (
    not source_config.is_file()
    or file_sha256(source_config) != block.get("source_config_sha256")
    or not build_path.is_file()
    or file_sha256(build_path) != evidence.get("build_record_sha256")
    or not verify_path.is_file()
    or file_sha256(verify_path) != evidence.get("verify_record_sha256")
  ):
    raise ValueError("validation-prior registration evidence is missing or changed")
  build = _json(build_path)
  verification = _json(verify_path)
  if any(
    build.get(key) != block.get(field)
    or verification.get(key) != block.get(field)
    for key, field in (
      ("archive_sha256", "sha256"),
      ("archive_byte_size", "byte_size"),
      ("manifest_sha256", "manifest_sha256"),
      ("prior_records", "prior_records"),
    )
  ):
    raise ValueError("validation-prior registered identity differs from archive evidence")
  prior_manifest = root / str(block.get("prior_manifest_path", ""))
  success_marker = root / str(block.get("success_marker_path", ""))
  prior_record = verify_csv_artifact(prior_manifest)
  verify_success_marker(success_marker.parent)
  if (
    prior_record.get("schema_name") != "medsam2_prior_manifest_v1"
    or prior_record.get("split") != "validation"
    or int(prior_record.get("row_count", -1)) != int(block["prior_records"])
    or file_sha256(prior_manifest) != block.get("prior_manifest_sha256")
    or not block.get("uri")
    or not block.get("version")
  ):
    raise ValueError("local validation-prior completion differs from registration")
  return {
    "status": "complete",
    "registered_config_sha256": file_sha256(config_path),
    "source_config_sha256": block["source_config_sha256"],
    "archive_sha256": block["sha256"],
    "manifest_sha256": block["manifest_sha256"],
    "prior_manifest_sha256": block["prior_manifest_sha256"],
    "prior_records": int(block["prior_records"]),
  }
