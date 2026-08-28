"""Register immutable decoder archives without mutating the training configuration."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from urllib.parse import urlparse

import yaml

from .artifacts import atomic_write_json, atomic_write_text, file_sha256
from .config import load_yaml_strict


def _json(path: Path) -> dict:
  value = json.loads(path.read_text(encoding="utf-8"))
  if not isinstance(value, dict):
    raise ValueError(f"expected a JSON object: {path}")
  return value


def _relative(path: Path, root: Path) -> str:
  try:
    return path.resolve().relative_to(root.resolve()).as_posix()
  except ValueError:
    return path.resolve().as_posix()


def _validate_archive_uri(uri: str, version: str) -> None:
  parsed = urlparse(uri)
  if parsed.scheme.lower() in ("", "file"):
    raise ValueError("immutable archive URI must not be a local or relative path")
  if parsed.scheme.lower() in ("http", "https") and not parsed.netloc:
    raise ValueError("HTTP archive URI must include a host")
  if not version.strip():
    raise ValueError("an explicit immutable archive version/object identifier is required")


def register_decoder_archive(
  root: Path,
  training_config_path: Path,
  build_record_path: Path,
  verify_record_path: Path,
  verified_download_path: Path,
  *,
  archive_uri: str,
  archive_version: str,
  output_config_path: Path,
  registration_record_path: Path,
) -> dict:
  """Create a downstream config bound to a separately downloaded immutable archive."""
  root = root.resolve()
  training_config_path = training_config_path.resolve()
  build_record_path = build_record_path.resolve()
  verify_record_path = verify_record_path.resolve()
  verified_download_path = verified_download_path.resolve()
  output_config_path = output_config_path.resolve()
  registration_record_path = registration_record_path.resolve()
  if output_config_path.exists() or registration_record_path.exists():
    raise FileExistsError("refusing to overwrite checkpoint registration artifacts")
  _validate_archive_uri(archive_uri, archive_version)
  training_cfg = load_yaml_strict(training_config_path)
  build = _json(build_record_path)
  verification = _json(verify_record_path)
  if build.get("status") != "local_bundle_complete_pending_immutable_upload":
    raise ValueError("decoder archive build record is not ready for registration")
  if verification.get("status") != "complete":
    raise ValueError("decoder archive verification record is incomplete")
  shared_fields = ("archive_sha256", "archive_byte_size", "manifest_sha256")
  mismatches = {
    key: (build.get(key), verification.get(key))
    for key in shared_fields if build.get(key) != verification.get(key)
  }
  if mismatches:
    raise ValueError(f"decoder archive build/verification mismatch: {mismatches}")
  local_archive = (root / str(build["archive_path"])).resolve()
  if verified_download_path == local_archive:
    raise ValueError("registration requires a separately downloaded archive copy")
  if not verified_download_path.is_file():
    raise FileNotFoundError(f"verified archive download missing: {verified_download_path}")
  if (
    file_sha256(verified_download_path) != build["archive_sha256"]
    or verified_download_path.stat().st_size != int(build["archive_byte_size"])
  ):
    raise ValueError("downloaded immutable archive differs from the local verified bundle")

  cfg = copy.deepcopy(training_cfg)
  training_config_sha = file_sha256(training_config_path)
  checkpoint_sha = {
    dataset: str(build["datasets"][dataset]["checkpoint_sha256"])
    for dataset in ("brainweb", "hecktor")
  }
  checkpoint_cfg = cfg["medsam2"]
  checkpoint_cfg["training_config_path"] = _relative(training_config_path, root)
  checkpoint_cfg["training_config_sha256"] = training_config_sha
  checkpoint_cfg["checkpoint_sha256"] = checkpoint_sha
  checkpoint_cfg["checkpoint_archive_uri"] = {
    dataset: archive_uri for dataset in ("brainweb", "hecktor")
  }
  checkpoint_cfg["checkpoint_archive_version"] = archive_version
  checkpoint_cfg["checkpoint_archive_sha256"] = build["archive_sha256"]
  checkpoint_cfg["checkpoint_archive_manifest_sha256"] = build["manifest_sha256"]
  checkpoint_cfg["checkpoint_archive_evidence"] = {
    "build_record_path": _relative(build_record_path, root),
    "build_record_sha256": file_sha256(build_record_path),
    "verify_record_path": _relative(verify_record_path, root),
    "verify_record_sha256": file_sha256(verify_record_path),
  }
  # This checkpoint-only archive is not the later full result archive.
  if cfg["outputs"]["artifact_policy"].get("immutable_archive_uri") is not None:
    raise ValueError("training config unexpectedly claims a full immutable result archive")
  atomic_write_text(output_config_path, yaml.safe_dump(cfg, sort_keys=False))
  result = {
    "artifact_kind": "decoder_checkpoint_archive_registration",
    "schema_version": 1,
    "status": "registered_for_development_validation",
    "run_id": cfg["run_id"],
    "training_config_path": _relative(training_config_path, root),
    "training_config_sha256": training_config_sha,
    "registered_config_path": _relative(output_config_path, root),
    "registered_config_sha256": file_sha256(output_config_path),
    "archive_uri": archive_uri,
    "archive_version": archive_version,
    "archive_sha256": build["archive_sha256"],
    "archive_byte_size": build["archive_byte_size"],
    "manifest_sha256": build["manifest_sha256"],
    "verified_download_path": verified_download_path.as_posix(),
    "verified_download_sha256": file_sha256(verified_download_path),
    "checkpoint_sha256": checkpoint_sha,
    "full_result_archive_uri": None,
  }
  try:
    atomic_write_json(registration_record_path, result)
  except BaseException:
    output_config_path.unlink(missing_ok=True)
    raise
  return result


def validate_registered_decoder_configuration(
  root: Path, config_path: Path, cfg: dict | None = None,
) -> dict:
  """Verify the registered config, original training config, records and local checkpoints."""
  root = root.resolve()
  config_path = config_path.resolve()
  cfg = load_yaml_strict(config_path) if cfg is None else cfg
  checkpoint_cfg = cfg["medsam2"]
  training_config = root / str(checkpoint_cfg.get("training_config_path", ""))
  training_sha = checkpoint_cfg.get("training_config_sha256")
  if not training_config.is_file() or file_sha256(training_config) != training_sha:
    raise ValueError("registered decoder training configuration is missing or changed")
  evidence = checkpoint_cfg.get("checkpoint_archive_evidence", {})
  build_path = root / str(evidence.get("build_record_path", ""))
  verify_path = root / str(evidence.get("verify_record_path", ""))
  if (
    not build_path.is_file() or file_sha256(build_path) != evidence.get("build_record_sha256")
    or not verify_path.is_file() or file_sha256(verify_path) != evidence.get("verify_record_sha256")
  ):
    raise ValueError("registered decoder archive evidence is missing or changed")
  build = _json(build_path)
  verification = _json(verify_path)
  if (
    build.get("archive_sha256") != checkpoint_cfg.get("checkpoint_archive_sha256")
    or verification.get("archive_sha256") != checkpoint_cfg.get("checkpoint_archive_sha256")
    or build.get("manifest_sha256") != checkpoint_cfg.get("checkpoint_archive_manifest_sha256")
    or verification.get("manifest_sha256") != checkpoint_cfg.get("checkpoint_archive_manifest_sha256")
  ):
    raise ValueError("registered decoder archive identity differs from its evidence")
  for dataset in ("brainweb", "hecktor"):
    checkpoint = root / str(checkpoint_cfg["checkpoint_paths"][dataset])
    provenance = _json(checkpoint.with_suffix(".json"))
    expected_sha = checkpoint_cfg["checkpoint_sha256"][dataset]
    if (
      not checkpoint.is_file() or file_sha256(checkpoint) != expected_sha
      or provenance.get("checkpoint_sha256") != expected_sha
      or provenance.get("config_sha256") != training_sha
      or not checkpoint_cfg.get("checkpoint_archive_uri", {}).get(dataset)
    ):
      raise ValueError(f"registered decoder checkpoint identity mismatch: {dataset}")
  return {
    "status": "complete",
    "registered_config_sha256": file_sha256(config_path),
    "training_config_sha256": training_sha,
    "archive_sha256": checkpoint_cfg["checkpoint_archive_sha256"],
    "manifest_sha256": checkpoint_cfg["checkpoint_archive_manifest_sha256"],
  }
