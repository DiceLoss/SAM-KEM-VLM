"""Create a provenance-preserving continuation copy of an interrupted Qwen batch."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from pathlib import Path


def _sha256(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as handle:
    for chunk in iter(lambda: handle.read(1024 * 1024), b""):
      digest.update(chunk)
  return digest.hexdigest()


def _paths(directory: Path) -> list[Path]:
  return sorted(
    path for path in directory.rglob("c*.json")
    if path.name != "_SUCCESS.json"
  )


def _set_sha256(directory: Path, paths: list[Path]) -> str:
  digest = hashlib.sha256()
  for path in paths:
    relative = path.relative_to(directory).as_posix()
    digest.update(f"{relative}\0{_sha256(path)}\n".encode())
  return digest.hexdigest()


def _atomic_json(path: Path, value: dict) -> None:
  temporary = path.with_suffix(path.suffix + ".tmp")
  temporary.write_text(
    json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8",
  )
  temporary.replace(path)


def migrate(
  root: Path, manifest_path: Path, source_dir: Path, destination_dir: Path,
  resolution_path: Path, output_audit: Path,
) -> dict:
  if destination_dir.exists():
    raise FileExistsError(f"destination must not exist: {destination_dir}")
  if output_audit.exists():
    raise FileExistsError(f"migration audit already exists: {output_audit}")
  resolution = json.loads(resolution_path.read_text(encoding="utf-8"))
  forensic = root / resolution["forensic_audit"]["path"]
  if _sha256(forensic) != resolution["forensic_audit"]["sha256"]:
    raise ValueError("forensic audit SHA-256 differs from the resolution")
  with manifest_path.open("r", encoding="utf-8", newline="") as handle:
    expected = {
      (row["dataset"], row["subject_id"], row["slice_id"], int(row["count_level"]))
      for row in csv.DictReader(handle)
    }
  source_paths = _paths(source_dir)
  expected_count = int(resolution["source_response_set"]["records"])
  if len(source_paths) != expected_count:
    raise ValueError(f"source record count differs: {len(source_paths)} != {expected_count}")
  source_set_sha = _set_sha256(source_dir, source_paths)
  if source_set_sha != resolution["source_response_set"]["sha256"]:
    raise ValueError("source response-set SHA-256 differs from the forensic audit")
  incident_relative = str(resolution["incident"]["relative_path"])
  incident_seen = 0
  original_incident_sha = ""
  for path in source_paths:
    relative = path.relative_to(source_dir).as_posix()
    document = json.loads(path.read_text(encoding="utf-8"))
    record = document.get("record", {})
    key = (
      str(record.get("dataset", "")), str(record.get("subject_id", "")),
      str(record.get("slice_id", "")), int(record.get("count_level", -1)),
    )
    if key not in expected:
      raise ValueError(f"response key is absent from the prompt manifest: {relative}")
    if document.get("request_policy", {}).get("sha256") != resolution["request_policy_sha256"]:
      raise ValueError(f"wrong request policy: {relative}")
    if (
      document.get("request_implementation", {}).get("sha256")
      != resolution["request_implementation_sha256"]
    ):
      raise ValueError(f"wrong request implementation: {relative}")
    source = record.get("source")
    if source == "qwen_batch_configuration_failure" and relative != incident_relative:
      raise ValueError(f"unresolved batch-configuration failure: {relative}")
    if relative == incident_relative:
      incident_seen += 1
      original_incident_sha = _sha256(path)
      expected_key = tuple(resolution["incident"]["cache_key"])
      statuses = [
        int(attempt["http_status"]) for attempt in document.get("attempts", [])
        if attempt.get("http_status") is not None
      ]
      error_text = json.dumps(document, ensure_ascii=False).lower()
      if (
        key != expected_key
        or source != resolution["incident"]["expected_original_source"]
        or statuses != [int(resolution["incident"]["expected_http_status"])]
        or str(resolution["incident"]["expected_provider_error_code"]).lower()
        not in error_text
      ):
        raise ValueError("incident record differs from the approved resolution")
  if incident_seen != 1:
    raise ValueError(f"expected exactly one incident record, found {incident_seen}")

  destination_dir.mkdir(parents=True)
  copied = 0
  transformed = 0
  transformed_sha = ""
  resolution_sha = _sha256(resolution_path)
  for source_path in source_paths:
    relative = source_path.relative_to(source_dir)
    destination = destination_dir / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    if relative.as_posix() != incident_relative:
      shutil.copy2(source_path, destination)
      if _sha256(source_path) != _sha256(destination):
        raise RuntimeError(f"byte-copy verification failed: {relative.as_posix()}")
      copied += 1
      continue
    document = json.loads(source_path.read_text(encoding="utf-8"))
    document["record"]["source"] = resolution["incident"]["resolved_source"]
    document["incident_resolution"] = {
      "resolution_id": resolution["resolution_id"],
      "resolution_manifest_path": resolution_path.relative_to(root).as_posix(),
      "resolution_manifest_sha256": resolution_sha,
      "original_relative_path": incident_relative,
      "original_file_sha256": original_incident_sha,
      "original_response_set_sha256": source_set_sha,
      "original_source": resolution["incident"]["expected_original_source"],
      "resolved_source": resolution["incident"]["resolved_source"],
      "additional_api_attempt": False,
    }
    _atomic_json(destination, document)
    transformed_sha = _sha256(destination)
    transformed += 1
  destination_paths = _paths(destination_dir)
  if len(destination_paths) != expected_count or (destination_dir / "_SUCCESS.json").exists():
    raise RuntimeError("continuation directory has the wrong pre-resume state")
  audit = {
    "schema_version": 1,
    "artifact_kind": "qwen_partial_batch_migration_audit",
    "source_directory": str(source_dir.resolve()),
    "destination_directory": str(destination_dir.resolve()),
    "source_records": len(source_paths),
    "source_response_set_sha256": source_set_sha,
    "destination_records": len(destination_paths),
    "destination_response_set_sha256": _set_sha256(destination_dir, destination_paths),
    "byte_identical_copies": copied,
    "transformed_records": transformed,
    "incident_original_sha256": original_incident_sha,
    "incident_transformed_sha256": transformed_sha,
    "resolution_manifest_sha256": resolution_sha,
    "success_marker_present": False,
    "remaining_unprocessed_keys": len(expected) - len(destination_paths),
  }
  output_audit.parent.mkdir(parents=True, exist_ok=True)
  _atomic_json(output_audit, audit)
  return audit


def main() -> int:
  parser = argparse.ArgumentParser()
  parser.add_argument("--root", type=Path, default=Path("."))
  parser.add_argument("--manifest", type=Path, required=True)
  parser.add_argument("--source-responses", type=Path, required=True)
  parser.add_argument("--destination-responses", type=Path, required=True)
  parser.add_argument("--resolution", type=Path, required=True)
  parser.add_argument("--output-audit", type=Path, required=True)
  args = parser.parse_args()
  audit = migrate(
    args.root.resolve(), args.manifest.resolve(), args.source_responses.resolve(),
    args.destination_responses.resolve(), args.resolution.resolve(),
    args.output_audit.resolve(),
  )
  print(json.dumps(audit, indent=2, sort_keys=True))
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
