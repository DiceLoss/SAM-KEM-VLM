"""Read-only forensic audit for an interrupted Qwen response directory."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import shutil
import sys
from collections import Counter
from pathlib import Path


def _sha256(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as handle:
    for chunk in iter(lambda: handle.read(1024 * 1024), b""):
      digest.update(chunk)
  return digest.hexdigest()


def _module_fingerprint(name: str) -> dict:
  spec = importlib.util.find_spec(name)
  origin = Path(spec.origin).resolve() if spec is not None and spec.origin else None
  return {
    "module": name,
    "path": None if origin is None else str(origin),
    "sha256": None if origin is None or not origin.is_file() else _sha256(origin),
  }


def _error_class(attempt: dict) -> str:
  value = str(attempt.get("error", ""))
  return value.split("(", 1)[0] if value else ""


def audit(root: Path, manifest_path: Path, response_dir: Path) -> dict:
  with manifest_path.open("r", encoding="utf-8", newline="") as handle:
    manifest = list(csv.DictReader(handle))
  expected = {
    (row["dataset"], row["subject_id"], row["slice_id"], int(row["count_level"]))
    for row in manifest
  }
  paths = sorted(
    path for path in response_dir.rglob("c*.json")
    if path.name != "_SUCCESS.json"
  )
  source_counts: Counter[str] = Counter()
  attempt_counts: Counter[int] = Counter()
  http_counts: Counter[int] = Counter()
  implementation_counts: Counter[str] = Counter()
  missing_success_http = 0
  missing_implementation = 0
  missing_image_sha = 0
  duplicate_keys = 0
  unknown_keys = []
  keys = set()
  request_ids = []
  failures = []
  set_digest = hashlib.sha256()
  for path in paths:
    relative = path.relative_to(response_dir).as_posix()
    file_sha = _sha256(path)
    set_digest.update(f"{relative}\0{file_sha}\n".encode())
    document = json.loads(path.read_text(encoding="utf-8"))
    record = document.get("record", {})
    key = (
      str(record.get("dataset", "")), str(record.get("subject_id", "")),
      str(record.get("slice_id", "")), int(record.get("count_level", -1)),
    )
    if key in keys:
      duplicate_keys += 1
    keys.add(key)
    if key not in expected:
      unknown_keys.append(relative)
    source = str(record.get("source", ""))
    source_counts[source] += 1
    attempt_counts[int(record.get("attempt_count", 0))] += 1
    if not record.get("image_sha256"):
      missing_image_sha += 1
    request_id = str(record.get("request_id", ""))
    if request_id:
      request_ids.append(request_id)
    implementation = document.get("request_implementation", {}).get("sha256")
    if implementation:
      implementation_counts[str(implementation)] += 1
    else:
      missing_implementation += 1
    attempts = document.get("attempts", [])
    for attempt in attempts:
      status = attempt.get("http_status")
      if status is not None:
        http_counts[int(status)] += 1
      if attempt.get("status") == "success" and status is None:
        missing_success_http += 1
    if source not in {"qwen3-vl-plus", "qwen_no_detection"}:
      failures.append({
        "path": relative,
        "source": source,
        "attempt_count": int(record.get("attempt_count", 0)),
        "http_statuses": [
          int(attempt["http_status"]) for attempt in attempts
          if attempt.get("http_status") is not None
        ],
        "error_classes": [_error_class(attempt) for attempt in attempts],
      })
  repo_modules = []
  for relative in (
    "code/sam_kem/qwen.py", "code/sam_kem/prompt_batch.py", "code/sam_kem/cli.py",
  ):
    path = root / relative
    repo_modules.append({
      "path": relative,
      "sha256": _sha256(path) if path.is_file() else None,
    })
  return {
    "schema_version": 1,
    "artifact_kind": "qwen_partial_batch_forensic_audit",
    "read_only": True,
    "runtime": {
      "python_executable": sys.executable,
      "sam_kem_entrypoint": shutil.which("sam-kem"),
      "imported_modules": [
        _module_fingerprint("sam_kem.qwen"),
        _module_fingerprint("sam_kem.prompt_batch"),
        _module_fingerprint("sam_kem.cli"),
      ],
      "repository_modules": repo_modules,
    },
    "responses": {
      "directory": str(response_dir.resolve()),
      "records": len(paths),
      "expected_records": len(expected),
      "missing_expected_keys": len(expected - keys),
      "unknown_key_paths": unknown_keys,
      "duplicate_keys": duplicate_keys,
      "source_counts": dict(sorted(source_counts.items())),
      "attempt_count_distribution": {
        str(key): value for key, value in sorted(attempt_counts.items())
      },
      "explicit_http_status_distribution": {
        str(key): value for key, value in sorted(http_counts.items())
      },
      "successful_attempts_missing_http_status": missing_success_http,
      "request_implementation_sha256_counts": dict(sorted(implementation_counts.items())),
      "records_missing_request_implementation": missing_implementation,
      "records_missing_image_sha256": missing_image_sha,
      "nonempty_request_ids": len(request_ids),
      "unique_nonempty_request_ids": len(set(request_ids)),
      "response_set_sha256": set_digest.hexdigest(),
      "failure_records": failures,
      "success_marker_present": (response_dir / "_SUCCESS.json").is_file(),
    },
  }


def main() -> int:
  parser = argparse.ArgumentParser()
  parser.add_argument("--root", type=Path, default=Path("."))
  parser.add_argument("--manifest", type=Path, required=True)
  parser.add_argument("--responses", type=Path, required=True)
  parser.add_argument("--output", type=Path, required=True)
  args = parser.parse_args()
  report = audit(args.root.resolve(), args.manifest.resolve(), args.responses.resolve())
  args.output.parent.mkdir(parents=True, exist_ok=True)
  args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
  print(args.output)
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
