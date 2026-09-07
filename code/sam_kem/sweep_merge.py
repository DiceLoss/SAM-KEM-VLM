"""Transactional merge gate for audited validation-sweep slice shards."""

from __future__ import annotations

import csv
import json
import os
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from .artifacts import atomic_write_json, file_sha256
from .config import load_contract, require_development_execution
from .prior_registration import validate_registered_validation_prior_configuration
from .protocols import load_protocol
from .sweep_evaluation import SWEEP_FIELDS
from .sweep_sharded import PRIMARY_KEY, _paths, _shard_set_sha256


REVIEWED_SHARD_AUDIT_SHA256 = (
  "25e83bd7b9097840cc2759075388225e70fec0d83d7c3b294a4b1c525f708079"
)


def _json(path: Path) -> dict:
  value = json.loads(path.read_text(encoding="utf-8"))
  if not isinstance(value, dict):
    raise ValueError(f"expected a JSON object: {path}")
  return value


def _clean_commit(root: Path) -> str:
  status = subprocess.run(
    ["git", "status", "--porcelain"], cwd=root, check=True,
    capture_output=True, text=True,
  ).stdout.strip()
  if status:
    raise RuntimeError(f"sweep merge requires a clean worktree: {status.splitlines()}")
  return subprocess.run(
    ["git", "rev-parse", "HEAD"], cwd=root, check=True,
    capture_output=True, text=True,
  ).stdout.strip()


def stream_merge_csv_fragments(fragment_paths: list[Path], output_path: Path) -> dict:
  """Copy audited CSV bodies in order, requiring one byte-identical header."""
  if not fragment_paths:
    raise ValueError("cannot merge an empty fragment list")
  if output_path.exists():
    raise FileExistsError(f"refusing to overwrite merged sweep: {output_path}")
  output_path.parent.mkdir(parents=True, exist_ok=True)
  expected_header = None
  row_count = 0
  with output_path.open("xb") as destination:
    for fragment_path in fragment_paths:
      with fragment_path.open("rb") as source:
        header = source.readline()
        if expected_header is None:
          expected_header = header
          destination.write(header)
        elif header != expected_header:
          raise ValueError(f"shard CSV header differs: {fragment_path}")
        while chunk := source.read(1024 * 1024):
          row_count += chunk.count(b"\n")
          destination.write(chunk)
    destination.flush()
    os.fsync(destination.fileno())
  if expected_header is None:
    raise ValueError("merged sweep header is absent")
  columns = next(csv.reader([expected_header.decode("utf-8").rstrip("\r\n")]))
  if columns != SWEEP_FIELDS:
    raise ValueError(f"merged sweep has unexpected columns: {columns}")
  return {"row_count": row_count, "columns": columns}


def _current_shard_identities(shard_dir: Path, shards: list[dict]) -> list[dict]:
  identities = []
  for shard in shards:
    fragment, sidecar, marker = _paths(shard_dir, shard)
    if not all(path.is_file() for path in (fragment, sidecar, marker)):
      raise FileNotFoundError(f"audited shard artifact is missing: {shard['shard_id']}")
    identities.append({
      "shard_id": shard["shard_id"],
      "fragment_sha256": file_sha256(fragment),
      "fragment_manifest_sha256": file_sha256(sidecar),
      "success_marker_sha256": file_sha256(marker),
    })
  return identities


def _verify_transaction_inventory(shard_dir: Path, audit: dict) -> int:
  block = audit.get("temporary_transaction_directories", {})
  if block.get("disposition") != (
    "preserved_read_only_excluded_from_published_shard_set"
  ):
    raise ValueError("shard audit does not preserve interrupted transaction evidence")
  entries = block.get("entries", [])
  for entry in entries:
    directory = shard_dir / entry["path"]
    expected = {
      item["path"]: (int(item["byte_size"]), item["sha256"])
      for item in entry["files"]
    }
    actual_paths = [path for path in directory.rglob("*") if path.is_file()]
    actual = {
      path.relative_to(directory).as_posix(): (path.stat().st_size, file_sha256(path))
      for path in actual_paths
    }
    if actual != expected:
      raise ValueError(f"interrupted transaction evidence changed: {entry['path']}")
  return len(entries)


def merge_audited_validation_sweep(
  root: Path, config_path: Path, protocol_path: Path, plan_path: Path,
  shard_dir: Path, shard_audit_path: Path, output_dir: Path, audit_path: Path,
) -> dict:
  """Merge every reviewed shard without selecting or freezing hyperparameters."""
  root = root.resolve()
  config_path = config_path.resolve()
  protocol_path = protocol_path.resolve()
  plan_path = plan_path.resolve()
  shard_dir = shard_dir.resolve()
  shard_audit_path = shard_audit_path.resolve()
  output_dir = output_dir.resolve()
  audit_path = audit_path.resolve()
  if output_dir.exists() or audit_path.exists():
    raise FileExistsError("refusing to overwrite merge output or audit")
  merge_commit = _clean_commit(root)
  if file_sha256(shard_audit_path) != REVIEWED_SHARD_AUDIT_SHA256:
    raise ValueError("shard audit is not the researcher-reviewed artifact")
  plan = _json(plan_path)
  audit = _json(shard_audit_path)
  cfg, _ = load_contract(config_path)
  require_development_execution(cfg)
  registration = validate_registered_validation_prior_configuration(root, config_path, cfg)
  protocol = load_protocol(protocol_path)
  frozen_path = root / str(cfg["outputs"]["frozen_hyperparameters"])
  if (
    audit.get("status") != "PASS_complete_pending_merge"
    or audit.get("read_only") is not True
    or audit.get("plan_sha256") != file_sha256(plan_path)
    or audit.get("config_sha256") != file_sha256(config_path)
    or audit.get("protocol_sha256") != file_sha256(protocol_path)
    or audit.get("manifest_sha256") != plan.get("manifest_sha256")
    or audit.get("shards_total") != 741
    or audit.get("total_cases") != 8_892
    or audit.get("total_result_rows") != 5_157_360
    or audit.get("total_operator_updates") != 41_347_800
    or audit.get("published_file_counts", {}).get("shard_artifacts") != 2_223
    or audit.get("merge_created") is not False
    or audit.get("frozen_hyperparameters_created") is not False
    or audit.get("test_locked_access") != "prohibited"
    or plan.get("prior_registration") != registration
    or protocol.kind != "hyperparameter_sweep"
    or protocol.split != "validation"
    or output_dir != root / protocol.output
    or frozen_path.exists()
  ):
    raise ValueError("reviewed shard audit or merge inputs are inconsistent")
  shards = plan.get("shards", [])
  if len(shards) != 741:
    raise ValueError("frozen plan does not contain 741 shards")
  identities = _current_shard_identities(shard_dir, shards)
  shard_set_sha256 = _shard_set_sha256(identities)
  if shard_set_sha256 != audit.get("shard_set_sha256"):
    raise ValueError("current shard bytes differ from the reviewed shard set")
  completion_path = shard_dir / "shard_batch_SUCCESS.json"
  if file_sha256(completion_path) != audit.get("shard_batch_success_sha256"):
    raise ValueError("shard batch completion marker changed after audit")
  transaction_count = _verify_transaction_inventory(shard_dir, audit)

  output_dir.parent.mkdir(parents=True, exist_ok=True)
  transaction_dir = Path(tempfile.mkdtemp(
    prefix=".hyperparameter_sweep.merge.", dir=output_dir.parent,
  ))
  merged_path = transaction_dir / "validation_sweep.csv"
  fragment_paths = [_paths(shard_dir, shard)[0] for shard in shards]
  merge_details = stream_merge_csv_fragments(fragment_paths, merged_path)
  if int(merge_details["row_count"]) != 5_157_360:
    raise ValueError(
      f"merged sweep has {merge_details['row_count']} rows, expected 5157360"
    )
  merged_sha256 = file_sha256(merged_path)
  manifest_path = merged_path.with_suffix(merged_path.suffix + ".manifest.json")
  atomic_write_json(manifest_path, {
    "schema_version": 1, "schema_name": "validation_hyperparameter_sweep_v3",
    "artifact_path": merged_path.name, "artifact_sha256": merged_sha256,
    "byte_size": merged_path.stat().st_size, "row_count": 5_157_360,
    "columns": SWEEP_FIELDS, "primary_key": PRIMARY_KEY, "split": "validation",
    "source_shard_count": 741, "source_shard_set_sha256": shard_set_sha256,
    "completed_utc": datetime.now(timezone.utc).isoformat(),
  })
  manifest_sha256 = file_sha256(manifest_path)
  record_path = transaction_dir / "merge_record.json"
  merge_record = {
    "artifact_kind": "validation_sweep_shard_merge", "schema_version": 1,
    "status": "complete_pending_hyperparameter_freeze",
    "merge_git_commit": merge_commit,
    "execution_git_commit": audit["execution_git_commit"],
    "shard_audit_path": shard_audit_path.relative_to(root).as_posix(),
    "shard_audit_sha256": REVIEWED_SHARD_AUDIT_SHA256,
    "plan_sha256": file_sha256(plan_path), "shard_set_sha256": shard_set_sha256,
    "source_shards": 741, "source_cases": 8_892,
    "merged_result_rows": 5_157_360,
    "merged_csv_sha256": merged_sha256,
    "merged_csv_byte_size": merged_path.stat().st_size,
    "merged_manifest_sha256": manifest_sha256,
    "method_row_counts": audit["method_row_counts"],
    "global_primary_key_uniqueness": audit["global_primary_key_uniqueness"],
    "temporary_transaction_directories_preserved": transaction_count,
    "merge_order": "frozen_shard_id_order_then_existing_fragment_row_order",
    "frozen_hyperparameters_created": False, "test_locked_access": "prohibited",
  }
  atomic_write_json(record_path, merge_record)
  record_sha256 = file_sha256(record_path)
  success_path = transaction_dir / "merge_SUCCESS.json"
  atomic_write_json(success_path, {
    "artifact_kind": "validation_sweep_merge_success", "schema_version": 1,
    "status": "complete_pending_hyperparameter_freeze",
    "merged_csv_sha256": merged_sha256,
    "merged_manifest_sha256": manifest_sha256,
    "merge_record_sha256": record_sha256,
    "shard_set_sha256": shard_set_sha256, "row_count": 5_157_360,
    "frozen_hyperparameters_created": False, "test_locked_access": "prohibited",
  })
  success_sha256 = file_sha256(success_path)
  os.replace(transaction_dir, output_dir)
  published_merged = output_dir / merged_path.name
  published_manifest = output_dir / manifest_path.name
  published_record = output_dir / record_path.name
  published_success = output_dir / success_path.name
  if (
    file_sha256(published_merged) != merged_sha256
    or file_sha256(published_manifest) != manifest_sha256
    or file_sha256(published_record) != record_sha256
    or file_sha256(published_success) != success_sha256
    or frozen_path.exists()
  ):
    raise ValueError("published merge changed during transactional rename")
  result = {
    **merge_record,
    "status": "PASS_complete_pending_hyperparameter_freeze",
    "merged_csv_path": published_merged.relative_to(root).as_posix(),
    "merged_manifest_path": published_manifest.relative_to(root).as_posix(),
    "merge_record_path": published_record.relative_to(root).as_posix(),
    "merge_success_path": published_success.relative_to(root).as_posix(),
    "merge_success_sha256": success_sha256,
    "formal_merge_artifact_count": 4,
  }
  atomic_write_json(audit_path, result)
  return result
