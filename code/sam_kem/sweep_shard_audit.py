"""Read-only integrity audit for a completed formal validation shard batch."""

from __future__ import annotations

import json
import subprocess
from collections import Counter
from datetime import datetime
from pathlib import Path

from .artifacts import atomic_write_json, file_sha256
from .config import load_contract, require_development_execution
from .experiments import load_case_manifest, validate_case_contract
from .prior_registration import validate_registered_validation_prior_configuration
from .protocols import load_protocol
from .sweep_sharded import (
  _paths,
  _shard_set_sha256,
  line_ending_equivalent_sha256,
  verify_sweep_shard,
)
from .sweep_sharding import group_validation_cases


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
    raise RuntimeError(f"shard audit requires a clean worktree: {status.splitlines()}")
  return subprocess.run(
    ["git", "rev-parse", "HEAD"], cwd=root, check=True,
    capture_output=True, text=True,
  ).stdout.strip()


def _is_ancestor(root: Path, ancestor: str, descendant: str) -> bool:
  return subprocess.run(
    ["git", "merge-base", "--is-ancestor", ancestor, descendant], cwd=root,
    capture_output=True, text=True,
  ).returncode == 0


def audit_validation_sweep_shards(
  root: Path, config_path: Path, protocol_path: Path, plan_path: Path,
  output_dir: Path, audit_path: Path,
) -> dict:
  """Verify all published shards and inventory, without changing the run namespace."""
  root = root.resolve()
  config_path = config_path.resolve()
  protocol_path = protocol_path.resolve()
  plan_path = plan_path.resolve()
  output_dir = output_dir.resolve()
  audit_path = audit_path.resolve()
  if audit_path.exists():
    raise FileExistsError(f"refusing to overwrite shard integrity audit: {audit_path}")
  audit_commit = _clean_commit(root)
  if not output_dir.is_dir():
    raise FileNotFoundError(f"formal shard namespace is absent: {output_dir}")
  plan = _json(plan_path)
  plan_sha256 = file_sha256(plan_path)
  cfg, contract = load_contract(config_path)
  require_development_execution(cfg)
  registration = validate_registered_validation_prior_configuration(root, config_path, cfg)
  protocol = load_protocol(protocol_path)
  manifest_path = root / str(cfg["outputs"]["validation_manifest"])
  if (
    plan.get("status") != "ready_for_formal_sharded_sweep_pending_researcher_release"
    or plan.get("shard_count") != 741
    or plan.get("total_cases") != 8_892
    or plan.get("total_result_rows") != 5_157_360
    or plan.get("total_operator_updates") != 41_347_800
    or file_sha256(config_path) != plan.get("config_sha256")
    or not line_ending_equivalent_sha256(protocol_path, str(plan.get("protocol_sha256")))
    or file_sha256(manifest_path) != plan.get("manifest_sha256")
    or registration != plan.get("prior_registration")
    or protocol.kind != "hyperparameter_sweep"
    or protocol.split != "validation"
    or output_dir.relative_to(root).as_posix() != plan.get("formal_output_namespace")
  ):
    raise ValueError("completed shard inputs differ from the frozen plan")
  cases = load_case_manifest(root, manifest_path, split="validation")
  validate_case_contract(cases, contract)
  shards = group_validation_cases(cases)
  if shards != plan.get("shards"):
    raise ValueError("completed shard membership differs from the frozen plan")

  execution_path = output_dir / "execution_record.json"
  progress_path = output_dir / "progress.json"
  completion_path = output_dir / "shard_batch_SUCCESS.json"
  execution = _json(execution_path)
  progress = _json(progress_path)
  completion = _json(completion_path)
  execution_sha256 = file_sha256(execution_path)
  execution_commit = str(execution.get("execution_git_commit", ""))
  if (
    execution.get("artifact_kind") != "validation_sweep_shard_execution"
    or execution.get("status") != "running"
    or execution.get("plan_sha256") != plan_sha256
    or execution.get("config_sha256") != file_sha256(config_path)
    or execution.get("protocol_sha256") != file_sha256(protocol_path)
    or execution.get("manifest_sha256") != file_sha256(manifest_path)
    or execution.get("execution_backend")
    != "scipy_sparse_csr_cpu_float64_multiprocess"
    or execution.get("worker_processes") != 8
    or execution.get("gpu_used") is not False
    or any(execution.get("thread_environment", {}).get(name) is not None for name in (
      "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
    ))
    or execution.get("shard_count") != 741
    or execution.get("total_cases") != 8_892
    or execution.get("total_result_rows") != 5_157_360
    or execution.get("test_locked_access") != "prohibited"
    or not _is_ancestor(root, execution_commit, audit_commit)
  ):
    raise ValueError("formal execution record failed identity verification")
  if (
    progress.get("status") != "complete"
    or progress.get("shards_total") != 741
    or progress.get("shards_completed") != 741
    or int(progress.get("shards_reused_verified", -1))
    + int(progress.get("shards_written", -1)) != 741
    or progress.get("failed_shard") is not None
    or completion.get("status") != "complete_pending_merge_integrity_gate"
    or completion.get("execution_git_commit") != execution_commit
    or completion.get("execution_record_sha256") != execution_sha256
    or completion.get("plan_sha256") != plan_sha256
    or completion.get("worker_processes") != 8
    or completion.get("shards_total") != 741
    or completion.get("shards_completed") != 741
    or completion.get("total_cases") != 8_892
    or completion.get("total_result_rows") != 5_157_360
    or completion.get("merge_created") is not False
    or completion.get("frozen_hyperparameters_created") is not False
    or completion.get("test_locked_access") != "prohibited"
  ):
    raise ValueError("progress or shard batch completion marker is inconsistent")

  verified = []
  for shard in shards:
    item = verify_sweep_shard(
      output_dir, shard, execution_commit=execution_commit,
      execution_record_sha256=execution_sha256,
      plan_sha256=plan_sha256, run_id=contract.run_id,
    )
    if item is None:
      raise ValueError(f"published shard is absent: {shard['shard_id']}")
    verified.append(item)
  shard_set_sha256 = _shard_set_sha256(verified)
  if shard_set_sha256 != completion.get("shard_set_sha256"):
    raise ValueError("verified shard set differs from the batch completion marker")

  expected_files = {execution_path, progress_path, completion_path}
  for shard in shards:
    expected_files.update(_paths(output_dir, shard))
  transaction_dirs = sorted(
    path for path in output_dir.rglob("*")
    if path.is_dir() and path.name.startswith(".shard_")
  )
  transaction_files = {
    path.resolve()
    for directory in transaction_dirs
    for path in directory.rglob("*") if path.is_file()
  }
  actual_files = {path.resolve() for path in output_dir.rglob("*") if path.is_file()}
  unexpected = actual_files - {path.resolve() for path in expected_files} - transaction_files
  if unexpected or any(not path.is_file() for path in expected_files):
    raise ValueError(
      f"formal shard namespace has missing/unexpected published files: "
      f"{[path.as_posix() for path in sorted(unexpected)]}"
    )
  transaction_inventory = []
  for directory in transaction_dirs:
    files = sorted(path for path in directory.rglob("*") if path.is_file())
    transaction_inventory.append({
      "path": directory.relative_to(output_dir).as_posix(),
      "files": [{
        "path": path.relative_to(directory).as_posix(),
        "byte_size": path.stat().st_size, "sha256": file_sha256(path),
      } for path in files],
    })

  method_counts: dict[str, Counter] = {}
  for dataset in ("brainweb", "hecktor"):
    aggregate = Counter()
    for item in verified:
      if item["dataset"] == dataset:
        aggregate.update(item["method_row_counts"])
    method_counts[dataset] = aggregate
  expected_method_counts = {
    "brainweb": {
      "MLEM": 8 * 8_352, "OSEM": 5 * 8_352,
      "Bowsher-KEM": 63 * 8_352, "SelfKEM": 63 * 8_352,
      "MR-KEM": 63 * 8_352,
      "SAM-KEM-MedSAM2-Qwen3-full": 378 * 8_352,
    },
    "hecktor": {
      "MLEM": 8 * 540, "OSEM": 5 * 540,
      "Bowsher-KEM": 63 * 540, "SelfKEM": 63 * 540,
      "CT-KEM": 63 * 540,
      "SAM-KEM-MedSAM2-Qwen3-full": 378 * 540,
    },
  }
  if any(dict(method_counts[name]) != expected for name, expected in expected_method_counts.items()):
    raise ValueError("aggregate method-grid row counts differ from the frozen protocol")

  started = datetime.fromisoformat(str(execution["created_utc"]))
  completed = datetime.fromisoformat(str(progress["updated_utc"]))
  shard_compute_seconds = sum(float(item.get("elapsed_seconds", 0.0)) for item in verified)
  process_ids = sorted({int(item["process_id"]) for item in verified})
  first = verified[0]
  last = verified[-1]
  result = {
    "artifact_kind": "validation_sweep_shard_integrity_audit", "schema_version": 1,
    "status": "PASS_complete_pending_merge", "read_only": True,
    "audit_git_commit": audit_commit, "execution_git_commit": execution_commit,
    "plan_sha256": plan_sha256, "config_sha256": file_sha256(config_path),
    "protocol_sha256": file_sha256(protocol_path),
    "manifest_sha256": file_sha256(manifest_path),
    "execution_record_sha256": execution_sha256,
    "progress_sha256": file_sha256(progress_path),
    "shard_batch_success_sha256": file_sha256(completion_path),
    "shard_set_sha256": shard_set_sha256,
    "shards_total": len(verified),
    "shards_reused_verified": int(progress["shards_reused_verified"]),
    "shards_written": int(progress["shards_written"]),
    "dataset_shard_counts": dict(Counter(item["dataset"] for item in verified)),
    "total_cases": sum(int(item["case_count"]) for item in verified),
    "total_result_rows": sum(int(item["row_count"]) for item in verified),
    "total_operator_updates": int(plan["total_operator_updates"]),
    "method_row_counts": {name: dict(values) for name, values in method_counts.items()},
    "global_primary_key_uniqueness": (
      "verified_per_shard_and_by_disjoint_dataset_subject_slice_membership"
    ),
    "published_file_counts": {
      "result_csv": 741, "csv_manifest": 741, "shard_success": 741,
      "shard_artifacts": 2_223, "top_level_json": 3, "total": 2_226,
    },
    "first_shard": first, "last_shard": last,
    "execution_created_utc": execution["created_utc"],
    "completion_updated_utc": progress["updated_utc"],
    "elapsed_wall_seconds_including_interruptions": (completed - started).total_seconds(),
    "sum_shard_compute_seconds": shard_compute_seconds,
    "recorded_process_ids": process_ids,
    "recorded_process_id_count": len(process_ids),
    "temporary_transaction_directories": {
      "count": len(transaction_inventory),
      "disposition": "preserved_read_only_excluded_from_published_shard_set",
      "entries": transaction_inventory,
    },
    "merge_created": False, "frozen_hyperparameters_created": False,
    "test_locked_access": "prohibited",
  }
  atomic_write_json(audit_path, result)
  return result
