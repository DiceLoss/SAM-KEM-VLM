"""Resumable eight-process execution of the frozen validation sweep shards."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import platform
import subprocess
import sys
import tempfile
import time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import scipy

from .artifacts import (
  atomic_csv_writer,
  atomic_write_json,
  file_sha256,
  finalize_csv,
  verify_csv_artifact,
)
from .config import load_contract, require_development_execution
from .experiments import Case, load_case_manifest, validate_case_contract
from .operator_factory import build_discrete_radon
from .prior_registration import validate_registered_validation_prior_configuration
from .protocols import load_protocol
from .reconstruction import operator_support_mask
from .sweep_evaluation import SWEEP_FIELDS, evaluate_validation_case, parse_sweep_grid
from .sweep_sharding import group_validation_cases


PRIMARY_KEY = [
  "run_id", "dataset", "subject_id", "slice_id", "realization_id", "count_level",
  "method", "sigma", "radius", "neighbours", "iterations", "alpha",
]
THREAD_VARIABLES = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")
_WORKER_STATE: dict = {}


def line_ending_equivalent_sha256(path: Path, expected: str) -> bool:
  """Accept only exact or LF/CRLF-equivalent text bytes for a frozen YAML."""
  payload = path.read_bytes()
  lf = payload.replace(b"\r\n", b"\n")
  variants = (payload, lf, lf.replace(b"\n", b"\r\n"))
  return expected in {hashlib.sha256(value).hexdigest() for value in variants}


def _clean_commit(root: Path) -> str:
  status = subprocess.run(
    ["git", "status", "--porcelain"], cwd=root, check=True,
    capture_output=True, text=True,
  ).stdout.strip()
  if status:
    raise RuntimeError(f"formal shard run requires a clean worktree: {status.splitlines()}")
  return subprocess.run(
    ["git", "rev-parse", "HEAD"], cwd=root, check=True,
    capture_output=True, text=True,
  ).stdout.strip()


def _paths(output_dir: Path, shard: dict) -> tuple[Path, Path, Path]:
  directory = (
    output_dir / shard["dataset"] / shard["subject_id"] / shard["slice_id"]
    / shard["shard_id"]
  )
  fragment = directory / "results.csv"
  return (
    fragment,
    fragment.with_suffix(fragment.suffix + ".manifest.json"),
    directory / "_SUCCESS.json",
  )


def verify_sweep_shard(
  output_dir: Path, shard: dict, *, execution_commit: str,
  execution_record_sha256: str, plan_sha256: str, run_id: str,
) -> dict | None:
  """Return verified shard identity, None when absent, or reject partial state."""
  fragment, sidecar, marker_path = _paths(output_dir, shard)
  present = [path.is_file() for path in (fragment, sidecar, marker_path)]
  if not fragment.parent.exists():
    return None
  if not all(present):
    raise ValueError(
      f"incomplete shard cannot be resumed or overwritten: {shard['shard_id']}"
    )
  record = verify_csv_artifact(fragment)
  marker = json.loads(marker_path.read_text(encoding="utf-8"))
  with fragment.open("r", encoding="utf-8", newline="") as handle:
    rows = list(csv.DictReader(handle))
  expected_keys = {
    (int(item["count_level"]), int(item["realization_id"]))
    for item in shard["case_keys"]
  }
  case_rows = Counter((int(row["count_level"]), int(row["realization_id"])) for row in rows)
  method_rows = Counter(row["method"] for row in rows)
  anatomy_method = "MR-KEM" if shard["dataset"] == "brainweb" else "CT-KEM"
  expected_method_rows = {
    "MLEM": 8 * int(shard["case_count"]),
    "OSEM": 5 * int(shard["case_count"]),
    "Bowsher-KEM": 63 * int(shard["case_count"]),
    "SelfKEM": 63 * int(shard["case_count"]),
    anatomy_method: 63 * int(shard["case_count"]),
    "SAM-KEM-MedSAM2-Qwen3-full": 378 * int(shard["case_count"]),
  }
  valid_rows = all(
    row["run_id"] == run_id
    and row["dataset"] == shard["dataset"]
    and row["subject_id"] == shard["subject_id"]
    and row["slice_id"] == shard["slice_id"]
    and row["split"] == "validation"
    for row in rows
  )
  if (
    record.get("schema_name") != "validation_sweep_slice_shard_v1"
    or record.get("split") != "validation"
    or record.get("primary_key") != PRIMARY_KEY
    or int(record.get("row_count", -1)) != int(shard["expected_result_rows"])
    or len(rows) != int(shard["expected_result_rows"])
    or set(case_rows) != expected_keys
    or any(value != 580 for value in case_rows.values())
    or dict(method_rows) != expected_method_rows
    or not valid_rows
    or marker.get("artifact_kind") != "validation_sweep_slice_shard_success"
    or marker.get("status") != "complete"
    or marker.get("shard_id") != shard["shard_id"]
    or marker.get("execution_git_commit") != execution_commit
    or marker.get("execution_record_sha256") != execution_record_sha256
    or marker.get("plan_sha256") != plan_sha256
    or int(marker.get("row_count", -1)) != int(shard["expected_result_rows"])
    or marker.get("fragment_sha256") != file_sha256(fragment)
    or marker.get("fragment_manifest_sha256") != file_sha256(sidecar)
  ):
    raise ValueError(f"shard failed strict resume verification: {shard['shard_id']}")
  return {
    "shard_id": shard["shard_id"], "dataset": shard["dataset"],
    "subject_id": shard["subject_id"], "slice_id": shard["slice_id"],
    "case_count": int(shard["case_count"]), "row_count": len(rows),
    "method_row_counts": dict(method_rows),
    "fragment_path": fragment.relative_to(output_dir).as_posix(),
    "fragment_sha256": file_sha256(fragment),
    "fragment_manifest_sha256": file_sha256(sidecar),
    "success_marker_sha256": file_sha256(marker_path),
    "elapsed_seconds": float(marker.get("elapsed_seconds", 0.0)),
    "process_id": int(marker.get("process_id", -1)),
  }


def _worker_runtime(root: Path, config_path: Path, protocol_path: Path) -> dict:
  key = (str(root), str(config_path), str(protocol_path))
  if _WORKER_STATE.get("key") != key:
    cfg, contract = load_contract(config_path)
    protocol = load_protocol(protocol_path)
    operator = build_discrete_radon(
      repo_root=root, size=int(cfg["forward_model"]["image_size"]),
      angles=contract.angles, operator_id=contract.operator_id,
      expected_implementation_sha256=cfg["forward_model"]["implementation_sha256"],
    )
    _WORKER_STATE.clear()
    _WORKER_STATE.update({
      "key": key, "cfg": cfg, "contract": contract, "protocol": protocol,
      "grid": parse_sweep_grid(protocol), "operator": operator,
      "support": operator_support_mask(operator),
    })
  return _WORKER_STATE


def _write_shard_process(payload: tuple) -> dict:
  (
    root, config_path, protocol_path, output_dir, shard, cases,
    execution_commit, execution_record_sha256, plan_sha256,
  ) = payload
  runtime = _worker_runtime(root, config_path, protocol_path)
  fragment, sidecar, marker_path = _paths(output_dir, shard)
  final_directory = fragment.parent
  if final_directory.exists():
    raise FileExistsError(f"refusing to overwrite shard: {shard['shard_id']}")
  final_directory.parent.mkdir(parents=True, exist_ok=True)
  temporary_directory = Path(tempfile.mkdtemp(
    prefix=f'.{shard["shard_id"]}.', dir=final_directory.parent,
  ))
  temporary_fragment = temporary_directory / fragment.name
  temporary_sidecar = temporary_fragment.with_suffix(
    temporary_fragment.suffix + ".manifest.json"
  )
  temporary_marker = temporary_directory / marker_path.name
  started = time.perf_counter()
  row_count = 0
  with atomic_csv_writer(temporary_fragment, SWEEP_FIELDS) as writer:
    for case in cases:
      records = evaluate_validation_case(
        case, runtime["operator"], runtime["support"], runtime["cfg"],
        runtime["contract"].run_id, runtime["protocol"].methods, runtime["grid"],
      )
      if len(records) != 580:
        raise ValueError(
          f"case in {shard['shard_id']} produced {len(records)} rows, expected 580"
        )
      writer.writerows(records)
      row_count += len(records)
  if row_count != int(shard["expected_result_rows"]):
    raise ValueError(f"shard {shard['shard_id']} produced {row_count} rows")
  finalize_csv(
    temporary_fragment, schema_name="validation_sweep_slice_shard_v1",
    primary_key=PRIMARY_KEY, split="validation",
  )
  atomic_write_json(temporary_marker, {
    "artifact_kind": "validation_sweep_slice_shard_success", "schema_version": 1,
    "status": "complete", "execution_git_commit": execution_commit,
    "execution_record_sha256": execution_record_sha256, "plan_sha256": plan_sha256,
    "shard_id": shard["shard_id"], "dataset": shard["dataset"],
    "subject_id": shard["subject_id"], "slice_id": shard["slice_id"],
    "case_count": int(shard["case_count"]), "row_count": row_count,
    "fragment_sha256": file_sha256(temporary_fragment),
    "fragment_manifest_sha256": file_sha256(temporary_sidecar),
    "elapsed_seconds": time.perf_counter() - started, "process_id": os.getpid(),
  })
  os.replace(temporary_directory, final_directory)
  result = verify_sweep_shard(
    output_dir, shard, execution_commit=execution_commit,
    execution_record_sha256=execution_record_sha256,
    plan_sha256=plan_sha256, run_id=runtime["contract"].run_id,
  )
  if result is None:
    raise RuntimeError(f"new shard disappeared after publication: {shard['shard_id']}")
  result["elapsed_seconds"] = time.perf_counter() - started
  result["process_id"] = os.getpid()
  return result


def _shard_set_sha256(items: list[dict]) -> str:
  digest = hashlib.sha256()
  for item in sorted(items, key=lambda value: value["shard_id"]):
    digest.update(
      (
        f'{item["shard_id"]}\0{item["fragment_sha256"]}\0'
        f'{item["fragment_manifest_sha256"]}\0{item["success_marker_sha256"]}\n'
      ).encode("utf-8")
    )
  return digest.hexdigest()


def _write_progress(
  path: Path, *, status: str, total: int, reused: int, written: int,
  failed_shard: str | None = None,
) -> None:
  atomic_write_json(path, {
    "artifact_kind": "validation_sweep_shard_progress", "schema_version": 1,
    "status": status, "shards_total": total,
    "shards_reused_verified": reused, "shards_written": written,
    "shards_completed": reused + written, "failed_shard": failed_shard,
    "updated_utc": datetime.now(timezone.utc).isoformat(),
  }, replace=True)


def run_formal_validation_sweep_shards(
  root: Path, config_path: Path, protocol_path: Path, plan_path: Path,
  output_dir: Path, *, workers: int, resume_verified: bool,
  release_check_only: bool = False,
) -> dict:
  """Run or strictly resume all frozen slice shards without merging or freezing."""
  root = root.resolve()
  config_path = config_path.resolve()
  protocol_path = protocol_path.resolve()
  plan_path = plan_path.resolve()
  output_dir = output_dir.resolve()
  if workers != 8:
    raise ValueError("the released formal sweep requires exactly 8 worker processes")
  thread_environment = {name: os.environ.get(name) for name in THREAD_VARIABLES}
  if any(value is not None for value in thread_environment.values()):
    raise RuntimeError(f"formal sweep requires unset BLAS thread overrides: {thread_environment}")
  execution_commit = _clean_commit(root)
  plan = json.loads(plan_path.read_text(encoding="utf-8"))
  plan_sha256 = file_sha256(plan_path)
  if (
    plan.get("status") != "ready_for_formal_sharded_sweep_pending_researcher_release"
    or plan.get("execution_backend") != "scipy_sparse_csr_cpu_float64_multiprocess"
    or plan.get("worker_processes") != workers
    or plan.get("shard_count") != 741
    or plan.get("total_cases") != 8_892
    or plan.get("total_result_rows") != 5_157_360
    or plan.get("total_operator_updates") != 41_347_800
    or plan.get("test_locked_access") != "prohibited"
  ):
    raise ValueError("formal sweep plan is not the reviewed release candidate")
  cfg, contract = load_contract(config_path)
  require_development_execution(cfg)
  registration = validate_registered_validation_prior_configuration(root, config_path, cfg)
  protocol = load_protocol(protocol_path)
  manifest_path = root / str(cfg["outputs"]["validation_manifest"])
  if (
    protocol.kind != "hyperparameter_sweep" or protocol.split != "validation"
    or file_sha256(config_path) != plan.get("config_sha256")
    or not line_ending_equivalent_sha256(protocol_path, str(plan.get("protocol_sha256")))
    or file_sha256(manifest_path) != plan.get("manifest_sha256")
    or registration != plan.get("prior_registration")
    or output_dir.relative_to(root).as_posix() != plan.get("formal_output_namespace")
  ):
    raise ValueError("formal sweep inputs differ from the frozen shard plan")
  cases = load_case_manifest(root, manifest_path, split="validation")
  validate_case_contract(cases, contract)
  shards = group_validation_cases(cases)
  if shards != plan.get("shards"):
    raise ValueError("current validation cases differ from the frozen shard membership")
  grouped_cases: dict[tuple[str, str, str], list[Case]] = defaultdict(list)
  for case in cases:
    grouped_cases[(case.dataset, case.subject_id, case.slice_id)].append(case)
  for members in grouped_cases.values():
    members.sort(key=lambda case: (case.count_level, case.realization_id))

  if release_check_only:
    if resume_verified:
      raise ValueError("release check cannot be combined with --resume-verified")
    if output_dir.exists():
      raise FileExistsError("release check requires an absent formal output namespace")
    return {
      "artifact_kind": "validation_sweep_shard_release_check", "schema_version": 1,
      "status": "PASS_ready_for_formal_shard_execution",
      "execution_git_commit": execution_commit,
      "plan_path": plan_path.relative_to(root).as_posix(), "plan_sha256": plan_sha256,
      "config_sha256": file_sha256(config_path),
      "protocol_sha256": file_sha256(protocol_path),
      "protocol_plan_sha256": plan["protocol_sha256"],
      "protocol_binding": "exact_or_lf_crlf_equivalent_bytes",
      "manifest_sha256": file_sha256(manifest_path),
      "prior_registration": registration,
      "execution_backend": "scipy_sparse_csr_cpu_float64_multiprocess",
      "worker_processes": workers, "gpu_used": False,
      "thread_environment": thread_environment,
      "shard_count": len(shards),
      "dataset_shard_counts": dict(Counter(item["dataset"] for item in shards)),
      "total_cases": len(cases), "total_result_rows": int(plan["total_result_rows"]),
      "total_operator_updates": int(plan["total_operator_updates"]),
      "formal_output_namespace": output_dir.relative_to(root).as_posix(),
      "formal_output_created": False, "merge_created": False,
      "frozen_hyperparameters_created": False, "test_locked_access": "prohibited",
    }

  execution_record_path = output_dir / "execution_record.json"
  progress_path = output_dir / "progress.json"
  completion_path = output_dir / "shard_batch_SUCCESS.json"
  if output_dir.exists():
    if not resume_verified:
      raise FileExistsError("formal shard namespace exists; use --resume-verified")
    if not execution_record_path.is_file():
      raise ValueError("formal shard namespace lacks its immutable execution record")
    execution_record = json.loads(execution_record_path.read_text(encoding="utf-8"))
    if (
      execution_record.get("execution_git_commit") != execution_commit
      or execution_record.get("plan_sha256") != plan_sha256
      or execution_record.get("config_sha256") != file_sha256(config_path)
      or execution_record.get("protocol_sha256") != file_sha256(protocol_path)
      or execution_record.get("manifest_sha256") != file_sha256(manifest_path)
      or execution_record.get("worker_processes") != workers
    ):
      raise ValueError("resume execution identity differs from the existing shard run")
  else:
    output_dir.mkdir(parents=True)
    execution_record = {
      "artifact_kind": "validation_sweep_shard_execution", "schema_version": 1,
      "status": "running", "execution_git_commit": execution_commit,
      "plan_path": plan_path.relative_to(root).as_posix(), "plan_sha256": plan_sha256,
      "config_sha256": file_sha256(config_path),
      "protocol_sha256": file_sha256(protocol_path),
      "protocol_plan_sha256": plan["protocol_sha256"],
      "protocol_binding": "exact_or_lf_crlf_equivalent_bytes",
      "manifest_sha256": file_sha256(manifest_path),
      "execution_backend": "scipy_sparse_csr_cpu_float64_multiprocess",
      "worker_processes": workers, "gpu_used": False,
      "thread_environment": thread_environment,
      "python": sys.version, "platform": platform.platform(),
      "numpy_version": np.__version__, "scipy_version": scipy.__version__,
      "shard_count": len(shards), "total_cases": len(cases),
      "total_result_rows": int(plan["total_result_rows"]),
      "test_locked_access": "prohibited",
      "created_utc": datetime.now(timezone.utc).isoformat(),
    }
    atomic_write_json(execution_record_path, execution_record)
  execution_record_sha256 = file_sha256(execution_record_path)

  verified: list[dict] = []
  pending: list[dict] = []
  for shard in shards:
    item = verify_sweep_shard(
      output_dir, shard, execution_commit=execution_commit,
      execution_record_sha256=execution_record_sha256,
      plan_sha256=plan_sha256, run_id=contract.run_id,
    )
    if item is None:
      pending.append(shard)
    else:
      verified.append(item)
  if verified and not resume_verified:
    raise FileExistsError("verified shards exist; use --resume-verified")
  if completion_path.exists():
    completion = json.loads(completion_path.read_text(encoding="utf-8"))
    if (
      pending
      or completion.get("status") != "complete_pending_merge_integrity_gate"
      or completion.get("execution_git_commit") != execution_commit
      or completion.get("execution_record_sha256") != execution_record_sha256
      or completion.get("plan_sha256") != plan_sha256
      or completion.get("shards_completed") != len(shards)
      or completion.get("total_cases") != 8_892
      or completion.get("total_result_rows") != 5_157_360
      or completion.get("shard_set_sha256") != _shard_set_sha256(verified)
      or completion.get("merge_created") is not False
      or completion.get("frozen_hyperparameters_created") is not False
    ):
      raise ValueError("existing shard completion marker failed full verification")
    _write_progress(
      progress_path, status="complete", total=len(shards),
      reused=len(verified), written=0,
    )
    return completion
  _write_progress(
    progress_path, status="running", total=len(shards), reused=len(verified), written=0,
  )
  written: list[dict] = []
  failed_shard = None
  executor = ProcessPoolExecutor(max_workers=workers)
  futures = {}
  try:
    for shard in pending:
      key = (shard["dataset"], shard["subject_id"], shard["slice_id"])
      payload = (
        root, config_path, protocol_path, output_dir, shard, grouped_cases[key],
        execution_commit, execution_record_sha256, plan_sha256,
      )
      futures[executor.submit(_write_shard_process, payload)] = shard["shard_id"]
    for future in as_completed(futures):
      failed_shard = futures[future]
      written.append(future.result())
      failed_shard = None
      print(json.dumps({
        "status": "running", "shards_total": len(shards),
        "shards_reused_verified": len(verified), "shards_written": len(written),
        "shards_completed": len(verified) + len(written),
        "latest_shard": written[-1]["shard_id"],
      }, sort_keys=True), flush=True)
      _write_progress(
        progress_path, status="running", total=len(shards),
        reused=len(verified), written=len(written),
      )
  except BaseException:
    for future in futures:
      future.cancel()
    _write_progress(
      progress_path, status="interrupted", total=len(shards),
      reused=len(verified), written=len(written), failed_shard=failed_shard,
    )
    executor.shutdown(wait=True, cancel_futures=True)
    raise
  else:
    executor.shutdown(wait=True)

  completed = []
  for shard in shards:
    item = verify_sweep_shard(
      output_dir, shard, execution_commit=execution_commit,
      execution_record_sha256=execution_record_sha256,
      plan_sha256=plan_sha256, run_id=contract.run_id,
    )
    if item is None:
      raise RuntimeError(f"shard missing after completed executor run: {shard['shard_id']}")
    completed.append(item)
  result = {
    "artifact_kind": "validation_sweep_shard_batch_completion", "schema_version": 1,
    "status": "complete_pending_merge_integrity_gate",
    "execution_git_commit": execution_commit,
    "execution_record_sha256": execution_record_sha256,
    "plan_sha256": plan_sha256, "worker_processes": workers,
    "shards_total": len(shards), "shards_reused_verified": len(verified),
    "shards_written": len(written), "shards_completed": len(completed),
    "total_cases": sum(int(item["case_count"]) for item in completed),
    "total_result_rows": sum(int(item["row_count"]) for item in completed),
    "shard_set_sha256": _shard_set_sha256(completed),
    "merge_created": False, "frozen_hyperparameters_created": False,
    "test_locked_access": "prohibited",
  }
  atomic_write_json(completion_path, result)
  _write_progress(
    progress_path, status="complete", total=len(shards),
    reused=len(verified), written=len(written),
  )
  return result
