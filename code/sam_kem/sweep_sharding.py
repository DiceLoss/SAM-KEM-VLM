"""Immutable slice-group execution plan for the development validation sweep."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

from .artifacts import atomic_write_json, file_sha256
from .config import load_contract, require_development_execution
from .experiments import Case, load_case_manifest, validate_case_contract
from .prior_registration import validate_registered_validation_prior_configuration
from .protocols import load_protocol
from .sweep_preflight import _commit_allowing_untracked, sweep_workload


RESULT_ROWS_PER_CASE = 580
CASES_PER_SHARD = 12
RESULT_ROWS_PER_SHARD = RESULT_ROWS_PER_CASE * CASES_PER_SHARD


def group_validation_cases(cases: list[Case]) -> list[dict]:
  """Return stable dataset/subject/slice shards and their ordered case keys."""
  grouped: dict[tuple[str, str, str], list[Case]] = defaultdict(list)
  for case in cases:
    grouped[(case.dataset, case.subject_id, case.slice_id)].append(case)
  shards = []
  for index, key in enumerate(sorted(grouped)):
    dataset, subject_id, slice_id = key
    members = sorted(
      grouped[key], key=lambda case: (case.count_level, case.realization_id),
    )
    shards.append({
      "shard_id": f"shard_{index:04d}",
      "dataset": dataset,
      "subject_id": subject_id,
      "slice_id": slice_id,
      "case_count": len(members),
      "expected_result_rows": len(members) * RESULT_ROWS_PER_CASE,
      "case_keys": [
        {
          "count_level": case.count_level,
          "realization_id": case.realization_id,
        }
        for case in members
      ],
    })
  return shards


def _load_json(path: Path) -> dict:
  value = json.loads(path.read_text(encoding="utf-8"))
  if not isinstance(value, dict):
    raise ValueError(f"expected a JSON object: {path}")
  return value


def build_validation_sweep_shard_plan(
  root: Path,
  config_path: Path,
  protocol_path: Path,
  runtime_preflight_path: Path,
  parallel_preflight_path: Path,
  smoke_audit_path: Path,
  output_path: Path,
) -> dict:
  """Validate reviewed evidence and freeze the complete 741-shard plan."""
  root = root.resolve()
  paths = [
    config_path, protocol_path, runtime_preflight_path, parallel_preflight_path,
    smoke_audit_path, output_path,
  ]
  (
    config_path, protocol_path, runtime_preflight_path, parallel_preflight_path,
    smoke_audit_path, output_path,
  ) = [path.resolve() for path in paths]
  if output_path.exists():
    raise FileExistsError(f"refusing to overwrite shard plan: {output_path}")
  commit = _commit_allowing_untracked(root, {
    runtime_preflight_path, parallel_preflight_path, smoke_audit_path,
  })
  runtime = _load_json(runtime_preflight_path)
  parallel = _load_json(parallel_preflight_path)
  smoke = _load_json(smoke_audit_path)
  if (
    runtime.get("status") != "PASS"
    or runtime.get("execution_backend") != "scipy_sparse_csr_cpu_float64"
    or runtime.get("gpu_used") is not False
    or runtime.get("declared_workload", {}).get("total_result_rows") != 5_157_360
    or runtime.get("declared_workload", {}).get("total_operator_updates") != 41_347_800
  ):
    raise ValueError("serial runtime preflight is not the reviewed PASS artifact")
  if (
    parallel.get("status") != "PASS"
    or parallel.get("execution_backend")
    != "scipy_sparse_csr_cpu_float64_multiprocess"
    or parallel.get("gpu_used") is not False
    or parallel.get("worker_processes") != 8
    or parallel.get("result_evaluations_per_sample") != RESULT_ROWS_PER_CASE
    or parallel.get("formal_output_created") is not False
    or parallel.get("frozen_hyperparameters_created") is not False
  ):
    raise ValueError("parallel scaling preflight is not the reviewed PASS artifact")
  equivalence = smoke.get("metric_equivalence", {})
  if (
    smoke.get("status") != "PASS"
    or smoke.get("fragments_written") != 2
    or smoke.get("fragments_resume_verified") != 2
    or smoke.get("total_rows") != 1_160
    or smoke.get("formal_output_created") is not False
    or smoke.get("frozen_hyperparameters_created") is not False
    or set(equivalence) != {"brainweb", "hecktor"}
    or not all(item.get("all_methods_exact") is True for item in equivalence.values())
  ):
    raise ValueError("shard-writer smoke is not the reviewed PASS artifact")
  evidence = (runtime, parallel, smoke)
  if (
    parallel.get("serial_runtime_preflight_sha256") != file_sha256(runtime_preflight_path)
    or smoke.get("runtime_preflight_sha256") != file_sha256(runtime_preflight_path)
    or smoke.get("parallel_preflight_sha256") != file_sha256(parallel_preflight_path)
  ):
    raise ValueError("preflight evidence SHA chain is incomplete or inconsistent")
  if any(item.get("test_locked_access") != "prohibited" for item in evidence):
    raise ValueError("preflight evidence does not prohibit locked-test access")

  cfg, contract = load_contract(config_path)
  require_development_execution(cfg)
  registration = validate_registered_validation_prior_configuration(root, config_path, cfg)
  protocol = load_protocol(protocol_path)
  if protocol.kind != "hyperparameter_sweep" or protocol.split != "validation":
    raise ValueError("shard planning requires the validation hyperparameter sweep protocol")
  manifest_path = root / str(cfg["outputs"]["validation_manifest"])
  cases = load_case_manifest(root, manifest_path, split="validation")
  validate_case_contract(cases, contract)
  case_counts = dict(Counter(case.dataset for case in cases))
  if case_counts != {"brainweb": 8_352, "hecktor": 540}:
    raise ValueError(f"unexpected validation case counts: {case_counts}")
  shards = group_validation_cases(cases)
  shard_counts = dict(Counter(item["dataset"] for item in shards))
  if shard_counts != {"brainweb": 696, "hecktor": 45} or len(shards) != 741:
    raise ValueError(f"unexpected validation shard counts: {shard_counts}")
  expected_case_keys = [
    {"count_level": count, "realization_id": realization}
    for count in (100_000, 200_000, 500_000, 1_000_000)
    for realization in (0, 1, 2)
  ]
  if any(
    item["case_count"] != CASES_PER_SHARD
    or item["expected_result_rows"] != RESULT_ROWS_PER_SHARD
    or item["case_keys"] != expected_case_keys
    for item in shards
  ):
    raise ValueError("each slice shard must contain the declared 4-count x 3-realization grid")
  workload = sweep_workload(
    protocol, case_counts,
    prompt_mlem_iterations=int(cfg["reconstruction"]["prompt_mlem_iterations"]),
    osem_subsets=int(cfg["reconstruction"]["osem_subsets"]),
  )
  if (
    workload["total_cases"] != 8_892
    or workload["total_result_rows"] != 5_157_360
    or workload["total_operator_updates"] != 41_347_800
  ):
    raise ValueError(f"unexpected declared workload: {workload}")
  if any(
    item.get("config_sha256") != file_sha256(config_path)
    or item.get("protocol_sha256") != file_sha256(protocol_path)
    or item.get("manifest_sha256") != file_sha256(manifest_path)
    for item in evidence
  ):
    raise ValueError("preflight evidence binding differs from current inputs")

  result = {
    "artifact_kind": "validation_sweep_shard_plan", "schema_version": 1,
    "status": "ready_for_formal_sharded_sweep_pending_researcher_release",
    "git_commit": commit,
    "config_path": config_path.relative_to(root).as_posix(),
    "config_sha256": file_sha256(config_path),
    "protocol_path": protocol_path.relative_to(root).as_posix(),
    "protocol_sha256": file_sha256(protocol_path),
    "manifest_path": manifest_path.relative_to(root).as_posix(),
    "manifest_sha256": file_sha256(manifest_path),
    "runtime_preflight_path": runtime_preflight_path.relative_to(root).as_posix(),
    "runtime_preflight_sha256": file_sha256(runtime_preflight_path),
    "parallel_preflight_path": parallel_preflight_path.relative_to(root).as_posix(),
    "parallel_preflight_sha256": file_sha256(parallel_preflight_path),
    "shard_smoke_path": smoke_audit_path.relative_to(root).as_posix(),
    "shard_smoke_sha256": file_sha256(smoke_audit_path),
    "evidence_git_commits": {
      "runtime_preflight": runtime.get("git_commit"),
      "parallel_preflight": parallel.get("git_commit"),
      "shard_smoke": smoke.get("git_commit"),
    },
    "prior_registration": registration,
    "execution_backend": "scipy_sparse_csr_cpu_float64_multiprocess",
    "gpu_used": False, "worker_processes": 8,
    "shard_unit": "dataset_subject_slice",
    "shard_count": len(shards), "dataset_shard_counts": shard_counts,
    "total_cases": len(cases), "result_rows_per_case": RESULT_ROWS_PER_CASE,
    "cases_per_shard": CASES_PER_SHARD,
    "result_rows_per_shard": RESULT_ROWS_PER_SHARD,
    "total_result_rows": workload["total_result_rows"],
    "total_operator_updates": workload["total_operator_updates"],
    "declared_workload": workload,
    "projected_full_sweep_hours_at_measured_throughput": parallel.get(
      "projected_full_sweep_hours_at_measured_throughput"
    ),
    "resume_policy": (
      "skip_only_complete_shards_after_csv_manifest_success_marker_primary_key_"
      "row_count_and_sha256_verification_on_the_same_git_commit"
    ),
    "overwrite_policy": "prohibited",
    "merge_policy": "separate_post_run_integrity_gate",
    "hyperparameter_freeze_policy": "separate_post_merge_researcher_gate",
    "formal_output_namespace": "runs/dev_trainmax_v1/hyperparameter_sweep_shards",
    "test_locked_access": "prohibited",
    "shards": shards,
  }
  atomic_write_json(output_path, result)
  return result
