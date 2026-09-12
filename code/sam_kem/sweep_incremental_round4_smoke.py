"""Two-slice numerical smoke gate for the fourth incremental shell."""

from __future__ import annotations

import json
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from .artifacts import atomic_write_json, file_sha256
from .config import load_contract
from .experiments import Case, load_case_manifest, validate_case_contract
from .sweep_incremental import _write_shard_process, verify_incremental_shard
from .sweep_incremental_round4 import (
  EXPECTED_METHOD_ROWS,
  EXPECTED_OPERATOR_UPDATES,
  EXPECTED_ROWS,
  FIXED_SMOKE_KEYS,
  OUTPUT_NAMESPACE,
  REVIEWED_ROUND4_PLAN_SHA256,
  build_round4_shards,
  release_check_round4_incremental_validation_sweep,
  round4_candidate_map,
)
from .sweep_incremental_smoke import _fragment_summary


def fixed_round4_smoke_shards(
  cases: list[Case], candidates: dict,
) -> tuple[list[dict], dict[tuple[str, str, str], list[Case]]]:
  """Select the fixed BW/HE slice shards and all 12 cases in each."""
  all_shards = build_round4_shards(cases, candidates)
  selected = [
    shard for shard in all_shards
    if (shard["dataset"], shard["subject_id"], shard["slice_id"])
    in FIXED_SMOKE_KEYS
  ]
  if len(selected) != 2 or {
    (item["dataset"], item["subject_id"], item["slice_id"]) for item in selected
  } != FIXED_SMOKE_KEYS:
    raise ValueError("fixed round-4 smoke shards are absent or ambiguous")
  grouped: dict[tuple[str, str, str], list[Case]] = defaultdict(list)
  for case in cases:
    key = (case.dataset, case.subject_id, case.slice_id)
    if key in FIXED_SMOKE_KEYS:
      grouped[key].append(case)
  for members in grouped.values():
    members.sort(key=lambda case: (case.count_level, case.realization_id))
  if len(grouped) != 2 or any(len(members) != 12 for members in grouped.values()):
    raise ValueError("each fixed round-4 smoke shard must contain 12 cases")
  return selected, dict(grouped)


def run_round4_incremental_validation_sweep_smoke(
  root: Path, config_path: Path, protocol_path: Path,
  union_audit_path: Path, round3_plan_path: Path,
  round3_merge_audit_path: Path, round4_plan_path: Path,
  output_dir: Path, audit_path: Path,
) -> dict:
  """Execute and resume-verify only the fixed BW/HE Round-4 shards."""
  root = root.resolve()
  config_path = config_path.resolve()
  protocol_path = protocol_path.resolve()
  union_audit_path = union_audit_path.resolve()
  round3_plan_path = round3_plan_path.resolve()
  round3_merge_audit_path = round3_merge_audit_path.resolve()
  round4_plan_path = round4_plan_path.resolve()
  output_dir = output_dir.resolve()
  audit_path = audit_path.resolve()
  if output_dir.exists() or audit_path.exists():
    raise FileExistsError("refusing to overwrite round-4 smoke artifacts")

  release = release_check_round4_incremental_validation_sweep(
    root, config_path, protocol_path, union_audit_path, round3_plan_path,
    round3_merge_audit_path, round4_plan_path, root / OUTPUT_NAMESPACE,
    workers=8, release_check_only=True,
  )
  plan = json.loads(round4_plan_path.read_text(encoding="utf-8"))
  candidates = round4_candidate_map(plan)
  cfg, contract = load_contract(config_path)
  manifest_path = root / str(cfg["outputs"]["validation_manifest"])
  cases = load_case_manifest(root, manifest_path, split="validation")
  validate_case_contract(cases, contract)
  shards, grouped = fixed_round4_smoke_shards(cases, candidates)

  output_dir.mkdir(parents=True)
  execution_record_path = output_dir / "smoke_execution_record.json"
  atomic_write_json(execution_record_path, {
    "artifact_kind": "validation_hyperparameter_expansion_round4_smoke_execution",
    "schema_version": 1,
    "status": "running",
    "execution_git_commit": release["execution_git_commit"],
    "round4_plan_sha256": REVIEWED_ROUND4_PLAN_SHA256,
    "round3_plan_sha256": release["round3_plan_sha256"],
    "union_selection_audit_sha256": release["union_selection_audit_sha256"],
    "round3_merge_audit_sha256": release["round3_merge_audit_sha256"],
    "config_sha256": release["config_sha256"],
    "protocol_sha256": release["protocol_sha256"],
    "manifest_sha256": release["manifest_sha256"],
    "output_role": "staging_smoke_not_formal_round4_expansion",
    "test_locked_access": "prohibited",
  })
  execution_record_sha256 = file_sha256(execution_record_path)
  payloads = []
  for shard in shards:
    key = (shard["dataset"], shard["subject_id"], shard["slice_id"])
    payloads.append((
      root, config_path, output_dir, shard, grouped[key], candidates,
      release["execution_git_commit"], execution_record_sha256,
      REVIEWED_ROUND4_PLAN_SHA256,
    ))
  with ProcessPoolExecutor(max_workers=2) as executor:
    written = list(executor.map(_write_shard_process, payloads))

  verified = []
  for shard in shards:
    item = verify_incremental_shard(
      output_dir, shard, candidates,
      execution_commit=release["execution_git_commit"],
      execution_record_sha256=execution_record_sha256,
      plan_sha256=REVIEWED_ROUND4_PLAN_SHA256,
      run_id=contract.run_id,
    )
    if item is None:
      raise RuntimeError(f"round-4 smoke shard disappeared: {shard['shard_id']}")
    verified.append(item)
  summaries = [_fragment_summary(output_dir, shard) for shard in shards]
  if (
    sum(item["row_count"] for item in summaries) != 30
    or any(
      item["method_row_counts"] != EXPECTED_METHOD_ROWS[item["dataset"]]
      for item in summaries
    )
    or len({item["process_id"] for item in summaries}) != 2
    or {item["shard_id"] for item in written}
    != {item["shard_id"] for item in verified}
  ):
    raise ValueError("round-4 smoke fragments differ from the reviewed plan")

  result = {
    "artifact_kind": "validation_hyperparameter_expansion_round4_shard_smoke",
    "schema_version": 1,
    "status": "PASS",
    "execution_git_commit": release["execution_git_commit"],
    "round4_plan_sha256": REVIEWED_ROUND4_PLAN_SHA256,
    "round3_plan_sha256": release["round3_plan_sha256"],
    "union_selection_audit_sha256": release["union_selection_audit_sha256"],
    "round3_merge_audit_sha256": release["round3_merge_audit_sha256"],
    "original_merged_csv_sha256": release["original_merged_csv_sha256"],
    "round1_merged_csv_sha256": release["round1_merged_csv_sha256"],
    "round2_merged_csv_sha256": release["round2_merged_csv_sha256"],
    "round3_merged_csv_sha256": release["round3_merged_csv_sha256"],
    "execution_backend": "scipy_sparse_csr_cpu_float64_multiprocess",
    "worker_processes": 2,
    "gpu_used": False,
    "fixed_shard_count": 2,
    "fragments_written": len(written),
    "fragments_resume_verified": len(verified),
    "total_rows": 30,
    "process_ids_unique": True,
    "formal_declared_result_rows": EXPECTED_ROWS,
    "formal_declared_operator_updates": EXPECTED_OPERATOR_UPDATES,
    "fragments": summaries,
    "output_namespace": output_dir.relative_to(root).as_posix(),
    "output_role": "staging_smoke_not_formal_round4_expansion",
    "formal_output_created": False,
    "merge_created": False,
    "union_selection_audit_created": False,
    "frozen_hyperparameters_created": False,
    "test_locked_access": "prohibited",
  }
  atomic_write_json(audit_path, result)
  return result
