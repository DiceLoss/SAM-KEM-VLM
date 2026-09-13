"""Transactional merge gate for reviewed sixth-shell validation shards."""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from .artifacts import atomic_write_json, file_sha256
from .config import load_contract, require_development_execution
from .experiments import load_case_manifest, validate_case_contract
from .prior_registration import validate_registered_validation_prior_configuration
from .protocols import load_protocol
from .sweep_evaluation import SWEEP_FIELDS
from .sweep_incremental import verify_incremental_shard
from .sweep_incremental_merge import _verify_transaction_inventory
from .sweep_incremental_round2_merge import ROUND2_MERGED_NAME
from .sweep_incremental_round3_merge import ROUND3_MERGED_NAME
from .sweep_incremental_round4_merge import ROUND4_MERGED_NAME
from .sweep_incremental_round5_merge import ROUND5_MERGED_NAME
from .sweep_incremental_round6 import (
  EXPECTED_CANDIDATES,
  EXPECTED_OPERATOR_UPDATES,
  EXPECTED_ORIGINAL_MERGED_CSV_SHA256,
  EXPECTED_ROUND1_MERGED_CSV_SHA256,
  EXPECTED_ROUND2_MERGED_CSV_SHA256,
  EXPECTED_ROUND3_MERGED_CSV_SHA256,
  EXPECTED_ROUND4_MERGED_CSV_SHA256,
  EXPECTED_ROUND5_MERGED_CSV_SHA256,
  EXPECTED_ROWS,
  REVIEWED_ROUND5_MERGE_AUDIT_SHA256,
  REVIEWED_ROUND5_PLAN_SHA256,
  REVIEWED_ROUND5_UNION_SELECTION_AUDIT_SHA256,
  REVIEWED_ROUND6_PLAN_SHA256,
  REVIEWED_ROUND6_SMOKE_AUDIT_SHA256,
  build_round6_shards,
  round6_candidate_map,
  validate_reviewed_round6_smoke,
)
from .sweep_incremental_round6 import OUTPUT_NAMESPACE as SHARD_NAMESPACE
from .sweep_merge import stream_merge_csv_fragments
from .sweep_sharded import (
  PRIMARY_KEY,
  _paths,
  _shard_set_sha256,
  line_ending_equivalent_sha256,
)

REVIEWED_ROUND6_SHARD_AUDIT_SHA256 = (
  "2d3db8deb0f28750d3f96d43b2322919bc004b243eb3464ac9f745b6ca8e6009"
)
REVIEWED_ROUND6_SHARD_SET_SHA256 = (
  "e9102291d1083b7b4201014560481d953320dd1cf614f7d9c7495088d005acce"
)
OUTPUT_NAMESPACE = "runs/dev_trainmax_v1/hyperparameter_expansion_round6"
ROUND6_MERGED_NAME = "validation_hyperparameter_expansion_round6.csv"
NEXT_UNION_RESULT_ROWS = 5_349_609 + EXPECTED_ROWS


def _json(path: Path) -> dict:
  value = json.loads(path.read_text(encoding="utf-8"))
  if not isinstance(value, dict):
    raise TypeError(f"expected a JSON object: {path}")
  return value


def _clean_commit(root: Path) -> str:
  status = subprocess.run(
    ["git", "status", "--porcelain"], cwd=root, check=True,
    capture_output=True, text=True,
  ).stdout.strip()
  if status:
    raise RuntimeError(
      f"round-6 incremental merge requires a clean worktree: {status.splitlines()}"
    )
  return subprocess.run(
    ["git", "rev-parse", "HEAD"], cwd=root, check=True,
    capture_output=True, text=True,
  ).stdout.strip()


def stream_merge_round6_fragments(
  fragment_paths: list[Path], output_path: Path, *, expected_rows: int = EXPECTED_ROWS,
) -> dict:
  """Merge Round-6 fragments, including reviewed header-only fragments."""
  details = stream_merge_csv_fragments(fragment_paths, output_path)
  if int(details["row_count"]) != expected_rows:
    raise ValueError(
      f"round-6 merge has {details['row_count']} rows, expected {expected_rows}"
    )
  return details


def merge_audited_round6_incremental_validation_sweep(
  root: Path, config_path: Path, protocol_path: Path,
  union_audit_path: Path, round5_plan_path: Path,
  round5_merge_audit_path: Path, round6_plan_path: Path,
  smoke_audit_path: Path, shard_dir: Path, shard_audit_path: Path,
  output_dir: Path, audit_path: Path,
) -> dict:
  """Merge Round-6 shards without physical union, selection, or freezing."""
  root = root.resolve()
  config_path = config_path.resolve()
  protocol_path = protocol_path.resolve()
  union_audit_path = union_audit_path.resolve()
  round5_plan_path = round5_plan_path.resolve()
  round5_merge_audit_path = round5_merge_audit_path.resolve()
  round6_plan_path = round6_plan_path.resolve()
  smoke_audit_path = smoke_audit_path.resolve()
  shard_dir = shard_dir.resolve()
  shard_audit_path = shard_audit_path.resolve()
  output_dir = output_dir.resolve()
  audit_path = audit_path.resolve()
  if output_dir.exists() or audit_path.exists():
    raise FileExistsError("refusing to overwrite round-6 merge output or audit")
  merge_commit = _clean_commit(root)
  if (
    file_sha256(union_audit_path)
    != REVIEWED_ROUND5_UNION_SELECTION_AUDIT_SHA256
    or file_sha256(round5_plan_path) != REVIEWED_ROUND5_PLAN_SHA256
    or file_sha256(round5_merge_audit_path) != REVIEWED_ROUND5_MERGE_AUDIT_SHA256
    or file_sha256(round6_plan_path) != REVIEWED_ROUND6_PLAN_SHA256
    or file_sha256(smoke_audit_path) != REVIEWED_ROUND6_SMOKE_AUDIT_SHA256
    or file_sha256(shard_audit_path) != REVIEWED_ROUND6_SHARD_AUDIT_SHA256
  ):
    raise ValueError("round-6 merge inputs are not the reviewed artifacts")

  union_audit = _json(union_audit_path)
  round5_plan = _json(round5_plan_path)
  round5_merge_audit = _json(round5_merge_audit_path)
  round6_plan = _json(round6_plan_path)
  smoke_audit = _json(smoke_audit_path)
  shard_audit = _json(shard_audit_path)
  validate_reviewed_round6_smoke(smoke_audit)
  cfg, contract = load_contract(config_path)
  require_development_execution(cfg)
  registration = validate_registered_validation_prior_configuration(
    root, config_path, cfg,
  )
  protocol = load_protocol(protocol_path)
  case_manifest_path = root / str(cfg["outputs"]["validation_manifest"])
  original_merged_path = root / str(cfg["outputs"]["hyperparameter_sweep"])
  round1_merged_path = root / (
    "runs/dev_trainmax_v1/hyperparameter_expansion/"
    "validation_hyperparameter_expansion.csv"
  )
  round2_merged_path = root / (
    "runs/dev_trainmax_v1/hyperparameter_expansion_round2"
  ) / ROUND2_MERGED_NAME
  round3_merged_path = root / (
    "runs/dev_trainmax_v1/hyperparameter_expansion_round3"
  ) / ROUND3_MERGED_NAME
  round4_merged_path = root / (
    "runs/dev_trainmax_v1/hyperparameter_expansion_round4"
  ) / ROUND4_MERGED_NAME
  round5_merged_path = root / (
    "runs/dev_trainmax_v1/hyperparameter_expansion_round5"
  ) / ROUND5_MERGED_NAME
  frozen_path = root / str(cfg["outputs"]["frozen_hyperparameters"])
  if (
    union_audit.get("status")
    != "BLOCKED_expanded_boundary_requires_additional_shell"
    or union_audit.get("expanded_boundary_selection_count") != 3
    or union_audit.get("union_rows_streamed") != 5_349_609
    or round5_plan.get("incremental_candidate_count") != 6
    or round5_merge_audit.get("status")
    != "PASS_complete_pending_round5_union_selection_audit"
    or round5_merge_audit.get("round5_merged_csv_sha256")
    != EXPECTED_ROUND5_MERGED_CSV_SHA256
    or round6_plan.get("status") != "ready_for_researcher_review_no_execution"
    or round6_plan.get("expansion_round") != 6
    or round6_plan.get("incremental_candidate_count") != EXPECTED_CANDIDATES
    or round6_plan.get("expected_incremental_result_rows") != EXPECTED_ROWS
    or round6_plan.get("expected_incremental_operator_updates")
    != EXPECTED_OPERATOR_UPDATES
    or shard_audit.get("status")
    != "PASS_complete_pending_round6_incremental_merge"
    or shard_audit.get("read_only") is not True
    or shard_audit.get("union_selection_audit_sha256")
    != REVIEWED_ROUND5_UNION_SELECTION_AUDIT_SHA256
    or shard_audit.get("round5_plan_sha256") != REVIEWED_ROUND5_PLAN_SHA256
    or shard_audit.get("round5_merge_audit_sha256")
    != REVIEWED_ROUND5_MERGE_AUDIT_SHA256
    or shard_audit.get("round6_plan_sha256") != REVIEWED_ROUND6_PLAN_SHA256
    or shard_audit.get("round6_smoke_audit_sha256")
    != REVIEWED_ROUND6_SMOKE_AUDIT_SHA256
    or shard_audit.get("config_sha256") != file_sha256(config_path)
    or not line_ending_equivalent_sha256(
      protocol_path, str(shard_audit.get("protocol_sha256")),
    )
    or shard_audit.get("manifest_sha256") != file_sha256(case_manifest_path)
    or shard_audit.get("original_merged_csv_sha256")
    != EXPECTED_ORIGINAL_MERGED_CSV_SHA256
    or shard_audit.get("round1_merged_csv_sha256")
    != EXPECTED_ROUND1_MERGED_CSV_SHA256
    or shard_audit.get("round2_merged_csv_sha256")
    != EXPECTED_ROUND2_MERGED_CSV_SHA256
    or shard_audit.get("round3_merged_csv_sha256")
    != EXPECTED_ROUND3_MERGED_CSV_SHA256
    or shard_audit.get("round4_merged_csv_sha256")
    != EXPECTED_ROUND4_MERGED_CSV_SHA256
    or shard_audit.get("round5_merged_csv_sha256")
    != EXPECTED_ROUND5_MERGED_CSV_SHA256
    or shard_audit.get("shard_set_sha256") != REVIEWED_ROUND6_SHARD_SET_SHA256
    or shard_audit.get("shards_total") != 741
    or shard_audit.get("shards_written") != 741
    or shard_audit.get("shards_reused_verified") != 0
    or shard_audit.get("dataset_shard_counts")
    != {"brainweb": 696, "hecktor": 45}
    or shard_audit.get("dataset_result_rows")
    != {"brainweb": 6_264, "hecktor": 0}
    or shard_audit.get("incremental_candidate_count") != EXPECTED_CANDIDATES
    or shard_audit.get("total_cases") != 8_892
    or shard_audit.get("total_result_rows") != EXPECTED_ROWS
    or shard_audit.get("total_operator_updates") != EXPECTED_OPERATOR_UPDATES
    or shard_audit.get("method_row_counts")
    != {"brainweb": {"MR-KEM": 6_264}, "hecktor": {}}
    or shard_audit.get("candidate_domain_verified") is not True
    or shard_audit.get("metrics_finite") is not True
    or shard_audit.get("empty_candidate_domain_verified") is not True
    or shard_audit.get("zero_row_shard_count") != 45
    or shard_audit.get("published_file_counts", {}).get("shard_artifacts") != 2_223
    or shard_audit.get("merge_created") is not False
    or shard_audit.get("union_selection_audit_created") is not False
    or shard_audit.get("frozen_hyperparameters_created") is not False
    or shard_audit.get("test_locked_access") != "prohibited"
    or protocol.kind != "hyperparameter_sweep"
    or protocol.split != "validation"
    or registration != union_audit.get("prior_registration")
    or file_sha256(original_merged_path) != EXPECTED_ORIGINAL_MERGED_CSV_SHA256
    or file_sha256(round1_merged_path) != EXPECTED_ROUND1_MERGED_CSV_SHA256
    or file_sha256(round2_merged_path) != EXPECTED_ROUND2_MERGED_CSV_SHA256
    or file_sha256(round3_merged_path) != EXPECTED_ROUND3_MERGED_CSV_SHA256
    or file_sha256(round4_merged_path) != EXPECTED_ROUND4_MERGED_CSV_SHA256
    or file_sha256(round5_merged_path) != EXPECTED_ROUND5_MERGED_CSV_SHA256
    or shard_dir.relative_to(root).as_posix() != SHARD_NAMESPACE
    or output_dir.relative_to(root).as_posix() != OUTPUT_NAMESPACE
    or frozen_path.exists()
  ):
    raise ValueError("reviewed round-6 audit or merge inputs are inconsistent")

  candidates = round6_candidate_map(round6_plan)
  cases = load_case_manifest(root, case_manifest_path, split="validation")
  validate_case_contract(cases, contract)
  shards = build_round6_shards(cases, candidates)
  execution_path = shard_dir / "execution_record.json"
  execution_record_sha256 = file_sha256(execution_path)
  if execution_record_sha256 != shard_audit.get("execution_record_sha256"):
    raise ValueError("round-6 execution record changed after audit")
  execution_commit = str(shard_audit.get("execution_git_commit", ""))
  verified = []
  for shard in shards:
    item = verify_incremental_shard(
      shard_dir, shard, candidates,
      execution_commit=execution_commit,
      execution_record_sha256=execution_record_sha256,
      plan_sha256=REVIEWED_ROUND6_PLAN_SHA256,
      run_id=contract.run_id,
    )
    if item is None:
      raise FileNotFoundError(f"reviewed round-6 shard is absent: {shard['shard_id']}")
    verified.append(item)
  shard_set_sha256 = _shard_set_sha256(verified)
  if shard_set_sha256 != REVIEWED_ROUND6_SHARD_SET_SHA256:
    raise ValueError("current round-6 shard bytes differ from the reviewed set")
  completion_path = shard_dir / "shard_batch_SUCCESS.json"
  if file_sha256(completion_path) != shard_audit.get("shard_batch_success_sha256"):
    raise ValueError("round-6 completion marker changed after audit")
  transaction_count = _verify_transaction_inventory(shard_dir, shard_audit)

  output_dir.parent.mkdir(parents=True, exist_ok=True)
  transaction_dir = Path(tempfile.mkdtemp(
    prefix=".hyperparameter_expansion_round6.merge.", dir=output_dir.parent,
  ))
  merged_path = transaction_dir / ROUND6_MERGED_NAME
  fragment_paths = [_paths(shard_dir, shard)[0] for shard in shards]
  stream_merge_round6_fragments(fragment_paths, merged_path)
  merged_sha256 = file_sha256(merged_path)
  manifest_path_out = merged_path.with_suffix(merged_path.suffix + ".manifest.json")
  atomic_write_json(manifest_path_out, {
    "schema_version": 1,
    "schema_name": "validation_hyperparameter_expansion_round6_v1",
    "artifact_path": merged_path.name,
    "artifact_sha256": merged_sha256,
    "byte_size": merged_path.stat().st_size,
    "row_count": EXPECTED_ROWS,
    "columns": SWEEP_FIELDS,
    "primary_key": PRIMARY_KEY,
    "split": "validation",
    "source_shard_count": 741,
    "source_shard_set_sha256": shard_set_sha256,
    "zero_row_source_shard_count": 45,
    "completed_utc": datetime.now(timezone.utc).isoformat(),
  })
  manifest_sha256 = file_sha256(manifest_path_out)
  record_path = transaction_dir / "merge_record.json"
  merge_record = {
    "artifact_kind": "validation_hyperparameter_expansion_round6_shard_merge",
    "schema_version": 1,
    "status": "complete_pending_round6_union_selection_audit",
    "merge_git_commit": merge_commit,
    "execution_git_commit": execution_commit,
    "round6_shard_audit_path": shard_audit_path.relative_to(root).as_posix(),
    "round6_shard_audit_sha256": REVIEWED_ROUND6_SHARD_AUDIT_SHA256,
    "round5_plan_sha256": REVIEWED_ROUND5_PLAN_SHA256,
    "round5_union_selection_audit_sha256": (
      REVIEWED_ROUND5_UNION_SELECTION_AUDIT_SHA256
    ),
    "round5_incremental_merge_audit_sha256": REVIEWED_ROUND5_MERGE_AUDIT_SHA256,
    "round6_plan_sha256": REVIEWED_ROUND6_PLAN_SHA256,
    "round6_smoke_audit_sha256": REVIEWED_ROUND6_SMOKE_AUDIT_SHA256,
    "round6_shard_set_sha256": shard_set_sha256,
    "source_shards": 741,
    "source_cases": 8_892,
    "round6_incremental_candidate_count": EXPECTED_CANDIDATES,
    "round6_incremental_result_rows": EXPECTED_ROWS,
    "round6_incremental_operator_updates": EXPECTED_OPERATOR_UPDATES,
    "round6_merged_csv_sha256": merged_sha256,
    "round6_merged_csv_byte_size": merged_path.stat().st_size,
    "round6_merged_manifest_sha256": manifest_sha256,
    "original_merged_csv_sha256": file_sha256(original_merged_path),
    "round1_merged_csv_sha256": file_sha256(round1_merged_path),
    "round2_merged_csv_sha256": file_sha256(round2_merged_path),
    "round3_merged_csv_sha256": file_sha256(round3_merged_path),
    "round4_merged_csv_sha256": file_sha256(round4_merged_path),
    "round5_merged_csv_sha256": file_sha256(round5_merged_path),
    "next_union_expected_result_rows": NEXT_UNION_RESULT_ROWS,
    "method_row_counts": shard_audit["method_row_counts"],
    "empty_candidate_domain_verified": True,
    "zero_row_source_shard_count": 45,
    "global_primary_key_uniqueness": shard_audit["global_primary_key_uniqueness"],
    "temporary_transaction_directories_preserved": transaction_count,
    "merge_order": "round6_shard_id_order_then_existing_fragment_row_order",
    "physical_union_created": False,
    "round6_union_selection_audit_created": False,
    "frozen_hyperparameters_created": False,
    "test_locked_access": "prohibited",
  }
  atomic_write_json(record_path, merge_record)
  record_sha256 = file_sha256(record_path)
  success_path = transaction_dir / "merge_SUCCESS.json"
  atomic_write_json(success_path, {
    "artifact_kind": "validation_hyperparameter_expansion_round6_merge_success",
    "schema_version": 1,
    "status": "complete_pending_round6_union_selection_audit",
    "round6_merged_csv_sha256": merged_sha256,
    "round6_merged_manifest_sha256": manifest_sha256,
    "merge_record_sha256": record_sha256,
    "round6_shard_set_sha256": shard_set_sha256,
    "round6_incremental_result_rows": EXPECTED_ROWS,
    "next_union_expected_result_rows": NEXT_UNION_RESULT_ROWS,
    "empty_candidate_domain_verified": True,
    "zero_row_source_shard_count": 45,
    "physical_union_created": False,
    "round6_union_selection_audit_created": False,
    "frozen_hyperparameters_created": False,
    "test_locked_access": "prohibited",
  })
  success_sha256 = file_sha256(success_path)
  os.replace(transaction_dir, output_dir)

  published_merged = output_dir / merged_path.name
  published_manifest = output_dir / manifest_path_out.name
  published_record = output_dir / record_path.name
  published_success = output_dir / success_path.name
  if (
    file_sha256(published_merged) != merged_sha256
    or file_sha256(published_manifest) != manifest_sha256
    or file_sha256(published_record) != record_sha256
    or file_sha256(published_success) != success_sha256
    or file_sha256(original_merged_path) != EXPECTED_ORIGINAL_MERGED_CSV_SHA256
    or file_sha256(round1_merged_path) != EXPECTED_ROUND1_MERGED_CSV_SHA256
    or file_sha256(round2_merged_path) != EXPECTED_ROUND2_MERGED_CSV_SHA256
    or file_sha256(round3_merged_path) != EXPECTED_ROUND3_MERGED_CSV_SHA256
    or file_sha256(round4_merged_path) != EXPECTED_ROUND4_MERGED_CSV_SHA256
    or file_sha256(round5_merged_path) != EXPECTED_ROUND5_MERGED_CSV_SHA256
    or frozen_path.exists()
  ):
    raise ValueError("published round-6 merge changed during transactional rename")
  result = {
    **merge_record,
    "status": "PASS_complete_pending_round6_union_selection_audit",
    "round6_merged_csv_path": published_merged.relative_to(root).as_posix(),
    "round6_merged_manifest_path": published_manifest.relative_to(root).as_posix(),
    "merge_record_path": published_record.relative_to(root).as_posix(),
    "merge_success_path": published_success.relative_to(root).as_posix(),
    "merge_success_sha256": success_sha256,
    "formal_merge_artifact_count": 4,
  }
  atomic_write_json(audit_path, result)
  return result
