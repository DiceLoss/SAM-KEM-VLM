"""Transactional merge gate for reviewed incremental validation shards."""

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
from .sweep_incremental import (
    EXPECTED_OPERATOR_UPDATES,
    EXPECTED_ROWS,
    EXPECTED_SELECTION_AUDIT_SHA256,
    REVIEWED_EXPANSION_PLAN_SHA256,
    REVIEWED_SMOKE_AUDIT_SHA256,
    build_incremental_shards,
    incremental_candidate_map,
    verify_incremental_shard,
)
from .sweep_merge import stream_merge_csv_fragments
from .sweep_sharded import (
    PRIMARY_KEY,
    _paths,
    _shard_set_sha256,
    line_ending_equivalent_sha256,
)

REVIEWED_INCREMENTAL_SHARD_AUDIT_SHA256 = (
    "717d328b6be6b89cdfeb57005503a28492e8c540f6a9837df6be4bae47e89c4a"
)
REVIEWED_INCREMENTAL_SHARD_SET_SHA256 = (
    "a8bb699f80767b4fc628969cb975e67fcf8b44bd8e13a2570604c0790665f16b"
)
OUTPUT_NAMESPACE = "runs/dev_trainmax_v1/hyperparameter_expansion"
INCREMENTAL_MERGED_NAME = "validation_hyperparameter_expansion.csv"
UNION_RESULT_ROWS = 5_157_360 + EXPECTED_ROWS


def _json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected a JSON object: {path}")
    return value


def _clean_commit(root: Path) -> str:
    status = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if status:
        raise RuntimeError(
            f"incremental merge requires a clean worktree: {status.splitlines()}"
        )
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def stream_merge_incremental_fragments(
    fragment_paths: list[Path],
    output_path: Path,
    *,
    expected_rows: int = EXPECTED_ROWS,
) -> dict:
    """Merge incremental fragments and enforce the reviewed result-row count."""
    details = stream_merge_csv_fragments(fragment_paths, output_path)
    if int(details["row_count"]) != expected_rows:
        raise ValueError(
            f"incremental merge has {details['row_count']} rows, expected {expected_rows}"
        )
    return details


def _verify_transaction_inventory(shard_dir: Path, audit: dict) -> int:
    block = audit.get("temporary_transaction_directories", {})
    if block.get("disposition") != (
        "preserved_read_only_excluded_from_published_shard_set"
    ):
        raise ValueError(
            "incremental shard audit has an invalid transaction disposition"
        )
    entries = block.get("entries", [])
    if int(block.get("count", -1)) != len(entries):
        raise ValueError("incremental transaction inventory count is inconsistent")
    for entry in entries:
        directory = shard_dir / entry["path"]
        expected = {
            item["path"]: (int(item["byte_size"]), item["sha256"])
            for item in entry["files"]
        }
        actual = {
            path.relative_to(directory).as_posix(): (
                path.stat().st_size,
                file_sha256(path),
            )
            for path in directory.rglob("*")
            if path.is_file()
        }
        if actual != expected:
            raise ValueError(
                f"incremental transaction evidence changed: {entry['path']}"
            )
    return len(entries)


def merge_audited_incremental_validation_sweep(
    root: Path,
    config_path: Path,
    protocol_path: Path,
    selection_audit_path: Path,
    plan_path: Path,
    smoke_audit_path: Path,
    shard_dir: Path,
    shard_audit_path: Path,
    output_dir: Path,
    audit_path: Path,
) -> dict:
    """Merge the reviewed incremental shell without union selection or freezing."""
    root = root.resolve()
    config_path = config_path.resolve()
    protocol_path = protocol_path.resolve()
    selection_audit_path = selection_audit_path.resolve()
    plan_path = plan_path.resolve()
    smoke_audit_path = smoke_audit_path.resolve()
    shard_dir = shard_dir.resolve()
    shard_audit_path = shard_audit_path.resolve()
    output_dir = output_dir.resolve()
    audit_path = audit_path.resolve()
    if output_dir.exists() or audit_path.exists():
        raise FileExistsError("refusing to overwrite incremental merge output or audit")
    merge_commit = _clean_commit(root)
    if file_sha256(shard_audit_path) != REVIEWED_INCREMENTAL_SHARD_AUDIT_SHA256:
        raise ValueError(
            "incremental shard audit is not the researcher-reviewed artifact"
        )
    if (
        file_sha256(selection_audit_path) != EXPECTED_SELECTION_AUDIT_SHA256
        or file_sha256(plan_path) != REVIEWED_EXPANSION_PLAN_SHA256
        or file_sha256(smoke_audit_path) != REVIEWED_SMOKE_AUDIT_SHA256
    ):
        raise ValueError("incremental merge inputs are not the reviewed artifacts")

    selection_audit = _json(selection_audit_path)
    plan = _json(plan_path)
    smoke_audit = _json(smoke_audit_path)
    shard_audit = _json(shard_audit_path)
    cfg, contract = load_contract(config_path)
    require_development_execution(cfg)
    registration = validate_registered_validation_prior_configuration(
        root,
        config_path,
        cfg,
    )
    protocol = load_protocol(protocol_path)
    manifest_path = root / str(cfg["outputs"]["validation_manifest"])
    original_merged_path = root / str(cfg["outputs"]["hyperparameter_sweep"])
    frozen_path = root / str(cfg["outputs"]["frozen_hyperparameters"])
    if (
        selection_audit.get("status") != "BLOCKED_grid_boundary_expansion_required"
        or plan.get("status") != "ready_for_researcher_review_no_execution"
        or plan.get("selection_audit_sha256") != EXPECTED_SELECTION_AUDIT_SHA256
        or plan.get("incremental_candidate_count") != 76
        or plan.get("expected_incremental_result_rows") != EXPECTED_ROWS
        or plan.get("expected_incremental_operator_updates")
        != EXPECTED_OPERATOR_UPDATES
        or smoke_audit.get("status") != "PASS"
        or smoke_audit.get("plan_sha256") != REVIEWED_EXPANSION_PLAN_SHA256
        or smoke_audit.get("selection_audit_sha256") != EXPECTED_SELECTION_AUDIT_SHA256
        or shard_audit.get("status") != "PASS_complete_pending_incremental_merge"
        or shard_audit.get("read_only") is not True
        or shard_audit.get("selection_audit_sha256") != EXPECTED_SELECTION_AUDIT_SHA256
        or shard_audit.get("plan_sha256") != REVIEWED_EXPANSION_PLAN_SHA256
        or shard_audit.get("smoke_audit_sha256") != REVIEWED_SMOKE_AUDIT_SHA256
        or shard_audit.get("config_sha256") != file_sha256(config_path)
        or not line_ending_equivalent_sha256(
            protocol_path,
            str(shard_audit.get("protocol_sha256")),
        )
        or shard_audit.get("manifest_sha256") != file_sha256(manifest_path)
        or shard_audit.get("merged_csv_sha256") != file_sha256(original_merged_path)
        or shard_audit.get("shard_set_sha256") != REVIEWED_INCREMENTAL_SHARD_SET_SHA256
        or shard_audit.get("shards_total") != 741
        or shard_audit.get("shards_written") != 741
        or shard_audit.get("shards_reused_verified") != 0
        or shard_audit.get("total_cases") != 8_892
        or shard_audit.get("total_result_rows") != EXPECTED_ROWS
        or shard_audit.get("total_operator_updates") != EXPECTED_OPERATOR_UPDATES
        or shard_audit.get("published_file_counts", {}).get("shard_artifacts") != 2_223
        or shard_audit.get("merge_created") is not False
        or shard_audit.get("frozen_hyperparameters_created") is not False
        or shard_audit.get("test_locked_access") != "prohibited"
        or protocol.kind != "hyperparameter_sweep"
        or protocol.split != "validation"
        or registration != selection_audit.get("prior_registration")
        or file_sha256(original_merged_path) != plan.get("merged_csv_sha256")
        or output_dir.relative_to(root).as_posix() != OUTPUT_NAMESPACE
        or frozen_path.exists()
    ):
        raise ValueError("reviewed incremental audit or merge inputs are inconsistent")

    candidates = incremental_candidate_map(plan)
    cases = load_case_manifest(root, manifest_path, split="validation")
    validate_case_contract(cases, contract)
    shards = build_incremental_shards(cases, candidates)
    execution_path = shard_dir / "execution_record.json"
    execution_record_sha256 = file_sha256(execution_path)
    if execution_record_sha256 != shard_audit.get("execution_record_sha256"):
        raise ValueError("incremental execution record changed after audit")
    execution_commit = str(shard_audit.get("execution_git_commit", ""))
    verified = []
    for shard in shards:
        item = verify_incremental_shard(
            shard_dir,
            shard,
            candidates,
            execution_commit=execution_commit,
            execution_record_sha256=execution_record_sha256,
            plan_sha256=REVIEWED_EXPANSION_PLAN_SHA256,
            run_id=contract.run_id,
        )
        if item is None:
            raise FileNotFoundError(
                f"reviewed incremental shard is absent: {shard['shard_id']}"
            )
        verified.append(item)
    shard_set_sha256 = _shard_set_sha256(verified)
    if shard_set_sha256 != REVIEWED_INCREMENTAL_SHARD_SET_SHA256:
        raise ValueError("current incremental shard bytes differ from the reviewed set")
    completion_path = shard_dir / "shard_batch_SUCCESS.json"
    if file_sha256(completion_path) != shard_audit.get("shard_batch_success_sha256"):
        raise ValueError("incremental completion marker changed after audit")
    transaction_count = _verify_transaction_inventory(shard_dir, shard_audit)

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    transaction_dir = Path(
        tempfile.mkdtemp(
            prefix=".hyperparameter_expansion.merge.",
            dir=output_dir.parent,
        )
    )
    merged_path = transaction_dir / INCREMENTAL_MERGED_NAME
    fragment_paths = [_paths(shard_dir, shard)[0] for shard in shards]
    stream_merge_incremental_fragments(fragment_paths, merged_path)
    merged_sha256 = file_sha256(merged_path)
    manifest_path_out = merged_path.with_suffix(merged_path.suffix + ".manifest.json")
    atomic_write_json(
        manifest_path_out,
        {
            "schema_version": 1,
            "schema_name": "validation_hyperparameter_expansion_v1",
            "artifact_path": merged_path.name,
            "artifact_sha256": merged_sha256,
            "byte_size": merged_path.stat().st_size,
            "row_count": EXPECTED_ROWS,
            "columns": SWEEP_FIELDS,
            "primary_key": PRIMARY_KEY,
            "split": "validation",
            "source_shard_count": 741,
            "source_shard_set_sha256": shard_set_sha256,
            "completed_utc": datetime.now(timezone.utc).isoformat(),
        },
    )
    manifest_sha256 = file_sha256(manifest_path_out)
    record_path = transaction_dir / "merge_record.json"
    merge_record = {
        "artifact_kind": "validation_hyperparameter_expansion_shard_merge",
        "schema_version": 1,
        "status": "complete_pending_union_selection_audit",
        "merge_git_commit": merge_commit,
        "execution_git_commit": execution_commit,
        "incremental_shard_audit_path": shard_audit_path.relative_to(root).as_posix(),
        "incremental_shard_audit_sha256": REVIEWED_INCREMENTAL_SHARD_AUDIT_SHA256,
        "selection_audit_sha256": EXPECTED_SELECTION_AUDIT_SHA256,
        "plan_sha256": REVIEWED_EXPANSION_PLAN_SHA256,
        "smoke_audit_sha256": REVIEWED_SMOKE_AUDIT_SHA256,
        "incremental_shard_set_sha256": shard_set_sha256,
        "source_shards": 741,
        "source_cases": 8_892,
        "incremental_result_rows": EXPECTED_ROWS,
        "incremental_operator_updates": EXPECTED_OPERATOR_UPDATES,
        "incremental_merged_csv_sha256": merged_sha256,
        "incremental_merged_csv_byte_size": merged_path.stat().st_size,
        "incremental_merged_manifest_sha256": manifest_sha256,
        "original_merged_csv_path": original_merged_path.relative_to(root).as_posix(),
        "original_merged_csv_sha256": file_sha256(original_merged_path),
        "union_expected_result_rows": UNION_RESULT_ROWS,
        "method_row_counts": shard_audit["method_row_counts"],
        "global_primary_key_uniqueness": shard_audit["global_primary_key_uniqueness"],
        "temporary_transaction_directories_preserved": transaction_count,
        "merge_order": "incremental_shard_id_order_then_existing_fragment_row_order",
        "union_selection_audit_created": False,
        "frozen_hyperparameters_created": False,
        "test_locked_access": "prohibited",
    }
    atomic_write_json(record_path, merge_record)
    record_sha256 = file_sha256(record_path)
    success_path = transaction_dir / "merge_SUCCESS.json"
    atomic_write_json(
        success_path,
        {
            "artifact_kind": "validation_hyperparameter_expansion_merge_success",
            "schema_version": 1,
            "status": "complete_pending_union_selection_audit",
            "incremental_merged_csv_sha256": merged_sha256,
            "incremental_merged_manifest_sha256": manifest_sha256,
            "merge_record_sha256": record_sha256,
            "incremental_shard_set_sha256": shard_set_sha256,
            "incremental_result_rows": EXPECTED_ROWS,
            "union_expected_result_rows": UNION_RESULT_ROWS,
            "union_selection_audit_created": False,
            "frozen_hyperparameters_created": False,
            "test_locked_access": "prohibited",
        },
    )
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
        or file_sha256(original_merged_path) != plan.get("merged_csv_sha256")
        or frozen_path.exists()
    ):
        raise ValueError(
            "published incremental merge changed during transactional rename"
        )
    result = {
        **merge_record,
        "status": "PASS_complete_pending_union_selection_audit",
        "incremental_merged_csv_path": published_merged.relative_to(root).as_posix(),
        "incremental_merged_manifest_path": published_manifest.relative_to(
            root
        ).as_posix(),
        "merge_record_path": published_record.relative_to(root).as_posix(),
        "merge_success_path": published_success.relative_to(root).as_posix(),
        "merge_success_sha256": success_sha256,
        "formal_merge_artifact_count": 4,
    }
    atomic_write_json(audit_path, result)
    return result
