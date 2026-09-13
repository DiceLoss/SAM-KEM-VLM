"""SHA-bound publication of reviewed validation hyperparameters."""

from __future__ import annotations

import math
from pathlib import Path

from .artifacts import (
  atomic_csv_writer,
  atomic_write_json,
  file_sha256,
  finalize_csv,
  write_success_marker,
)
from .config import load_contract, require_development_execution
from .prior_registration import validate_registered_validation_prior_configuration
from .protocols import load_protocol
from .sweep_incremental_merge import INCREMENTAL_MERGED_NAME
from .sweep_incremental_round2_merge import (
  OUTPUT_NAMESPACE as ROUND2_MERGE_NAMESPACE,
  ROUND2_MERGED_NAME,
)
from .sweep_incremental_round3_merge import (
  OUTPUT_NAMESPACE as ROUND3_MERGE_NAMESPACE,
  ROUND3_MERGED_NAME,
)
from .sweep_incremental_round4_merge import (
  OUTPUT_NAMESPACE as ROUND4_MERGE_NAMESPACE,
  ROUND4_MERGED_NAME,
)
from .sweep_incremental_round5_merge import (
  OUTPUT_NAMESPACE as ROUND5_MERGE_NAMESPACE,
  ROUND5_MERGED_NAME,
)
from .sweep_incremental_round6_merge import (
  OUTPUT_NAMESPACE as ROUND6_MERGE_NAMESPACE,
  ROUND6_MERGED_NAME,
)
from .sweep_sharded import line_ending_equivalent_sha256
from .sweep_union_selection import _clean_commit, _json

REVIEWED_FINAL_SELECTION_AUDIT_SHA256 = (
  "36dd6d9df4e822f76a6e8521377f73a52b4d09a84553f0eeb788b8fa8f53c9f5"
)
EXPECTED_UNION_ROWS = 5_355_873
EXPECTED_SUBJECT_AGGREGATES = 116_879
EXPECTED_SELECTIONS = 48
FROZEN_FIELDS = [
  "run_id", "dataset", "count_level", "method", "sigma", "radius",
  "neighbours", "iterations", "alpha", "validation_subject_mean_nrmse",
  "validation_subject_mean_ssim",
]


def _validated_rows(
  selection_audit: dict, protocol, run_id: str, count_levels: tuple[int, ...],
) -> list[dict]:
  rows = selection_audit.get("selected_hyperparameters")
  if not isinstance(rows, list) or len(rows) != EXPECTED_SELECTIONS:
    raise ValueError("freeze requires exactly 48 reviewed selections")
  allowed_methods = {
    "brainweb": tuple(method for method in protocol.methods if method != "CT-KEM"),
    "hecktor": tuple(method for method in protocol.methods if method != "MR-KEM"),
  }
  expected = {
    (dataset, count, method)
    for dataset, methods in allowed_methods.items()
    for count in count_levels
    for method in methods
  }
  actual = set()
  normalized = []
  for row in rows:
    if set(row) != set(FROZEN_FIELDS):
      raise ValueError("reviewed selection has an unexpected frozen schema")
    if row.get("run_id") != run_id:
      raise ValueError("reviewed selection belongs to a different run")
    identity = (str(row["dataset"]), int(row["count_level"]), str(row["method"]))
    if identity in actual:
      raise ValueError(f"duplicate frozen selection: {identity}")
    actual.add(identity)
    numeric = (
      float(row["sigma"]), float(row["radius"]), float(row["neighbours"]),
      float(row["iterations"]), float(row["alpha"]),
      float(row["validation_subject_mean_nrmse"]),
      float(row["validation_subject_mean_ssim"]),
    )
    if not all(math.isfinite(value) for value in numeric):
      raise ValueError(f"non-finite frozen selection: {identity}")
    if any(int(row[field]) < 0 for field in ("radius", "neighbours", "iterations")):
      raise ValueError(f"negative integer hyperparameter: {identity}")
    normalized.append({field: row[field] for field in FROZEN_FIELDS})
  if actual != expected:
    raise ValueError("reviewed selections do not cover the complete frozen domain")
  return normalized


def _source_paths(root: Path, original_path: Path) -> tuple[Path, ...]:
  return (
    original_path,
    root / "runs/dev_trainmax_v1/hyperparameter_expansion" / INCREMENTAL_MERGED_NAME,
    root / ROUND2_MERGE_NAMESPACE / ROUND2_MERGED_NAME,
    root / ROUND3_MERGE_NAMESPACE / ROUND3_MERGED_NAME,
    root / ROUND4_MERGE_NAMESPACE / ROUND4_MERGED_NAME,
    root / ROUND5_MERGE_NAMESPACE / ROUND5_MERGED_NAME,
    root / ROUND6_MERGE_NAMESPACE / ROUND6_MERGED_NAME,
  )


def _source_artifacts(source_paths: tuple[Path, ...]) -> list[Path]:
  artifacts = []
  for csv_path in source_paths:
    artifacts.extend((
      csv_path,
      csv_path.with_suffix(csv_path.suffix + ".manifest.json"),
      csv_path.parent / "merge_record.json",
      csv_path.parent / "merge_SUCCESS.json",
    ))
  return artifacts


def freeze_reviewed_hyperparameters(
  root: Path, config_path: Path, protocol_path: Path,
  selection_audit_path: Path, audit_path: Path, *, release_check_only: bool,
) -> dict:
  """Validate the terminal selection and optionally publish its frozen record."""
  root = root.resolve()
  config_path = config_path.resolve()
  protocol_path = protocol_path.resolve()
  selection_audit_path = selection_audit_path.resolve()
  audit_path = audit_path.resolve()
  freeze_commit = _clean_commit(root)
  if file_sha256(selection_audit_path) != REVIEWED_FINAL_SELECTION_AUDIT_SHA256:
    raise ValueError("final selection audit is not the researcher-reviewed artifact")
  cfg, contract = load_contract(config_path)
  require_development_execution(cfg)
  registration = validate_registered_validation_prior_configuration(
    root, config_path, cfg,
  )
  protocol = load_protocol(protocol_path)
  selection_audit = _json(selection_audit_path)
  original_path = root / str(cfg["outputs"]["hyperparameter_sweep"])
  frozen_path = root / str(cfg["outputs"]["frozen_hyperparameters"])
  frozen_manifest = frozen_path.with_suffix(frozen_path.suffix + ".manifest.json")
  completion_path = original_path.parent / "_SUCCESS.json"
  if any(path.exists() for path in (frozen_path, frozen_manifest, completion_path, audit_path)):
    raise FileExistsError("refusing to overwrite a freeze or validation completion artifact")
  if (
    protocol.kind != "hyperparameter_sweep"
    or protocol.split != "validation"
    or protocol.raw.get("test_locked_access") != "prohibited"
    or selection_audit.get("status") != "PASS_ready_for_hyperparameter_freeze"
    or selection_audit.get("read_only") is not True
    or selection_audit.get("config_sha256") != file_sha256(config_path)
    or not line_ending_equivalent_sha256(
      protocol_path, str(selection_audit.get("protocol_sha256")),
    )
    or selection_audit.get("prior_registration") != registration
    or selection_audit.get("union_rows_streamed") != EXPECTED_UNION_ROWS
    or selection_audit.get("union_subject_configuration_aggregates")
    != EXPECTED_SUBJECT_AGGREGATES
    or selection_audit.get("selection_count") != EXPECTED_SELECTIONS
    or selection_audit.get("dataset_selection_counts")
    != {"brainweb": 24, "hecktor": 24}
    or selection_audit.get("reviewed_boundary_group_count") != 3
    or selection_audit.get("unchanged_prior_union_selection_exact_match_count") != 45
    or selection_audit.get("source_boundary_winner_count") != 3
    or selection_audit.get("round6_incremental_candidate_selected_count") != 0
    or selection_audit.get("expanded_boundary_selection_count") != 0
    or selection_audit.get("expanded_boundary_axis_counts") != {}
    or selection_audit.get("expanded_boundary_selections") != []
    or selection_audit.get("union_csv_created") is not False
    or selection_audit.get("frozen_hyperparameters_created") is not False
    or selection_audit.get("test_locked_access") != "prohibited"
  ):
    raise ValueError("terminal selection audit is not eligible for freezing")
  rows = _validated_rows(
    selection_audit, protocol, contract.run_id, contract.count_levels,
  )
  source_paths = _source_paths(root, original_path)
  source_artifacts = _source_artifacts(source_paths)
  if not all(path.is_file() for path in source_artifacts):
    raise FileNotFoundError("one or more reviewed seven-source artifacts are missing")
  labels = ("original", "round1", "round2", "round3", "round4", "round5", "round6")
  if any(
    file_sha256(path) != selection_audit.get(f"{label}_merged_csv_sha256")
    for label, path in zip(labels, source_paths)
  ):
    raise ValueError("a selected source CSV changed after the final audit")
  result = {
    "artifact_kind": "validation_hyperparameter_freeze",
    "schema_version": 1,
    "status": (
      "PASS_ready_for_hyperparameter_freeze_execution"
      if release_check_only else "PASS_hyperparameters_frozen_pending_locked_config"
    ),
    "freeze_git_commit": freeze_commit,
    "selection_audit_sha256": REVIEWED_FINAL_SELECTION_AUDIT_SHA256,
    "config_sha256": file_sha256(config_path),
    "protocol_sha256": file_sha256(protocol_path),
    "prior_registration": registration,
    "source_csv_sha256": {
      label: file_sha256(path) for label, path in zip(labels, source_paths)
    },
    "source_artifact_count": len(source_artifacts),
    "selection_count": len(rows),
    "dataset_selection_counts": {"brainweb": 24, "hecktor": 24},
    "expanded_boundary_selection_count": 0,
    "release_check_only": release_check_only,
    "formal_output_created": False,
    "frozen_hyperparameters_created": False,
    "validation_completion_created": False,
    "locked_config_created": False,
    "immutable_archive_uri_modified": False,
    "test_locked_access": "prohibited",
  }
  if release_check_only:
    return result

  frozen_path.parent.mkdir(parents=True, exist_ok=True)
  with atomic_csv_writer(frozen_path, FROZEN_FIELDS) as writer:
    writer.writerows(rows)
  finalize_csv(
    frozen_path, schema_name="frozen_hyperparameters_v2",
    primary_key=["run_id", "dataset", "count_level", "method"],
    split="validation",
  )
  completion_artifacts = [
    *source_artifacts, selection_audit_path, frozen_path, frozen_manifest,
  ]
  write_success_marker(original_path.parent, completion_artifacts)
  result.update({
    "status": "PASS_hyperparameters_frozen_pending_locked_config",
    "formal_output_created": True,
    "frozen_hyperparameters_created": True,
    "validation_completion_created": True,
    "frozen_hyperparameters_path": frozen_path.relative_to(root).as_posix(),
    "frozen_hyperparameters_sha256": file_sha256(frozen_path),
    "frozen_manifest_path": frozen_manifest.relative_to(root).as_posix(),
    "frozen_manifest_sha256": file_sha256(frozen_manifest),
    "validation_completion_path": completion_path.relative_to(root).as_posix(),
    "validation_completion_sha256": file_sha256(completion_path),
    "validation_completion_artifact_count": len(completion_artifacts),
  })
  atomic_write_json(audit_path, result)
  return result
