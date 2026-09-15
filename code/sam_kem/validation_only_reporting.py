"""Build an explicitly selection-aware validation-only reporting package."""

from __future__ import annotations

import csv
import math
import os
import subprocess
import tempfile
from collections import Counter, defaultdict
from pathlib import Path

from .artifacts import (
  atomic_csv_writer,
  atomic_write_json,
  file_sha256,
  finalize_csv,
  verify_csv_artifact,
  verify_success_marker,
  write_success_marker,
)
from .config import load_contract, require_development_execution
from .hyperparameter_freeze import FROZEN_FIELDS, _source_paths, _validated_rows
from .protocols import load_protocol
from .splits import load_configured_subject_splits, split_counts
from .sweep_evaluation import SWEEP_FIELDS
from .sweep_sharded import PRIMARY_KEY
from .sweep_union_selection import _json, _parameter_key


REVIEWED_FINAL_SELECTION_AUDIT_SHA256 = (
  "36dd6d9df4e822f76a6e8521377f73a52b4d09a84553f0eeb788b8fa8f53c9f5"
)
REVIEWED_FREEZE_AUDIT_SHA256 = (
  "13bb3e7d45336c824efb4d6f5110dbcf7fd43b41552960a77174b21dfa779510"
)
REVIEWED_FROZEN_HYPERPARAMETERS_SHA256 = (
  "4b7624c0339b4e387a7c7a1e52ed359f757a89516de437a6a8f73871f12c204e"
)
EXPECTED_SOURCE_ROWS = 5_355_873
EXPECTED_SELECTED_ROWS = 53_352
EXPECTED_SUBJECT_AGGREGATES = 1_176
SOURCE_LABELS = ("original", "round1", "round2", "round3", "round4", "round5", "round6")
PER_CASE_FIELDS = ["source_label", *SWEEP_FIELDS]
SUBJECT_FIELDS = [
  "run_id", "dataset", "subject_id", "count_level", "method", "sigma", "radius",
  "neighbours", "iterations", "alpha", "observation_count", "mean_nrmse", "mean_ssim",
  "reporting_scope", "selection_and_reporting_same_split",
]
SUMMARY_FIELDS = [
  "run_id", "dataset", "count_level", "method", "sigma", "radius", "neighbours",
  "iterations", "alpha", "subject_count", "case_row_count", "subject_mean_nrmse",
  "subject_mean_ssim", "reporting_scope", "selection_and_reporting_same_split",
]
REPORTING_SCOPE = "development_validation_only_no_untouched_test_cohort"


class _CompensatedMetrics:
  __slots__ = ("count", "nrmse_compensation", "nrmse_sum", "ssim_compensation", "ssim_sum")

  def __init__(self) -> None:
    self.count = 0
    self.nrmse_sum = 0.0
    self.nrmse_compensation = 0.0
    self.ssim_sum = 0.0
    self.ssim_compensation = 0.0

  @staticmethod
  def _add(total: float, compensation: float, value: float) -> tuple[float, float]:
    corrected = value - compensation
    updated = total + corrected
    return updated, (updated - total) - corrected

  def add(self, nrmse: float, ssim: float) -> None:
    self.nrmse_sum, self.nrmse_compensation = self._add(
      self.nrmse_sum, self.nrmse_compensation, nrmse,
    )
    self.ssim_sum, self.ssim_compensation = self._add(
      self.ssim_sum, self.ssim_compensation, ssim,
    )
    self.count += 1


def _clean_commit(root: Path) -> str:
  status = subprocess.run(
    ["git", "status", "--porcelain"], cwd=root, check=True,
    capture_output=True, text=True,
  ).stdout.strip()
  if status:
    raise RuntimeError(f"validation-only reporting requires a clean worktree: {status.splitlines()}")
  return subprocess.run(
    ["git", "rev-parse", "HEAD"], cwd=root, check=True,
    capture_output=True, text=True,
  ).stdout.strip()


def _selected_parameter_map(
  selection_audit: dict, frozen_path: Path, protocol, run_id: str,
  count_levels: tuple[int, ...],
) -> dict[tuple[str, int, str], tuple[float, int, int, int, float]]:
  reviewed = _validated_rows(selection_audit, protocol, run_id, count_levels)
  with frozen_path.open("r", encoding="utf-8", newline="") as handle:
    reader = csv.DictReader(handle)
    if list(reader.fieldnames or ()) != FROZEN_FIELDS:
      raise ValueError("frozen hyperparameters have an unexpected schema")
    frozen = list(reader)
  if len(frozen) != len(reviewed):
    raise ValueError("frozen hyperparameters differ from the final reviewed selections")
  selected = {}
  for frozen_row, reviewed_row in zip(frozen, reviewed):
    identity = (
      frozen_row["dataset"], int(frozen_row["count_level"]), frozen_row["method"],
    )
    if identity != (
      str(reviewed_row["dataset"]), int(reviewed_row["count_level"]),
      str(reviewed_row["method"]),
    ):
      raise ValueError("frozen hyperparameter row order differs from the final selection audit")
    parameters = _parameter_key(frozen_row)
    reviewed_parameters = (
      float(reviewed_row["sigma"]), int(reviewed_row["radius"]),
      int(reviewed_row["neighbours"]), int(reviewed_row["iterations"]),
      float(reviewed_row["alpha"]),
    )
    if parameters != reviewed_parameters:
      raise ValueError(f"frozen parameters differ from the final selection: {identity}")
    if not math.isclose(
      float(frozen_row["validation_subject_mean_nrmse"]),
      float(reviewed_row["validation_subject_mean_nrmse"]), rel_tol=0.0, abs_tol=0.0,
    ) or not math.isclose(
      float(frozen_row["validation_subject_mean_ssim"]),
      float(reviewed_row["validation_subject_mean_ssim"]), rel_tol=0.0, abs_tol=0.0,
    ):
      raise ValueError(f"frozen validation score differs from the final selection: {identity}")
    if identity in selected:
      raise ValueError(f"duplicate frozen reporting identity: {identity}")
    selected[identity] = parameters
  return selected


def _validate_development_only_inventory(root: Path, cfg: dict) -> dict:
  split_rows = load_configured_subject_splits(root, cfg)
  counts = split_counts(split_rows)
  if any(row["split"] == "test_locked" for row in split_rows):
    raise ValueError("validation-only reporting refuses a split containing test_locked subjects")
  exposure_counts = Counter(row["previous_exposure_status"] for row in split_rows)
  if exposure_counts != {"development_allowed": len(split_rows)}:
    raise ValueError("validation-only scope requires the reviewed all-development inventory")
  source_manifest = root / str(cfg["data"]["source_activity_manifest"])
  with source_manifest.open("r", encoding="utf-8", newline="") as handle:
    source_rows = list(csv.DictReader(handle))
  if not source_rows or any(row.get("split") == "test_locked" for row in source_rows):
    raise ValueError("validation-only source manifest must be non-empty and contain no test rows")
  return {
    "subject_split_path": str(cfg["data"]["split_file"]),
    "subject_split_sha256": file_sha256(root / str(cfg["data"]["split_file"])),
    "subject_counts": counts,
    "subject_exposure_counts": dict(sorted(exposure_counts.items())),
    "test_locked_subject_count": 0,
    "source_manifest_path": str(cfg["data"]["source_activity_manifest"]),
    "source_manifest_sha256": file_sha256(source_manifest),
    "source_manifest_row_count": len(source_rows),
    "test_locked_source_row_count": 0,
  }


def _stream_selected_rows(
  source_paths: tuple[Path, ...], selected: dict, run_id: str,
  *, writer=None,
) -> tuple[int, int, dict, list[dict], list[dict]]:
  rows_streamed = 0
  rows_selected = 0
  keys = set()
  source_selected_counts = Counter()
  subject_metrics: dict[tuple, _CompensatedMetrics] = {}
  for label, path in zip(SOURCE_LABELS, source_paths):
    with path.open("r", encoding="utf-8", newline="") as handle:
      reader = csv.DictReader(handle)
      if list(reader.fieldnames or ()) != SWEEP_FIELDS:
        raise ValueError(f"{label} validation source has an unexpected schema")
      for row in reader:
        rows_streamed += 1
        identity = (row["dataset"], int(row["count_level"]), row["method"])
        parameters = _parameter_key(row)
        if row["run_id"] != run_id or row["split"] != "validation":
          raise ValueError(f"invalid validation source identity in {label} row {rows_streamed}")
        if selected.get(identity) != parameters:
          continue
        nrmse = float(row["nrmse"])
        ssim = float(row["ssim"])
        if not math.isfinite(nrmse) or not math.isfinite(ssim):
          raise ValueError(f"non-finite selected metric in {label} row {rows_streamed}")
        key = tuple(row[field] for field in PRIMARY_KEY)
        if key in keys:
          raise ValueError(f"selected reporting row appears in multiple reviewed sources: {key}")
        keys.add(key)
        rows_selected += 1
        source_selected_counts[label] += 1
        if writer is not None:
          writer.writerow({"source_label": label, **row})
        subject_key = (
          row["dataset"], row["subject_id"], int(row["count_level"]), row["method"],
          *parameters,
        )
        aggregate = subject_metrics.setdefault(subject_key, _CompensatedMetrics())
        aggregate.add(nrmse, ssim)
  subject_rows = []
  summary_groups: dict[tuple, list[dict]] = defaultdict(list)
  for key in sorted(subject_metrics):
    dataset, subject, count, method, sigma, radius, neighbours, iterations, alpha = key
    aggregate = subject_metrics[key]
    row = {
      "run_id": run_id, "dataset": dataset, "subject_id": subject,
      "count_level": count, "method": method, "sigma": sigma, "radius": radius,
      "neighbours": neighbours, "iterations": iterations, "alpha": alpha,
      "observation_count": aggregate.count,
      "mean_nrmse": aggregate.nrmse_sum / aggregate.count,
      "mean_ssim": aggregate.ssim_sum / aggregate.count,
      "reporting_scope": REPORTING_SCOPE,
      "selection_and_reporting_same_split": True,
    }
    subject_rows.append(row)
    summary_groups[(dataset, count, method, sigma, radius, neighbours, iterations, alpha)].append(row)
  summary_rows = []
  for key in sorted(summary_groups):
    dataset, count, method, sigma, radius, neighbours, iterations, alpha = key
    values = summary_groups[key]
    summary_rows.append({
      "run_id": run_id, "dataset": dataset, "count_level": count, "method": method,
      "sigma": sigma, "radius": radius, "neighbours": neighbours,
      "iterations": iterations, "alpha": alpha, "subject_count": len(values),
      "case_row_count": sum(int(row["observation_count"]) for row in values),
      "subject_mean_nrmse": math.fsum(float(row["mean_nrmse"]) for row in values) / len(values),
      "subject_mean_ssim": math.fsum(float(row["mean_ssim"]) for row in values) / len(values),
      "reporting_scope": REPORTING_SCOPE,
      "selection_and_reporting_same_split": True,
    })
  return (
    rows_streamed, rows_selected, dict(sorted(source_selected_counts.items())),
    subject_rows, summary_rows,
  )


def build_validation_only_reporting_package(
  root: Path, config_path: Path, protocol_path: Path, selection_audit_path: Path,
  freeze_audit_path: Path, output_dir: Path, audit_path: Path, *, release_check_only: bool,
) -> dict:
  """Verify terminal validation evidence and optionally publish reporting-only tables."""
  root = root.resolve()
  config_path = config_path.resolve()
  protocol_path = protocol_path.resolve()
  selection_audit_path = selection_audit_path.resolve()
  freeze_audit_path = freeze_audit_path.resolve()
  output_dir = output_dir.resolve()
  audit_path = audit_path.resolve()
  reporting_commit = _clean_commit(root)
  if output_dir.exists() or audit_path.exists():
    raise FileExistsError("refusing to overwrite validation-only reporting artifacts")
  if file_sha256(selection_audit_path) != REVIEWED_FINAL_SELECTION_AUDIT_SHA256:
    raise ValueError("final selection audit is not the researcher-reviewed artifact")
  if file_sha256(freeze_audit_path) != REVIEWED_FREEZE_AUDIT_SHA256:
    raise ValueError("freeze audit is not the researcher-reviewed artifact")
  cfg, contract = load_contract(config_path)
  require_development_execution(cfg)
  protocol = load_protocol(protocol_path)
  selection_audit = _json(selection_audit_path)
  freeze_audit = _json(freeze_audit_path)
  frozen_path = root / str(cfg["outputs"]["frozen_hyperparameters"])
  if file_sha256(frozen_path) != REVIEWED_FROZEN_HYPERPARAMETERS_SHA256:
    raise ValueError("frozen hyperparameters are not the researcher-reviewed artifact")
  frozen_manifest = verify_csv_artifact(frozen_path)
  original_path = root / str(cfg["outputs"]["hyperparameter_sweep"])
  completion = verify_success_marker(original_path.parent)
  if (
    selection_audit.get("status") != "PASS_ready_for_hyperparameter_freeze"
    or selection_audit.get("test_locked_access") != "prohibited"
    or freeze_audit.get("status") != "PASS_hyperparameters_frozen_pending_locked_config"
    or freeze_audit.get("selection_audit_sha256") != REVIEWED_FINAL_SELECTION_AUDIT_SHA256
    or freeze_audit.get("frozen_hyperparameters_sha256")
    != REVIEWED_FROZEN_HYPERPARAMETERS_SHA256
    or freeze_audit.get("config_sha256") != file_sha256(config_path)
    or frozen_manifest.get("schema_name") != "frozen_hyperparameters_v2"
    or frozen_manifest.get("split") != "validation"
    or frozen_manifest.get("row_count") != 48
    or cfg.get("locked") is not False
    or cfg.get("status") != "development_candidate_ready"
  ):
    raise ValueError("terminal validation freeze evidence is not eligible for reporting")
  protocol_raw = protocol.raw
  if (
    protocol.kind != "hyperparameter_sweep" or protocol.split != "validation"
    or protocol_raw.get("test_locked_access") != "prohibited"
  ):
    raise ValueError("validation-only reporting requires the frozen validation protocol")
  inventory = _validate_development_only_inventory(root, cfg)
  selected = _selected_parameter_map(
    selection_audit, frozen_path, protocol, contract.run_id, contract.count_levels,
  )
  source_paths = _source_paths(root, original_path)
  if any(
    not path.is_file() or file_sha256(path) != freeze_audit["source_csv_sha256"][label]
    for label, path in zip(SOURCE_LABELS, source_paths)
  ):
    raise ValueError("one or more reviewed validation source CSVs are missing or changed")
  rows_streamed, rows_selected, by_source, subject_rows, summary_rows = _stream_selected_rows(
    source_paths, selected, contract.run_id,
  )
  expected_subject_counts = {"brainweb": 4, "hecktor": 45}
  if (
    rows_streamed != EXPECTED_SOURCE_ROWS or rows_selected != EXPECTED_SELECTED_ROWS
    or len(subject_rows) != EXPECTED_SUBJECT_AGGREGATES or len(summary_rows) != 48
    or any(
      int(row["subject_count"]) != expected_subject_counts[row["dataset"]]
      for row in summary_rows
    )
  ):
    raise ValueError("selected validation reporting domain is incomplete")
  frozen_scores = {
    (row["dataset"], int(row["count_level"]), row["method"]): (
      float(row["validation_subject_mean_nrmse"]),
      float(row["validation_subject_mean_ssim"]),
    ) for row in selection_audit["selected_hyperparameters"]
  }
  for row in summary_rows:
    expected = frozen_scores[(row["dataset"], int(row["count_level"]), row["method"])]
    actual = (float(row["subject_mean_nrmse"]), float(row["subject_mean_ssim"]))
    if not all(math.isclose(a, b, rel_tol=1e-14, abs_tol=1e-15) for a, b in zip(actual, expected)):
      raise ValueError("reporting summary does not reproduce the frozen subject-balanced score")
  result = {
    "artifact_kind": "validation_only_reporting_package",
    "schema_version": 1,
    "status": (
      "PASS_ready_for_validation_only_reporting_execution"
      if release_check_only else "PASS_validation_only_reporting_complete"
    ),
    "reporting_git_commit": reporting_commit,
    "reporting_scope": REPORTING_SCOPE,
    "inference_scope": "descriptive_internal_validation_only",
    "independent_test_set_available": False,
    "selection_and_reporting_same_split": True,
    "selection_bias_warning_required": True,
    "config_sha256": file_sha256(config_path),
    "protocol_sha256": file_sha256(protocol_path),
    "selection_audit_sha256": file_sha256(selection_audit_path),
    "freeze_audit_sha256": file_sha256(freeze_audit_path),
    "frozen_hyperparameters_sha256": file_sha256(frozen_path),
    "validation_completion_sha256": file_sha256(original_path.parent / "_SUCCESS.json"),
    "validation_completion_artifact_count": len(completion["artifacts"]),
    "source_csv_sha256": {
      label: file_sha256(path) for label, path in zip(SOURCE_LABELS, source_paths)
    },
    "source_rows_streamed": rows_streamed,
    "selected_per_case_rows": rows_selected,
    "selected_rows_by_source": by_source,
    "subject_aggregate_rows": len(subject_rows),
    "summary_rows": len(summary_rows),
    "dataset_subject_counts": expected_subject_counts,
    "development_inventory": inventory,
    "release_check_only": release_check_only,
    "formal_output_created": False,
    "locked_config_created": False,
    "test_locked_access": "prohibited",
    "qwen_called": False,
    "new_reconstruction_executed": False,
  }
  if release_check_only:
    return result

  output_dir.parent.mkdir(parents=True, exist_ok=True)
  temporary = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.", dir=output_dir.parent))
  try:
    per_case = temporary / "selected_per_case_metrics.csv"
    with atomic_csv_writer(per_case, PER_CASE_FIELDS) as writer:
      replay = _stream_selected_rows(source_paths, selected, contract.run_id, writer=writer)
    if replay[0:4] != (rows_streamed, rows_selected, by_source, subject_rows):
      raise RuntimeError("selected reporting replay changed between validation and publication")
    per_case_manifest = finalize_csv(
      per_case, schema_name="validation_only_selected_per_case_metrics_v1",
      primary_key=PRIMARY_KEY, split="validation",
    )
    per_subject = temporary / "selected_per_subject_metrics.csv"
    with atomic_csv_writer(per_subject, SUBJECT_FIELDS) as writer:
      writer.writerows(subject_rows)
    per_subject_manifest = finalize_csv(
      per_subject, schema_name="validation_only_selected_per_subject_metrics_v1",
      primary_key=["run_id", "dataset", "subject_id", "count_level", "method"],
      split="validation",
    )
    summary = temporary / "selected_summary_metrics.csv"
    with atomic_csv_writer(summary, SUMMARY_FIELDS) as writer:
      writer.writerows(summary_rows)
    summary_manifest = finalize_csv(
      summary, schema_name="validation_only_selected_summary_metrics_v1",
      primary_key=["run_id", "dataset", "count_level", "method"], split="validation",
    )
    scope_path = temporary / "REPORTING_SCOPE.json"
    atomic_write_json(scope_path, {
      "artifact_kind": "validation_only_reporting_scope",
      "schema_version": 1,
      "reporting_scope": REPORTING_SCOPE,
      "independent_test_set_available": False,
      "selection_and_reporting_same_split": True,
      "permitted_claim": "descriptive_internal_validation_performance",
      "prohibited_claims": [
        "independent_test_performance", "external_generalization",
        "one_time_final_test_evaluation", "unbiased_post_selection_performance",
      ],
      "test_locked_access": "prohibited",
    })
    marker = write_success_marker(temporary, [
      per_case, per_case_manifest, per_subject, per_subject_manifest,
      summary, summary_manifest, scope_path,
    ])
    os.replace(temporary, output_dir)
  except BaseException:
    if temporary.exists():
      for path in sorted(temporary.rglob("*"), reverse=True):
        if path.is_file():
          path.unlink()
        elif path.is_dir():
          path.rmdir()
      temporary.rmdir()
    raise
  result.update({
    "formal_output_created": True,
    "output_dir": output_dir.relative_to(root).as_posix(),
    "selected_per_case_sha256": file_sha256(output_dir / per_case.name),
    "selected_per_subject_sha256": file_sha256(output_dir / per_subject.name),
    "selected_summary_sha256": file_sha256(output_dir / summary.name),
    "reporting_scope_sha256": file_sha256(output_dir / scope_path.name),
    "success_marker_sha256": file_sha256(output_dir / marker.name),
  })
  atomic_write_json(audit_path, result)
  return result
