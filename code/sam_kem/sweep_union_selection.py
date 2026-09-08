"""Streaming selection over reviewed original and incremental validation sweeps."""

from __future__ import annotations

import csv
import json
import math
import subprocess
from collections import Counter, defaultdict
from pathlib import Path

from .artifacts import atomic_write_json, file_sha256
from .config import load_contract, require_development_execution
from .prior_registration import validate_registered_validation_prior_configuration
from .protocols import load_protocol
from .sweep_evaluation import SWEEP_FIELDS, parse_sweep_grid
from .sweep_incremental import (
  EXPECTED_ROWS as EXPECTED_INCREMENTAL_ROWS,
  EXPECTED_SELECTION_AUDIT_SHA256,
  REVIEWED_EXPANSION_PLAN_SHA256,
)
from .sweep_incremental_merge import (
  INCREMENTAL_MERGED_NAME,
  REVIEWED_INCREMENTAL_SHARD_AUDIT_SHA256,
  REVIEWED_INCREMENTAL_SHARD_SET_SHA256,
  UNION_RESULT_ROWS,
)
from .sweep_selection import (
  EXPECTED_ROWS as EXPECTED_ORIGINAL_ROWS,
  EXPECTED_SELECTIONS,
  REVIEWED_MERGE_AUDIT_SHA256,
  _CompensatedPair,
  _expected_parameters,
  _parameter_key,
)
from .sweep_sharded import line_ending_equivalent_sha256


REVIEWED_INCREMENTAL_MERGE_AUDIT_SHA256 = (
  "121f9cc1aadefb0ee1c59b2680d2a2f91a9cba325b26ee6ed8a5f45927f60e20"
)
EXPECTED_ORIGINAL_SUBJECT_AGGREGATES = 113_680
EXPECTED_INCREMENTAL_SUBJECT_AGGREGATES = 1_944
EXPECTED_UNION_SUBJECT_AGGREGATES = (
  EXPECTED_ORIGINAL_SUBJECT_AGGREGATES + EXPECTED_INCREMENTAL_SUBJECT_AGGREGATES
)


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
      f"union selection audit requires a clean worktree: {status.splitlines()}"
    )
  return subprocess.run(
    ["git", "rev-parse", "HEAD"], cwd=root, check=True,
    capture_output=True, text=True,
  ).stdout.strip()


def _plan_candidate_maps(plan: dict) -> tuple[dict, dict, dict]:
  incremental: dict[tuple[str, int, str], set[tuple]] = {}
  expanded_axes: dict[tuple[str, int, str, tuple], tuple[str, ...]] = {}
  source_parameters: dict[tuple[str, int, str], tuple] = {}
  for group in plan.get("groups", []):
    identity = (
      str(group["dataset"]), int(group["count_level"]), str(group["method"]),
    )
    if identity in incremental:
      raise ValueError(f"duplicate expansion group: {identity}")
    source = group["source_selected_parameters"]
    source_parameters[identity] = (
      float(source["sigma"]), int(source["radius"]), int(source["neighbours"]),
      int(source["iterations"]), float(source["alpha"]),
    )
    candidates = set()
    for item in group["incremental_candidates"]:
      parameters = (
        float(item["sigma"]), int(item["radius"]), int(item["neighbours"]),
        int(item["iterations"]), float(item["alpha"]),
      )
      if parameters in candidates:
        raise ValueError(f"duplicate expansion candidate: {identity + parameters}")
      axes = tuple(item["expanded_axes"])
      if not axes:
        raise ValueError(f"expansion candidate has no expanded axis: {identity}")
      candidates.add(parameters)
      expanded_axes[(*identity, parameters)] = axes
    incremental[identity] = candidates
  if len(incremental) != 30 or sum(map(len, incremental.values())) != 76:
    raise ValueError("union selection requires the reviewed 30-group, 76-candidate shell")
  return incremental, expanded_axes, source_parameters


def _stream_source(
  path: Path, *, source: str, run_id: str, allowed_methods: dict,
  expected_parameters: dict, subject_totals: dict, method_counts: dict,
  count_method_counts: dict, dataset_subjects: dict, count_levels: dict,
) -> int:
  rows_streamed = 0
  with path.open("r", encoding="utf-8", newline="") as handle:
    reader = csv.DictReader(handle)
    if list(reader.fieldnames or ()) != SWEEP_FIELDS:
      raise ValueError(f"{source} sweep columns differ from the frozen schema")
    for row in reader:
      rows_streamed += 1
      dataset = row["dataset"]
      method = row["method"]
      count = int(row["count_level"])
      group = (dataset, count, method)
      parameters = _parameter_key(row)
      if (
        row["run_id"] != run_id or row["split"] != "validation"
        or dataset not in allowed_methods or method not in allowed_methods[dataset]
        or group not in expected_parameters
        or parameters not in expected_parameters[group]
      ):
        raise ValueError(f"invalid {source} row identity at data row {rows_streamed}")
      nrmse = float(row["nrmse"])
      ssim = float(row["ssim"])
      if not math.isfinite(nrmse) or not math.isfinite(ssim):
        raise ValueError(f"non-finite {source} metric at data row {rows_streamed}")
      subject = row["subject_id"]
      key = (*group, *parameters, subject)
      aggregate = subject_totals.get(key)
      if aggregate is None:
        aggregate = subject_totals[key] = _CompensatedPair()
      aggregate.add(nrmse, ssim)
      method_counts[source][dataset][method] += 1
      count_method_counts[source][group] += 1
      dataset_subjects[dataset].add(subject)
      count_levels[dataset].add(count)
  return rows_streamed


def stream_subject_balanced_union_selection(
  original_path: Path, incremental_path: Path, protocol, plan: dict, run_id: str,
) -> dict:
  """Stream both reviewed CSVs and select over their disjoint candidate union."""
  grid = parse_sweep_grid(protocol)
  allowed_methods = {
    "brainweb": tuple(method for method in protocol.methods if method != "CT-KEM"),
    "hecktor": tuple(method for method in protocol.methods if method != "MR-KEM"),
  }
  original_parameters = {
    (dataset, count, method): _expected_parameters(method, grid)
    for dataset, methods in allowed_methods.items()
    for count in (100_000, 200_000, 500_000, 1_000_000)
    for method in methods
  }
  incremental_parameters, expanded_axes, source_parameters = _plan_candidate_maps(plan)
  for identity, candidates in incremental_parameters.items():
    if candidates & original_parameters[identity]:
      raise ValueError(f"incremental shell overlaps the original grid: {identity}")
  union_parameters = {
    identity: parameters | incremental_parameters.get(identity, set())
    for identity, parameters in original_parameters.items()
  }

  subject_totals: dict[tuple, _CompensatedPair] = {}
  method_counts = {
    source: defaultdict(Counter) for source in ("original", "incremental")
  }
  count_method_counts = {
    source: defaultdict(int) for source in ("original", "incremental")
  }
  dataset_subjects: dict[str, set[str]] = defaultdict(set)
  count_levels: dict[str, set[int]] = defaultdict(set)
  original_rows = _stream_source(
    original_path, source="original", run_id=run_id,
    allowed_methods=allowed_methods, expected_parameters=original_parameters,
    subject_totals=subject_totals, method_counts=method_counts,
    count_method_counts=count_method_counts, dataset_subjects=dataset_subjects,
    count_levels=count_levels,
  )
  original_subject_aggregates = len(subject_totals)
  incremental_rows = _stream_source(
    incremental_path, source="incremental", run_id=run_id,
    allowed_methods=allowed_methods, expected_parameters=incremental_parameters,
    subject_totals=subject_totals, method_counts=method_counts,
    count_method_counts=count_method_counts, dataset_subjects=dataset_subjects,
    count_levels=count_levels,
  )

  candidate_subjects: dict[tuple, dict[str, tuple[float, float]]] = defaultdict(dict)
  for key, aggregate in subject_totals.items():
    dataset, count, method, sigma, radius, neighbours, iterations, alpha, subject = key
    candidate = (dataset, count, method, sigma, radius, neighbours, iterations, alpha)
    candidate_subjects[candidate][subject] = (
      aggregate.nrmse_sum / aggregate.count,
      aggregate.ssim_sum / aggregate.count,
    )

  selected = []
  selection_counts: Counter = Counter()
  for dataset in sorted(allowed_methods):
    subjects = dataset_subjects[dataset]
    for count in sorted(count_levels[dataset]):
      for method in allowed_methods[dataset]:
        identity = (dataset, count, method)
        candidates = {
          key[3:]: values for key, values in candidate_subjects.items()
          if key[:3] == identity
        }
        if set(candidates) != union_parameters[identity]:
          raise ValueError(f"union candidate grid is incomplete for {identity}")
        scored = []
        for parameters, subject_values in candidates.items():
          if set(subject_values) != subjects:
            raise ValueError(f"union subject coverage differs for {identity + parameters}")
          mean_nrmse = math.fsum(
            value[0] for value in subject_values.values()
          ) / len(subjects)
          mean_ssim = math.fsum(
            value[1] for value in subject_values.values()
          ) / len(subjects)
          scored.append((mean_nrmse, mean_ssim, parameters))
        mean_nrmse, mean_ssim, parameters = min(
          scored, key=lambda item: (item[0], -item[1], item[2]),
        )
        sigma, radius, neighbours, iterations, alpha = parameters
        selected.append({
          "run_id": run_id, "dataset": dataset, "count_level": count,
          "method": method, "sigma": sigma, "radius": radius,
          "neighbours": neighbours, "iterations": iterations, "alpha": alpha,
          "validation_subject_mean_nrmse": mean_nrmse,
          "validation_subject_mean_ssim": mean_ssim,
        })
        selection_counts[dataset] += 1

  expanded_boundaries = []
  expanded_axis_counts: Counter = Counter()
  incremental_selected = 0
  for selection in selected:
    identity = (
      selection["dataset"], int(selection["count_level"]), selection["method"],
    )
    parameters = (
      float(selection["sigma"]), int(selection["radius"]),
      int(selection["neighbours"]), int(selection["iterations"]),
      float(selection["alpha"]),
    )
    if identity in source_parameters:
      permitted = {source_parameters[identity]} | incremental_parameters[identity]
      if parameters not in permitted:
        raise ValueError(f"union winner is outside the reviewed local shell: {identity}")
      axes = expanded_axes.get((*identity, parameters), ())
      if axes:
        incremental_selected += 1
        expanded_boundaries.append({
          "selection": selection, "expanded_boundary_axes": list(axes),
        })
        expanded_axis_counts.update(axes)
    elif parameters not in original_parameters[identity]:
      raise ValueError(f"interior union winner changed unexpectedly: {identity}")

  source_method_counts = {
    source: {
      dataset: dict(sorted(counts.items()))
      for dataset, counts in sorted(by_dataset.items())
    }
    for source, by_dataset in method_counts.items()
  }
  union_method_counts = {
    dataset: dict(sorted(
      (method_counts["original"][dataset] +
       method_counts["incremental"][dataset]).items()
    ))
    for dataset in sorted(allowed_methods)
  }
  return {
    "original_rows_streamed": original_rows,
    "incremental_rows_streamed": incremental_rows,
    "union_rows_streamed": original_rows + incremental_rows,
    "original_subject_configuration_aggregates": original_subject_aggregates,
    "incremental_subject_configuration_aggregates": (
      len(subject_totals) - original_subject_aggregates
    ),
    "union_subject_configuration_aggregates": len(subject_totals),
    "dataset_subject_counts": {
      dataset: len(subjects) for dataset, subjects in sorted(dataset_subjects.items())
    },
    "dataset_count_levels": {
      dataset: sorted(values) for dataset, values in sorted(count_levels.items())
    },
    "source_method_row_counts": source_method_counts,
    "union_method_row_counts": union_method_counts,
    "source_dataset_count_method_row_counts": {
      source: {
        dataset: {
          str(count): {
            method: count_method_counts[source][(dataset, count, method)]
            for method in allowed_methods[dataset]
          }
          for count in sorted(count_levels[dataset])
        }
        for dataset in sorted(allowed_methods)
      }
      for source in ("original", "incremental")
    },
    "selection_count": len(selected),
    "dataset_selection_counts": dict(sorted(selection_counts.items())),
    "selected_hyperparameters": selected,
    "reviewed_boundary_group_count": len(source_parameters),
    "unchanged_interior_selection_count": EXPECTED_SELECTIONS - len(source_parameters),
    "incremental_candidate_selected_count": incremental_selected,
    "resolved_reviewed_boundary_group_count": len(source_parameters) - incremental_selected,
    "expanded_boundary_selection_count": len(expanded_boundaries),
    "expanded_boundary_axis_counts": dict(sorted(expanded_axis_counts.items())),
    "expanded_boundary_selections": expanded_boundaries,
  }


def _sum_method_counts(first: dict, second: dict) -> dict:
  return {
    dataset: dict(sorted((Counter(first[dataset]) + Counter(second[dataset])).items()))
    for dataset in ("brainweb", "hecktor")
  }


def _selection_parameters(selection: dict) -> tuple:
  return (
    float(selection["sigma"]), int(selection["radius"]),
    int(selection["neighbours"]), int(selection["iterations"]),
    float(selection["alpha"]),
  )


def audit_union_hyperparameter_selection(
  root: Path, config_path: Path, protocol_path: Path,
  original_merge_audit_path: Path, original_selection_audit_path: Path,
  plan_path: Path, incremental_merge_audit_path: Path, output_path: Path,
) -> dict:
  """Audit subject-balanced union selection without writing frozen parameters."""
  root = root.resolve()
  config_path = config_path.resolve()
  protocol_path = protocol_path.resolve()
  original_merge_audit_path = original_merge_audit_path.resolve()
  original_selection_audit_path = original_selection_audit_path.resolve()
  plan_path = plan_path.resolve()
  incremental_merge_audit_path = incremental_merge_audit_path.resolve()
  output_path = output_path.resolve()
  if output_path.exists():
    raise FileExistsError(f"refusing to overwrite union selection audit: {output_path}")
  audit_commit = _clean_commit(root)
  if (
    file_sha256(original_merge_audit_path) != REVIEWED_MERGE_AUDIT_SHA256
    or file_sha256(original_selection_audit_path) != EXPECTED_SELECTION_AUDIT_SHA256
    or file_sha256(plan_path) != REVIEWED_EXPANSION_PLAN_SHA256
    or file_sha256(incremental_merge_audit_path)
    != REVIEWED_INCREMENTAL_MERGE_AUDIT_SHA256
  ):
    raise ValueError("union selection inputs are not the reviewed artifacts")

  original_merge_audit = _json(original_merge_audit_path)
  original_selection_audit = _json(original_selection_audit_path)
  plan = _json(plan_path)
  incremental_merge_audit = _json(incremental_merge_audit_path)
  cfg, contract = load_contract(config_path)
  require_development_execution(cfg)
  registration = validate_registered_validation_prior_configuration(
    root, config_path, cfg,
  )
  protocol = load_protocol(protocol_path)
  original_path = root / str(cfg["outputs"]["hyperparameter_sweep"])
  original_manifest_path = original_path.with_suffix(original_path.suffix + ".manifest.json")
  original_record_path = original_path.parent / "merge_record.json"
  original_success_path = original_path.parent / "merge_SUCCESS.json"
  incremental_path = (
    root / "runs/dev_trainmax_v1/hyperparameter_expansion" / INCREMENTAL_MERGED_NAME
  )
  incremental_manifest_path = incremental_path.with_suffix(
    incremental_path.suffix + ".manifest.json"
  )
  incremental_record_path = incremental_path.parent / "merge_record.json"
  incremental_success_path = incremental_path.parent / "merge_SUCCESS.json"
  frozen_path = root / str(cfg["outputs"]["frozen_hyperparameters"])
  if (
    protocol.kind != "hyperparameter_sweep" or protocol.split != "validation"
    or protocol.raw.get("selection_unit") != "subject"
    or protocol.raw.get("selection_primary") != "mean_nrmse"
    or protocol.raw.get("selection_tiebreaker") != "mean_ssim"
    or protocol.raw.get("test_locked_access") != "prohibited"
    or not protocol.raw.get("expand_all_nonphysical_boundaries", False)
    or original_selection_audit.get("status")
    != "BLOCKED_grid_boundary_expansion_required"
    or original_selection_audit.get("prior_registration") != registration
    or original_selection_audit.get("selection_count") != EXPECTED_SELECTIONS
    or original_selection_audit.get("boundary_selection_count") != 30
    or plan.get("status") != "ready_for_researcher_review_no_execution"
    or plan.get("selection_audit_sha256") != EXPECTED_SELECTION_AUDIT_SHA256
    or plan.get("boundary_group_count") != 30
    or plan.get("unchanged_interior_selection_count") != 18
    or plan.get("incremental_candidate_count") != 76
    or plan.get("expected_incremental_result_rows") != EXPECTED_INCREMENTAL_ROWS
    or incremental_merge_audit.get("status")
    != "PASS_complete_pending_union_selection_audit"
    or incremental_merge_audit.get("incremental_shard_audit_sha256")
    != REVIEWED_INCREMENTAL_SHARD_AUDIT_SHA256
    or incremental_merge_audit.get("incremental_shard_set_sha256")
    != REVIEWED_INCREMENTAL_SHARD_SET_SHA256
    or incremental_merge_audit.get("selection_audit_sha256")
    != EXPECTED_SELECTION_AUDIT_SHA256
    or incremental_merge_audit.get("plan_sha256") != REVIEWED_EXPANSION_PLAN_SHA256
    or incremental_merge_audit.get("incremental_result_rows")
    != EXPECTED_INCREMENTAL_ROWS
    or incremental_merge_audit.get("union_expected_result_rows") != UNION_RESULT_ROWS
    or incremental_merge_audit.get("union_selection_audit_created") is not False
    or incremental_merge_audit.get("frozen_hyperparameters_created") is not False
    or incremental_merge_audit.get("test_locked_access") != "prohibited"
    or not line_ending_equivalent_sha256(
      protocol_path, str(original_selection_audit.get("protocol_sha256")),
    )
    or file_sha256(config_path) != original_selection_audit.get("config_sha256")
    or frozen_path.exists()
  ):
    raise ValueError("union selection protocol or reviewed chain is inconsistent")

  required = (
    original_path, original_manifest_path, original_record_path,
    original_success_path, incremental_path,
    incremental_manifest_path, incremental_record_path, incremental_success_path,
  )
  if not all(path.is_file() for path in required):
    raise FileNotFoundError("one or more reviewed union input artifacts are missing")
  incremental_manifest = _json(incremental_manifest_path)
  incremental_record = _json(incremental_record_path)
  incremental_success = _json(incremental_success_path)
  if (
    original_merge_audit.get("status")
    != "PASS_complete_pending_hyperparameter_freeze"
    or original_merge_audit.get("merged_result_rows") != EXPECTED_ORIGINAL_ROWS
    or original_merge_audit.get("merged_csv_sha256") != file_sha256(original_path)
    or original_merge_audit.get("merged_manifest_sha256")
    != file_sha256(original_manifest_path)
    or original_merge_audit.get("merge_success_sha256")
    != file_sha256(original_success_path)
    or original_selection_audit.get("merged_csv_sha256") != file_sha256(original_path)
    or original_selection_audit.get("merged_manifest_sha256")
    != file_sha256(original_manifest_path)
    or original_selection_audit.get("merge_record_sha256")
    != file_sha256(original_record_path)
    or original_selection_audit.get("merge_success_sha256")
    != file_sha256(original_success_path)
    or plan.get("merged_csv_sha256") != file_sha256(original_path)
    or incremental_merge_audit.get("original_merged_csv_sha256")
    != file_sha256(original_path)
    or incremental_merge_audit.get("incremental_merged_csv_sha256")
    != file_sha256(incremental_path)
    or incremental_merge_audit.get("incremental_merged_manifest_sha256")
    != file_sha256(incremental_manifest_path)
    or incremental_merge_audit.get("merge_success_sha256")
    != file_sha256(incremental_success_path)
    or incremental_manifest.get("artifact_sha256") != file_sha256(incremental_path)
    or incremental_manifest.get("row_count") != EXPECTED_INCREMENTAL_ROWS
    or incremental_manifest.get("columns") != SWEEP_FIELDS
    or incremental_record.get("incremental_merged_csv_sha256")
    != file_sha256(incremental_path)
    or incremental_record.get("incremental_shard_set_sha256")
    != REVIEWED_INCREMENTAL_SHARD_SET_SHA256
    or incremental_success.get("merge_record_sha256")
    != file_sha256(incremental_record_path)
    or incremental_success.get("incremental_merged_manifest_sha256")
    != file_sha256(incremental_manifest_path)
    or incremental_success.get("incremental_result_rows")
    != EXPECTED_INCREMENTAL_ROWS
  ):
    raise ValueError("union source artifacts differ from their reviewed audits")
  actual_incremental_files = {
    path.name for path in incremental_path.parent.iterdir() if path.is_file()
  }
  if actual_incremental_files != {
    INCREMENTAL_MERGED_NAME,
    f"{INCREMENTAL_MERGED_NAME}.manifest.json",
    "merge_record.json", "merge_SUCCESS.json",
  }:
    raise ValueError("incremental merge namespace does not contain exactly four files")

  selection = stream_subject_balanced_union_selection(
    original_path, incremental_path, protocol, plan, contract.run_id,
  )
  original_winners = {
    (item["dataset"], int(item["count_level"]), item["method"]):
    _selection_parameters(item)
    for item in original_selection_audit["selected_hyperparameters"]
  }
  boundary_groups = {
    (group["dataset"], int(group["count_level"]), group["method"])
    for group in plan["groups"]
  }
  reviewed_boundary_groups = {
    (
      item["selection"]["dataset"],
      int(item["selection"]["count_level"]),
      item["selection"]["method"],
    )
    for item in original_selection_audit["boundary_selections"]
  }
  if boundary_groups != reviewed_boundary_groups:
    raise ValueError("expansion groups differ from reviewed original boundaries")
  for group in plan["groups"]:
    identity = (group["dataset"], int(group["count_level"]), group["method"])
    if _selection_parameters(group["source_selected_parameters"]) != original_winners[identity]:
      raise ValueError(f"expansion source differs from original winner: {identity}")
  interior_exact = 0
  source_boundary_winners = 0
  for item in selection["selected_hyperparameters"]:
    identity = (item["dataset"], int(item["count_level"]), item["method"])
    if identity not in original_winners:
      raise ValueError(f"union selection produced an undeclared group: {identity}")
    if identity not in boundary_groups:
      if _selection_parameters(item) != original_winners[identity]:
        raise ValueError(f"original interior selection changed: {identity}")
      interior_exact += 1
    elif _selection_parameters(item) == original_winners[identity]:
      source_boundary_winners += 1
  selection["interior_selection_exact_match_count"] = interior_exact
  selection["source_boundary_winner_count"] = source_boundary_winners

  expected_incremental_count_method_rows = {
    dataset: {
      str(count): {method: 0 for method in (
        method for method in protocol.methods
        if not (dataset == "brainweb" and method == "CT-KEM")
        and not (dataset == "hecktor" and method == "MR-KEM")
      )}
      for count in contract.count_levels
    }
    for dataset in ("brainweb", "hecktor")
  }
  for group in plan["groups"]:
    expected_incremental_count_method_rows[group["dataset"]][
      str(group["count_level"])
    ][group["method"]] = int(group["expected_result_rows"])
  expected_union_method_counts = _sum_method_counts(
    original_merge_audit["method_row_counts"],
    incremental_merge_audit["method_row_counts"],
  )
  if (
    selection["original_rows_streamed"] != EXPECTED_ORIGINAL_ROWS
    or selection["incremental_rows_streamed"] != EXPECTED_INCREMENTAL_ROWS
    or selection["union_rows_streamed"] != UNION_RESULT_ROWS
    or selection["original_subject_configuration_aggregates"]
    != EXPECTED_ORIGINAL_SUBJECT_AGGREGATES
    or selection["incremental_subject_configuration_aggregates"]
    != EXPECTED_INCREMENTAL_SUBJECT_AGGREGATES
    or selection["union_subject_configuration_aggregates"]
    != EXPECTED_UNION_SUBJECT_AGGREGATES
    or selection["dataset_subject_counts"] != {"brainweb": 4, "hecktor": 45}
    or selection["dataset_count_levels"] != {
      dataset: sorted(contract.count_levels) for dataset in ("brainweb", "hecktor")
    }
    or selection["source_method_row_counts"]["original"]
    != original_merge_audit["method_row_counts"]
    or selection["source_method_row_counts"]["incremental"]
    != incremental_merge_audit["method_row_counts"]
    or selection["source_dataset_count_method_row_counts"]["original"]
    != original_selection_audit["dataset_count_method_row_counts"]
    or selection["source_dataset_count_method_row_counts"]["incremental"]
    != expected_incremental_count_method_rows
    or selection["union_method_row_counts"] != expected_union_method_counts
    or selection["selection_count"] != EXPECTED_SELECTIONS
    or selection["dataset_selection_counts"] != {"brainweb": 24, "hecktor": 24}
    or selection["reviewed_boundary_group_count"] != 30
    or selection["unchanged_interior_selection_count"] != 18
    or selection["interior_selection_exact_match_count"] != 18
    or selection["source_boundary_winner_count"]
    + selection["incremental_candidate_selected_count"] != 30
  ):
    raise ValueError("streamed union selection counts differ from the reviewed inputs")

  boundary_required = selection["expanded_boundary_selection_count"] > 0
  result = {
    "artifact_kind": "validation_hyperparameter_union_selection_audit",
    "schema_version": 1,
    "status": (
      "BLOCKED_expanded_boundary_requires_additional_shell"
      if boundary_required else "PASS_ready_for_hyperparameter_freeze"
    ),
    "read_only": True,
    "audit_git_commit": audit_commit,
    "config_sha256": file_sha256(config_path),
    "protocol_sha256": file_sha256(protocol_path),
    "original_merge_audit_sha256": REVIEWED_MERGE_AUDIT_SHA256,
    "original_selection_audit_sha256": EXPECTED_SELECTION_AUDIT_SHA256,
    "expansion_plan_sha256": REVIEWED_EXPANSION_PLAN_SHA256,
    "incremental_merge_audit_sha256": REVIEWED_INCREMENTAL_MERGE_AUDIT_SHA256,
    "original_merged_csv_sha256": file_sha256(original_path),
    "incremental_merged_csv_sha256": file_sha256(incremental_path),
    "incremental_merged_manifest_sha256": file_sha256(incremental_manifest_path),
    "incremental_merge_record_sha256": file_sha256(incremental_record_path),
    "incremental_merge_success_sha256": file_sha256(incremental_success_path),
    "original_shard_set_sha256": original_merge_audit["shard_set_sha256"],
    "incremental_shard_set_sha256": REVIEWED_INCREMENTAL_SHARD_SET_SHA256,
    "prior_registration": registration,
    "selection_unit": "subject",
    "selection_primary": "mean_nrmse",
    "selection_tiebreaker": "mean_ssim",
    "selection_domain": "reviewed_original_grid_union_incremental_local_shell",
    "streaming_aggregation": "kahan_per_subject_then_fsum_across_subjects",
    "adaptive_stopping_rule": plan["adaptive_stopping_rule"],
    "expanded_boundary_policy": "only_new_outward_values_trigger_another_shell",
    "global_primary_key_uniqueness": (
      "verified_within_reviewed_sources_and_by_disjoint_parameter_tuple_domains"
    ),
    **selection,
    "union_csv_created": False,
    "frozen_hyperparameters_created": False,
    "test_locked_access": "prohibited",
  }
  atomic_write_json(output_path, result)
  return result
