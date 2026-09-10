"""Streaming selection over original, round-1, and round-2 validation sweeps."""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from pathlib import Path

from .artifacts import atomic_write_json, file_sha256
from .config import load_contract, require_development_execution
from .prior_registration import validate_registered_validation_prior_configuration
from .protocols import load_protocol
from .sweep_evaluation import SWEEP_FIELDS, parse_sweep_grid
from .sweep_incremental import (
  EXPECTED_ROWS as EXPECTED_ROUND1_ROWS,
  EXPECTED_SELECTION_AUDIT_SHA256,
  REVIEWED_EXPANSION_PLAN_SHA256,
)
from .sweep_incremental_merge import (
  INCREMENTAL_MERGED_NAME,
  REVIEWED_INCREMENTAL_SHARD_SET_SHA256,
)
from .sweep_incremental_round2 import (
  EXPECTED_CANDIDATES as EXPECTED_ROUND2_CANDIDATES,
  EXPECTED_OPERATOR_UPDATES as EXPECTED_ROUND2_OPERATOR_UPDATES,
  EXPECTED_ROWS as EXPECTED_ROUND2_ROWS,
  REVIEWED_ROUND2_PLAN_SHA256,
)
from .sweep_incremental_round2_merge import (
  NEXT_UNION_RESULT_ROWS,
  OUTPUT_NAMESPACE as ROUND2_MERGE_NAMESPACE,
  REVIEWED_ROUND2_SHARD_SET_SHA256,
  ROUND2_MERGED_NAME,
)
from .sweep_selection import (
  EXPECTED_ROWS as EXPECTED_ORIGINAL_ROWS,
  EXPECTED_SELECTIONS,
  REVIEWED_MERGE_AUDIT_SHA256,
  _CompensatedPair,
  _expected_parameters,
)
from .sweep_sharded import line_ending_equivalent_sha256
from .sweep_union_selection import (
  EXPECTED_INCREMENTAL_SUBJECT_AGGREGATES,
  EXPECTED_ORIGINAL_SUBJECT_AGGREGATES,
  REVIEWED_INCREMENTAL_MERGE_AUDIT_SHA256,
  _clean_commit,
  _json,
  _plan_candidate_maps,
  _selection_parameters,
  _stream_source,
)


REVIEWED_UNION_SELECTION_AUDIT_SHA256 = (
  "854edf296ee368282125d225d93a0746399a0598d9319691e15a1ed062be6b33"
)
REVIEWED_ROUND2_MERGE_AUDIT_SHA256 = (
  "715f42e5481d7dc9f4acdac56fb71919049012c7ae8a82263b2801f0e39e94f8"
)
EXPECTED_ROUND2_SUBJECT_AGGREGATES = 853
EXPECTED_THREE_SOURCE_SUBJECT_AGGREGATES = (
  EXPECTED_ORIGINAL_SUBJECT_AGGREGATES
  + EXPECTED_INCREMENTAL_SUBJECT_AGGREGATES
  + EXPECTED_ROUND2_SUBJECT_AGGREGATES
)


def _round2_plan_candidate_maps(plan: dict) -> tuple[dict, dict, dict]:
  incremental: dict[tuple[str, int, str], set[tuple]] = {}
  expanded_axes: dict[tuple[str, int, str, tuple], tuple[str, ...]] = {}
  source_parameters: dict[tuple[str, int, str], tuple] = {}
  for group in plan.get("groups", []):
    identity = (
      str(group["dataset"]), int(group["count_level"]), str(group["method"]),
    )
    if identity in incremental:
      raise ValueError(f"duplicate round-2 expansion group: {identity}")
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
        raise ValueError(f"duplicate round-2 candidate: {identity + parameters}")
      axes = tuple(item["expanded_axes"])
      if not axes:
        raise ValueError(f"round-2 candidate has no expanded axis: {identity}")
      candidates.add(parameters)
      expanded_axes[(*identity, parameters)] = axes
    incremental[identity] = candidates
  if (
    len(incremental) != 25
    or sum(map(len, incremental.values())) != EXPECTED_ROUND2_CANDIDATES
  ):
    raise ValueError("round-2 selection requires the reviewed 25-group, 39-candidate shell")
  return incremental, expanded_axes, source_parameters


def stream_subject_balanced_round2_union_selection(
  original_path: Path, round1_path: Path, round2_path: Path,
  protocol, round1_plan: dict, round2_plan: dict, run_id: str,
) -> dict:
  """Stream three reviewed CSVs and select over their disjoint candidate union."""
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
  round1_parameters, _, _ = _plan_candidate_maps(round1_plan)
  round2_parameters, expanded_axes, source_parameters = (
    _round2_plan_candidate_maps(round2_plan)
  )
  for identity, candidates in round1_parameters.items():
    if identity not in original_parameters or candidates & original_parameters[identity]:
      raise ValueError(f"round-1 shell overlaps the original grid: {identity}")
  for identity, candidates in round2_parameters.items():
    if identity not in original_parameters:
      raise ValueError(f"round-2 shell has an undeclared group: {identity}")
    if candidates & (original_parameters[identity] | round1_parameters[identity]):
      raise ValueError(f"round-2 shell overlaps a reviewed earlier domain: {identity}")
  union_parameters = {
    identity: (
      parameters
      | round1_parameters.get(identity, set())
      | round2_parameters.get(identity, set())
    )
    for identity, parameters in original_parameters.items()
  }

  sources = (
    ("original", original_path, original_parameters),
    ("round1_incremental", round1_path, round1_parameters),
    ("round2_incremental", round2_path, round2_parameters),
  )
  subject_totals: dict[tuple, _CompensatedPair] = {}
  method_counts = {source: defaultdict(Counter) for source, _, _ in sources}
  count_method_counts = {source: defaultdict(int) for source, _, _ in sources}
  dataset_subjects: dict[str, set[str]] = defaultdict(set)
  count_levels: dict[str, set[int]] = defaultdict(set)
  rows: dict[str, int] = {}
  aggregates: dict[str, int] = {}
  previous_aggregate_count = 0
  for source, path, parameters in sources:
    rows[source] = _stream_source(
      path, source=source, run_id=run_id, allowed_methods=allowed_methods,
      expected_parameters=parameters, subject_totals=subject_totals,
      method_counts=method_counts, count_method_counts=count_method_counts,
      dataset_subjects=dataset_subjects, count_levels=count_levels,
    )
    aggregates[source] = len(subject_totals) - previous_aggregate_count
    previous_aggregate_count = len(subject_totals)

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
          raise ValueError(f"three-source candidate grid is incomplete for {identity}")
        scored = []
        for parameters, subject_values in candidates.items():
          if set(subject_values) != subjects:
            raise ValueError(
              f"three-source subject coverage differs for {identity + parameters}"
            )
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
  round2_selected = 0
  for selection in selected:
    identity = (
      selection["dataset"], int(selection["count_level"]), selection["method"],
    )
    parameters = _selection_parameters(selection)
    if identity in source_parameters:
      permitted = {source_parameters[identity]} | round2_parameters[identity]
      if parameters not in permitted:
        raise ValueError(
          f"three-source winner is outside the reviewed round-2 shell: {identity}"
        )
      axes = expanded_axes.get((*identity, parameters), ())
      if axes:
        round2_selected += 1
        expanded_boundaries.append({
          "selection": selection, "expanded_boundary_axes": list(axes),
        })
        expanded_axis_counts.update(axes)

  source_method_counts = {
    source: {
      dataset: dict(sorted(counts.items()))
      for dataset, counts in sorted(by_dataset.items())
    }
    for source, by_dataset in method_counts.items()
  }
  union_method_counts = {
    dataset: dict(sorted(sum(
      (method_counts[source][dataset] for source, _, _ in sources), Counter(),
    ).items()))
    for dataset in sorted(allowed_methods)
  }
  return {
    "original_rows_streamed": rows["original"],
    "round1_incremental_rows_streamed": rows["round1_incremental"],
    "round2_incremental_rows_streamed": rows["round2_incremental"],
    "union_rows_streamed": sum(rows.values()),
    "original_subject_configuration_aggregates": aggregates["original"],
    "round1_incremental_subject_configuration_aggregates": (
      aggregates["round1_incremental"]
    ),
    "round2_incremental_subject_configuration_aggregates": (
      aggregates["round2_incremental"]
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
      for source, _, _ in sources
    },
    "selection_count": len(selected),
    "dataset_selection_counts": dict(sorted(selection_counts.items())),
    "selected_hyperparameters": selected,
    "reviewed_boundary_group_count": len(source_parameters),
    "unchanged_prior_union_selection_count": EXPECTED_SELECTIONS - len(source_parameters),
    "round2_incremental_candidate_selected_count": round2_selected,
    "source_boundary_winner_count": len(source_parameters) - round2_selected,
    "resolved_reviewed_boundary_group_count": len(source_parameters) - round2_selected,
    "expanded_boundary_selection_count": len(expanded_boundaries),
    "expanded_boundary_axis_counts": dict(sorted(expanded_axis_counts.items())),
    "expanded_boundary_selections": expanded_boundaries,
  }


def _sum_method_counts(*values: dict) -> dict:
  return {
    dataset: dict(sorted(sum(
      (Counter(value[dataset]) for value in values), Counter(),
    ).items()))
    for dataset in ("brainweb", "hecktor")
  }


def _expected_incremental_count_method_rows(
  protocol, count_levels: tuple[int, ...], plan: dict,
) -> dict:
  result = {
    dataset: {
      str(count): {method: 0 for method in (
        method for method in protocol.methods
        if not (dataset == "brainweb" and method == "CT-KEM")
        and not (dataset == "hecktor" and method == "MR-KEM")
      )}
      for count in count_levels
    }
    for dataset in ("brainweb", "hecktor")
  }
  for group in plan["groups"]:
    result[group["dataset"]][str(group["count_level"])][group["method"]] = int(
      group["expected_result_rows"]
    )
  return result


def audit_round2_union_hyperparameter_selection(
  root: Path, config_path: Path, protocol_path: Path,
  original_merge_audit_path: Path, original_selection_audit_path: Path,
  round1_plan_path: Path, round1_merge_audit_path: Path,
  round1_union_audit_path: Path, round2_plan_path: Path,
  round2_merge_audit_path: Path, output_path: Path,
) -> dict:
  """Audit the three-source union without writing a union CSV or frozen parameters."""
  root = root.resolve()
  paths = [
    config_path, protocol_path, original_merge_audit_path,
    original_selection_audit_path, round1_plan_path, round1_merge_audit_path,
    round1_union_audit_path, round2_plan_path, round2_merge_audit_path, output_path,
  ]
  (
    config_path, protocol_path, original_merge_audit_path,
    original_selection_audit_path, round1_plan_path, round1_merge_audit_path,
    round1_union_audit_path, round2_plan_path, round2_merge_audit_path, output_path,
  ) = (path.resolve() for path in paths)
  if output_path.exists():
    raise FileExistsError(f"refusing to overwrite round-2 union audit: {output_path}")
  audit_commit = _clean_commit(root)
  reviewed = (
    (original_merge_audit_path, REVIEWED_MERGE_AUDIT_SHA256),
    (original_selection_audit_path, EXPECTED_SELECTION_AUDIT_SHA256),
    (round1_plan_path, REVIEWED_EXPANSION_PLAN_SHA256),
    (round1_merge_audit_path, REVIEWED_INCREMENTAL_MERGE_AUDIT_SHA256),
    (round1_union_audit_path, REVIEWED_UNION_SELECTION_AUDIT_SHA256),
    (round2_plan_path, REVIEWED_ROUND2_PLAN_SHA256),
    (round2_merge_audit_path, REVIEWED_ROUND2_MERGE_AUDIT_SHA256),
  )
  if any(file_sha256(path) != expected for path, expected in reviewed):
    raise ValueError("round-2 union inputs are not the reviewed artifacts")

  original_merge_audit = _json(original_merge_audit_path)
  original_selection_audit = _json(original_selection_audit_path)
  round1_plan = _json(round1_plan_path)
  round1_merge_audit = _json(round1_merge_audit_path)
  round1_union_audit = _json(round1_union_audit_path)
  round2_plan = _json(round2_plan_path)
  round2_merge_audit = _json(round2_merge_audit_path)
  cfg, contract = load_contract(config_path)
  require_development_execution(cfg)
  registration = validate_registered_validation_prior_configuration(
    root, config_path, cfg,
  )
  protocol = load_protocol(protocol_path)
  original_path = root / str(cfg["outputs"]["hyperparameter_sweep"])
  round1_path = root / (
    "runs/dev_trainmax_v1/hyperparameter_expansion/" + INCREMENTAL_MERGED_NAME
  )
  round2_path = root / ROUND2_MERGE_NAMESPACE / ROUND2_MERGED_NAME
  source_paths = (original_path, round1_path, round2_path)
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
    or round1_plan.get("status") != "ready_for_researcher_review_no_execution"
    or round1_plan.get("incremental_candidate_count") != 76
    or round1_merge_audit.get("status")
    != "PASS_complete_pending_union_selection_audit"
    or round1_union_audit.get("status")
    != "BLOCKED_expanded_boundary_requires_additional_shell"
    or round1_union_audit.get("selection_count") != EXPECTED_SELECTIONS
    or round1_union_audit.get("expanded_boundary_selection_count") != 25
    or round1_union_audit.get("incremental_merge_audit_sha256")
    != REVIEWED_INCREMENTAL_MERGE_AUDIT_SHA256
    or round2_plan.get("status") != "ready_for_researcher_review_no_execution"
    or round2_plan.get("source_union_selection_audit_sha256")
    != REVIEWED_UNION_SELECTION_AUDIT_SHA256
    or round2_plan.get("incremental_candidate_count") != EXPECTED_ROUND2_CANDIDATES
    or round2_plan.get("expected_incremental_result_rows") != EXPECTED_ROUND2_ROWS
    or round2_plan.get("expected_next_union_result_rows") != NEXT_UNION_RESULT_ROWS
    or round2_merge_audit.get("status")
    != "PASS_complete_pending_round2_union_selection_audit"
    or round2_merge_audit.get("round2_plan_sha256") != REVIEWED_ROUND2_PLAN_SHA256
    or round2_merge_audit.get("union_selection_audit_sha256")
    != REVIEWED_UNION_SELECTION_AUDIT_SHA256
    or round2_merge_audit.get("round1_incremental_merge_audit_sha256")
    != REVIEWED_INCREMENTAL_MERGE_AUDIT_SHA256
    or round2_merge_audit.get("round2_incremental_result_rows")
    != EXPECTED_ROUND2_ROWS
    or round2_merge_audit.get("round2_incremental_candidate_count")
    != EXPECTED_ROUND2_CANDIDATES
    or round2_merge_audit.get("round2_incremental_operator_updates")
    != EXPECTED_ROUND2_OPERATOR_UPDATES
    or round2_merge_audit.get("source_shards") != 741
    or round2_merge_audit.get("source_cases") != 8_892
    or round2_merge_audit.get("next_union_expected_result_rows")
    != NEXT_UNION_RESULT_ROWS
    or round2_merge_audit.get("physical_union_created") is not False
    or round2_merge_audit.get("round2_union_selection_audit_created") is not False
    or round2_merge_audit.get("frozen_hyperparameters_created") is not False
    or round2_merge_audit.get("test_locked_access") != "prohibited"
    or not line_ending_equivalent_sha256(
      protocol_path, str(original_selection_audit.get("protocol_sha256")),
    )
    or file_sha256(config_path) != original_selection_audit.get("config_sha256")
    or frozen_path.exists()
  ):
    raise ValueError("round-2 union protocol or reviewed chain is inconsistent")

  source_artifacts = []
  for csv_path in source_paths:
    source_artifacts.extend((
      csv_path,
      csv_path.with_suffix(csv_path.suffix + ".manifest.json"),
      csv_path.parent / "merge_record.json",
      csv_path.parent / "merge_SUCCESS.json",
    ))
  if not all(path.is_file() for path in source_artifacts):
    raise FileNotFoundError("one or more reviewed three-source artifacts are missing")
  original_manifest, original_record, original_success = source_artifacts[1:4]
  round1_manifest, round1_record, round1_success = source_artifacts[5:8]
  round2_manifest, round2_record, round2_success = source_artifacts[9:12]
  round2_manifest_json = _json(round2_manifest)
  round1_manifest_json = _json(round1_manifest)
  round1_record_json = _json(round1_record)
  round1_success_json = _json(round1_success)
  round2_record_json = _json(round2_record)
  round2_success_json = _json(round2_success)
  if (
    original_merge_audit.get("status")
    != "PASS_complete_pending_hyperparameter_freeze"
    or original_merge_audit.get("merged_result_rows") != EXPECTED_ORIGINAL_ROWS
    or original_merge_audit.get("merged_csv_sha256") != file_sha256(original_path)
    or original_selection_audit.get("merged_manifest_sha256")
    != file_sha256(original_manifest)
    or original_selection_audit.get("merge_record_sha256")
    != file_sha256(original_record)
    or original_selection_audit.get("merge_success_sha256")
    != file_sha256(original_success)
    or round1_merge_audit.get("incremental_result_rows") != EXPECTED_ROUND1_ROWS
    or round1_merge_audit.get("incremental_merged_csv_sha256")
    != file_sha256(round1_path)
    or round1_merge_audit.get("incremental_merged_manifest_sha256")
    != file_sha256(round1_manifest)
    or round1_merge_audit.get("merge_success_sha256") != file_sha256(round1_success)
    or round1_manifest_json.get("artifact_sha256") != file_sha256(round1_path)
    or round1_manifest_json.get("row_count") != EXPECTED_ROUND1_ROWS
    or round1_manifest_json.get("columns") != SWEEP_FIELDS
    or round1_record_json.get("incremental_merged_csv_sha256")
    != file_sha256(round1_path)
    or round1_record_json.get("incremental_shard_set_sha256")
    != REVIEWED_INCREMENTAL_SHARD_SET_SHA256
    or round1_success_json.get("merge_record_sha256") != file_sha256(round1_record)
    or round1_success_json.get("incremental_merged_manifest_sha256")
    != file_sha256(round1_manifest)
    or round1_success_json.get("incremental_result_rows") != EXPECTED_ROUND1_ROWS
    or round1_union_audit.get("original_merged_csv_sha256")
    != file_sha256(original_path)
    or round1_union_audit.get("incremental_merged_csv_sha256")
    != file_sha256(round1_path)
    or round2_merge_audit.get("original_merged_csv_sha256")
    != file_sha256(original_path)
    or round2_merge_audit.get("round1_merged_csv_sha256") != file_sha256(round1_path)
    or round2_merge_audit.get("round2_merged_csv_sha256") != file_sha256(round2_path)
    or round2_merge_audit.get("round2_merged_manifest_sha256")
    != file_sha256(round2_manifest)
    or round2_merge_audit.get("merge_success_sha256") != file_sha256(round2_success)
    or round2_manifest_json.get("artifact_sha256") != file_sha256(round2_path)
    or round2_manifest_json.get("row_count") != EXPECTED_ROUND2_ROWS
    or round2_manifest_json.get("columns") != SWEEP_FIELDS
    or round2_record_json.get("round2_merged_csv_sha256") != file_sha256(round2_path)
    or round2_record_json.get("round2_shard_set_sha256")
    != REVIEWED_ROUND2_SHARD_SET_SHA256
    or round2_success_json.get("merge_record_sha256") != file_sha256(round2_record)
    or round2_success_json.get("round2_merged_csv_sha256")
    != file_sha256(round2_path)
    or round2_success_json.get("round2_merged_manifest_sha256")
    != file_sha256(round2_manifest)
    or round2_success_json.get("round2_incremental_result_rows")
    != EXPECTED_ROUND2_ROWS
  ):
    raise ValueError("three-source artifacts differ from their reviewed audits")
  for csv_path in (round1_path, round2_path):
    actual_files = {path.name for path in csv_path.parent.iterdir() if path.is_file()}
    if actual_files != {
      csv_path.name, f"{csv_path.name}.manifest.json",
      "merge_record.json", "merge_SUCCESS.json",
    }:
      raise ValueError(f"merge namespace does not contain exactly four files: {csv_path}")

  selection = stream_subject_balanced_round2_union_selection(
    original_path, round1_path, round2_path, protocol,
    round1_plan, round2_plan, contract.run_id,
  )
  prior_winners = {
    (item["dataset"], int(item["count_level"]), item["method"]):
      _selection_parameters(item)
    for item in round1_union_audit["selected_hyperparameters"]
  }
  round2_groups = {
    (group["dataset"], int(group["count_level"]), group["method"])
    for group in round2_plan["groups"]
  }
  reviewed_round2_groups = {
    (
      item["selection"]["dataset"], int(item["selection"]["count_level"]),
      item["selection"]["method"],
    )
    for item in round1_union_audit["expanded_boundary_selections"]
  }
  if round2_groups != reviewed_round2_groups:
    raise ValueError("round-2 groups differ from reviewed first-union boundaries")
  for group in round2_plan["groups"]:
    identity = (group["dataset"], int(group["count_level"]), group["method"])
    if _selection_parameters(group["source_selected_parameters"]) != prior_winners[identity]:
      raise ValueError(f"round-2 source differs from prior union winner: {identity}")
  unchanged_exact = 0
  source_boundary_winners = 0
  for item in selection["selected_hyperparameters"]:
    identity = (item["dataset"], int(item["count_level"]), item["method"])
    parameters = _selection_parameters(item)
    if identity not in prior_winners:
      raise ValueError(f"three-source selection produced an undeclared group: {identity}")
    if identity not in round2_groups:
      if parameters != prior_winners[identity]:
        raise ValueError(f"unchanged prior union selection changed: {identity}")
      unchanged_exact += 1
    elif parameters == prior_winners[identity]:
      source_boundary_winners += 1
  selection["unchanged_prior_union_selection_exact_match_count"] = unchanged_exact
  selection["source_boundary_winner_count"] = source_boundary_winners

  expected_round1_count_rows = _expected_incremental_count_method_rows(
    protocol, contract.count_levels, round1_plan,
  )
  expected_round2_count_rows = _expected_incremental_count_method_rows(
    protocol, contract.count_levels, round2_plan,
  )
  expected_union_method_counts = _sum_method_counts(
    original_merge_audit["method_row_counts"],
    round1_merge_audit["method_row_counts"],
    round2_merge_audit["method_row_counts"],
  )
  if (
    selection["original_rows_streamed"] != EXPECTED_ORIGINAL_ROWS
    or selection["round1_incremental_rows_streamed"] != EXPECTED_ROUND1_ROWS
    or selection["round2_incremental_rows_streamed"] != EXPECTED_ROUND2_ROWS
    or selection["union_rows_streamed"] != NEXT_UNION_RESULT_ROWS
    or selection["original_subject_configuration_aggregates"]
    != EXPECTED_ORIGINAL_SUBJECT_AGGREGATES
    or selection["round1_incremental_subject_configuration_aggregates"]
    != EXPECTED_INCREMENTAL_SUBJECT_AGGREGATES
    or selection["round2_incremental_subject_configuration_aggregates"]
    != EXPECTED_ROUND2_SUBJECT_AGGREGATES
    or selection["union_subject_configuration_aggregates"]
    != EXPECTED_THREE_SOURCE_SUBJECT_AGGREGATES
    or selection["dataset_subject_counts"] != {"brainweb": 4, "hecktor": 45}
    or selection["dataset_count_levels"] != {
      dataset: sorted(contract.count_levels) for dataset in ("brainweb", "hecktor")
    }
    or selection["source_method_row_counts"]["original"]
    != original_merge_audit["method_row_counts"]
    or selection["source_method_row_counts"]["round1_incremental"]
    != round1_merge_audit["method_row_counts"]
    or selection["source_method_row_counts"]["round2_incremental"]
    != round2_merge_audit["method_row_counts"]
    or selection["source_dataset_count_method_row_counts"]["original"]
    != original_selection_audit["dataset_count_method_row_counts"]
    or selection["source_dataset_count_method_row_counts"]["round1_incremental"]
    != expected_round1_count_rows
    or selection["source_dataset_count_method_row_counts"]["round2_incremental"]
    != expected_round2_count_rows
    or selection["union_method_row_counts"] != expected_union_method_counts
    or selection["selection_count"] != EXPECTED_SELECTIONS
    or selection["dataset_selection_counts"] != {"brainweb": 24, "hecktor": 24}
    or selection["reviewed_boundary_group_count"] != 25
    or selection["unchanged_prior_union_selection_count"] != 23
    or selection["unchanged_prior_union_selection_exact_match_count"] != 23
    or selection["source_boundary_winner_count"]
    + selection["round2_incremental_candidate_selected_count"] != 25
  ):
    raise ValueError("three-source selection counts differ from reviewed inputs")

  boundary_required = selection["expanded_boundary_selection_count"] > 0
  result = {
    "artifact_kind": "validation_hyperparameter_round2_union_selection_audit",
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
    "round1_expansion_plan_sha256": REVIEWED_EXPANSION_PLAN_SHA256,
    "round1_merge_audit_sha256": REVIEWED_INCREMENTAL_MERGE_AUDIT_SHA256,
    "round1_union_selection_audit_sha256": REVIEWED_UNION_SELECTION_AUDIT_SHA256,
    "round2_expansion_plan_sha256": REVIEWED_ROUND2_PLAN_SHA256,
    "round2_merge_audit_sha256": REVIEWED_ROUND2_MERGE_AUDIT_SHA256,
    "original_merged_csv_sha256": file_sha256(original_path),
    "round1_merged_csv_sha256": file_sha256(round1_path),
    "round2_merged_csv_sha256": file_sha256(round2_path),
    "round2_merged_manifest_sha256": file_sha256(round2_manifest),
    "round2_merge_record_sha256": file_sha256(round2_record),
    "round2_merge_success_sha256": file_sha256(round2_success),
    "original_shard_set_sha256": original_merge_audit["shard_set_sha256"],
    "round1_shard_set_sha256": REVIEWED_INCREMENTAL_SHARD_SET_SHA256,
    "round2_shard_set_sha256": REVIEWED_ROUND2_SHARD_SET_SHA256,
    "prior_registration": registration,
    "selection_unit": "subject",
    "selection_primary": "mean_nrmse",
    "selection_tiebreaker": "mean_ssim",
    "selection_domain": "reviewed_original_grid_union_round1_and_round2_local_shells",
    "streaming_aggregation": "kahan_per_subject_then_fsum_across_subjects",
    "adaptive_stopping_rule": round2_plan["adaptive_stopping_rule"],
    "expanded_boundary_policy": "only_round2_new_outward_values_trigger_another_shell",
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
