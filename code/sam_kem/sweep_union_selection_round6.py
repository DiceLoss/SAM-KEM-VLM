"""Streaming selection over all seven reviewed validation sweep sources."""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from pathlib import Path

from .artifacts import atomic_write_json, file_sha256
from .config import load_contract, require_development_execution
from .prior_registration import validate_registered_validation_prior_configuration
from .protocols import load_protocol
from .sweep_evaluation import SWEEP_FIELDS, parse_sweep_grid
from .sweep_incremental import EXPECTED_ROWS as EXPECTED_ROUND1_ROWS
from .sweep_incremental import REVIEWED_EXPANSION_PLAN_SHA256
from .sweep_incremental_merge import INCREMENTAL_MERGED_NAME
from .sweep_incremental_round2 import (
  EXPECTED_ROWS as EXPECTED_ROUND2_ROWS,
  REVIEWED_ROUND2_PLAN_SHA256,
)
from .sweep_incremental_round2_merge import (
  OUTPUT_NAMESPACE as ROUND2_MERGE_NAMESPACE,
)
from .sweep_incremental_round2_merge import ROUND2_MERGED_NAME
from .sweep_incremental_round3 import (
  EXPECTED_ROWS as EXPECTED_ROUND3_ROWS,
  REVIEWED_ROUND3_PLAN_SHA256,
)
from .sweep_incremental_round3_merge import (
  OUTPUT_NAMESPACE as ROUND3_MERGE_NAMESPACE,
)
from .sweep_incremental_round3_merge import ROUND3_MERGED_NAME
from .sweep_incremental_round4 import (
  EXPECTED_ROWS as EXPECTED_ROUND4_ROWS,
  REVIEWED_ROUND4_PLAN_SHA256,
)
from .sweep_incremental_round4_merge import (
  OUTPUT_NAMESPACE as ROUND4_MERGE_NAMESPACE,
)
from .sweep_incremental_round4_merge import ROUND4_MERGED_NAME
from .sweep_incremental_round5 import (
  EXPECTED_ROWS as EXPECTED_ROUND5_ROWS,
  REVIEWED_ROUND5_PLAN_SHA256,
)
from .sweep_incremental_round5_merge import (
  OUTPUT_NAMESPACE as ROUND5_MERGE_NAMESPACE,
)
from .sweep_incremental_round5_merge import ROUND5_MERGED_NAME
from .sweep_incremental_round6 import (
  EXPECTED_CANDIDATES as EXPECTED_ROUND6_CANDIDATES,
  EXPECTED_OPERATOR_UPDATES as EXPECTED_ROUND6_OPERATOR_UPDATES,
  EXPECTED_ROWS as EXPECTED_ROUND6_ROWS,
  REVIEWED_ROUND6_PLAN_SHA256,
)
from .sweep_incremental_round6_merge import (
  NEXT_UNION_RESULT_ROWS,
  OUTPUT_NAMESPACE as ROUND6_MERGE_NAMESPACE,
  REVIEWED_ROUND6_SHARD_SET_SHA256,
  ROUND6_MERGED_NAME,
)
from .sweep_selection import EXPECTED_ROWS as EXPECTED_ORIGINAL_ROWS
from .sweep_selection import EXPECTED_SELECTIONS, _CompensatedPair, _expected_parameters
from .sweep_sharded import line_ending_equivalent_sha256
from .sweep_union_selection import (
  EXPECTED_INCREMENTAL_SUBJECT_AGGREGATES,
  EXPECTED_ORIGINAL_SUBJECT_AGGREGATES,
  _clean_commit,
  _json,
  _plan_candidate_maps,
  _selection_parameters,
  _stream_source,
)
from .sweep_union_selection_round2 import (
  EXPECTED_ROUND2_SUBJECT_AGGREGATES,
  _expected_incremental_count_method_rows,
  _round2_plan_candidate_maps,
  _sum_method_counts,
)
from .sweep_union_selection_round3 import (
  EXPECTED_ROUND3_SUBJECT_AGGREGATES,
  _round3_plan_candidate_maps,
)
from .sweep_union_selection_round4 import (
  EXPECTED_ROUND4_SUBJECT_AGGREGATES,
  _round4_plan_candidate_maps,
)
from .sweep_union_selection_round5 import (
  EXPECTED_ROUND5_SUBJECT_AGGREGATES,
  EXPECTED_SIX_SOURCE_SUBJECT_AGGREGATES,
  _round5_plan_candidate_maps,
)

REVIEWED_ROUND5_UNION_SELECTION_AUDIT_SHA256 = (
  "d6657c2c6dd67a763e59dec0713fa84dffc450c514d0038b5741bc065fb9d906"
)
REVIEWED_ROUND6_MERGE_AUDIT_SHA256 = (
  "8476a5aad905258203767437dcc1ccf256f014e293dc1ef29d0aa9dff7dcf827"
)
EXPECTED_ROUND6_MERGED_CSV_SHA256 = (
  "85bc8c535bb22970fefabe68dde8e363cfb68bd370d4c7f6b1a9fe4071046fda"
)
EXPECTED_ROUND6_SUBJECT_AGGREGATES = 12
EXPECTED_SEVEN_SOURCE_SUBJECT_AGGREGATES = (
  EXPECTED_SIX_SOURCE_SUBJECT_AGGREGATES + EXPECTED_ROUND6_SUBJECT_AGGREGATES
)


def _round6_plan_candidate_maps(plan: dict) -> tuple[dict, dict, dict]:
  incremental: dict[tuple[str, int, str], set[tuple]] = {}
  expanded_axes: dict[tuple[str, int, str, tuple], tuple[str, ...]] = {}
  source_parameters: dict[tuple[str, int, str], tuple] = {}
  for group in plan.get("groups", []):
    identity = (
      str(group["dataset"]), int(group["count_level"]), str(group["method"]),
    )
    if identity in incremental:
      raise ValueError(f"duplicate round-6 expansion group: {identity}")
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
        raise ValueError(f"duplicate round-6 candidate: {identity + parameters}")
      axes = tuple(item["expanded_axes"])
      if not axes:
        raise ValueError(f"round-6 candidate has no expanded axis: {identity}")
      candidates.add(parameters)
      expanded_axes[(*identity, parameters)] = axes
    incremental[identity] = candidates
  if (
    len(incremental) != 3
    or sum(map(len, incremental.values())) != EXPECTED_ROUND6_CANDIDATES
  ):
    raise ValueError("round-6 selection requires the reviewed 3-group, 3-candidate shell")
  return incremental, expanded_axes, source_parameters


def stream_subject_balanced_round6_union_selection(
  original_path: Path, round1_path: Path, round2_path: Path, round3_path: Path,
  round4_path: Path, round5_path: Path, round6_path: Path, protocol,
  round1_plan: dict, round2_plan: dict, round3_plan: dict, round4_plan: dict,
  round5_plan: dict, round6_plan: dict, run_id: str,
) -> dict:
  """Stream seven reviewed CSVs and select over their disjoint candidate union."""
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
  round2_parameters, _, _ = _round2_plan_candidate_maps(round2_plan)
  round3_parameters, _, _ = _round3_plan_candidate_maps(round3_plan)
  round4_parameters, _, _ = _round4_plan_candidate_maps(round4_plan)
  round5_parameters, _, _ = _round5_plan_candidate_maps(round5_plan)
  round6_parameters, expanded_axes, source_parameters = (
    _round6_plan_candidate_maps(round6_plan)
  )
  union_parameters = {
    identity: set(values) for identity, values in original_parameters.items()
  }
  for label, shell in (
    ("round-1", round1_parameters),
    ("round-2", round2_parameters),
    ("round-3", round3_parameters),
    ("round-4", round4_parameters),
    ("round-5", round5_parameters),
    ("round-6", round6_parameters),
  ):
    for identity, candidates in shell.items():
      if identity not in union_parameters or candidates & union_parameters[identity]:
        raise ValueError(f"{label} shell overlaps a reviewed earlier domain: {identity}")
      union_parameters[identity].update(candidates)

  sources = (
    ("original", original_path, original_parameters),
    ("round1_incremental", round1_path, round1_parameters),
    ("round2_incremental", round2_path, round2_parameters),
    ("round3_incremental", round3_path, round3_parameters),
    ("round4_incremental", round4_path, round4_parameters),
    ("round5_incremental", round5_path, round5_parameters),
    ("round6_incremental", round6_path, round6_parameters),
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
          raise ValueError(f"seven-source candidate grid is incomplete for {identity}")
        scored = []
        for parameters, subject_values in candidates.items():
          if set(subject_values) != subjects:
            raise ValueError(
              f"seven-source subject coverage differs for {identity + parameters}"
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
  round6_selected = 0
  for selection in selected:
    identity = (
      selection["dataset"], int(selection["count_level"]), selection["method"],
    )
    parameters = _selection_parameters(selection)
    if identity in source_parameters:
      permitted = {source_parameters[identity]} | round6_parameters[identity]
      if parameters not in permitted:
        raise ValueError(
          f"seven-source winner is outside the reviewed round-6 shell: {identity}"
        )
      axes = expanded_axes.get((*identity, parameters), ())
      if axes:
        round6_selected += 1
        expanded_boundaries.append({
          "selection": selection, "expanded_boundary_axes": list(axes),
        })
        expanded_axis_counts.update(axes)

  source_method_counts = {
    source: {
      dataset: dict(sorted(method_counts[source][dataset].items()))
      for dataset in sorted(allowed_methods)
    }
    for source, _, _ in sources
  }
  union_method_counts = {
    dataset: dict(sorted(sum(
      (method_counts[source][dataset] for source, _, _ in sources), Counter(),
    ).items()))
    for dataset in sorted(allowed_methods)
  }
  return {
    **{f"{source}_rows_streamed": rows[source] for source, _, _ in sources},
    "union_rows_streamed": sum(rows.values()),
    **{
      f"{source}_subject_configuration_aggregates": aggregates[source]
      for source, _, _ in sources
    },
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
    "round6_incremental_candidate_selected_count": round6_selected,
    "source_boundary_winner_count": len(source_parameters) - round6_selected,
    "resolved_reviewed_boundary_group_count": len(source_parameters) - round6_selected,
    "expanded_boundary_selection_count": len(expanded_boundaries),
    "expanded_boundary_axis_counts": dict(sorted(expanded_axis_counts.items())),
    "expanded_boundary_selections": expanded_boundaries,
  }


def audit_round6_union_hyperparameter_selection(
  root: Path, config_path: Path, protocol_path: Path,
  round1_plan_path: Path, round2_plan_path: Path, round3_plan_path: Path,
  round4_plan_path: Path, round5_plan_path: Path, round5_union_audit_path: Path,
  round6_plan_path: Path, round6_merge_audit_path: Path, output_path: Path,
) -> dict:
  """Audit the seven-source union without a union CSV or frozen parameters."""
  root = root.resolve()
  paths = [
    config_path, protocol_path, round1_plan_path, round2_plan_path,
    round3_plan_path, round4_plan_path, round5_plan_path, round5_union_audit_path,
    round6_plan_path, round6_merge_audit_path, output_path,
  ]
  (
    config_path, protocol_path, round1_plan_path, round2_plan_path,
    round3_plan_path, round4_plan_path, round5_plan_path, round5_union_audit_path,
    round6_plan_path, round6_merge_audit_path, output_path,
  ) = (path.resolve() for path in paths)
  if output_path.exists():
    raise FileExistsError(f"refusing to overwrite round-6 union audit: {output_path}")
  audit_commit = _clean_commit(root)
  reviewed = (
    (round1_plan_path, REVIEWED_EXPANSION_PLAN_SHA256),
    (round2_plan_path, REVIEWED_ROUND2_PLAN_SHA256),
    (round3_plan_path, REVIEWED_ROUND3_PLAN_SHA256),
    (round4_plan_path, REVIEWED_ROUND4_PLAN_SHA256),
    (round5_plan_path, REVIEWED_ROUND5_PLAN_SHA256),
    (round5_union_audit_path, REVIEWED_ROUND5_UNION_SELECTION_AUDIT_SHA256),
    (round6_plan_path, REVIEWED_ROUND6_PLAN_SHA256),
    (round6_merge_audit_path, REVIEWED_ROUND6_MERGE_AUDIT_SHA256),
  )
  if any(file_sha256(path) != expected for path, expected in reviewed):
    raise ValueError("round-6 union inputs are not the reviewed artifacts")

  plans = [_json(path) for path in (
    round1_plan_path, round2_plan_path, round3_plan_path,
    round4_plan_path, round5_plan_path, round6_plan_path,
  )]
  round5_union_audit = _json(round5_union_audit_path)
  round6_merge_audit = _json(round6_merge_audit_path)
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
  round3_path = root / ROUND3_MERGE_NAMESPACE / ROUND3_MERGED_NAME
  round4_path = root / ROUND4_MERGE_NAMESPACE / ROUND4_MERGED_NAME
  round5_path = root / ROUND5_MERGE_NAMESPACE / ROUND5_MERGED_NAME
  round6_path = root / ROUND6_MERGE_NAMESPACE / ROUND6_MERGED_NAME
  source_paths = (
    original_path, round1_path, round2_path, round3_path,
    round4_path, round5_path, round6_path,
  )
  frozen_path = root / str(cfg["outputs"]["frozen_hyperparameters"])
  round6_plan = plans[-1]
  if (
    protocol.kind != "hyperparameter_sweep" or protocol.split != "validation"
    or protocol.raw.get("selection_unit") != "subject"
    or protocol.raw.get("selection_primary") != "mean_nrmse"
    or protocol.raw.get("selection_tiebreaker") != "mean_ssim"
    or protocol.raw.get("test_locked_access") != "prohibited"
    or not protocol.raw.get("expand_all_nonphysical_boundaries", False)
    or round5_union_audit.get("status")
    != "BLOCKED_expanded_boundary_requires_additional_shell"
    or round5_union_audit.get("expanded_boundary_selection_count") != 3
    or round5_union_audit.get("union_rows_streamed") != 5_349_609
    or round5_union_audit.get("selection_count") != EXPECTED_SELECTIONS
    or round5_union_audit.get("prior_registration") != registration
    or round6_plan.get("status") != "ready_for_researcher_review_no_execution"
    or round6_plan.get("expansion_round") != 6
    or round6_plan.get("source_union_selection_audit_sha256")
    != REVIEWED_ROUND5_UNION_SELECTION_AUDIT_SHA256
    or round6_plan.get("source_expanded_boundary_group_count") != 3
    or round6_plan.get("unchanged_union_selection_count") != 45
    or round6_plan.get("incremental_candidate_count") != EXPECTED_ROUND6_CANDIDATES
    or round6_plan.get("expected_incremental_result_rows") != EXPECTED_ROUND6_ROWS
    or round6_plan.get("expected_incremental_operator_updates")
    != EXPECTED_ROUND6_OPERATOR_UPDATES
    or round6_plan.get("expected_next_union_result_rows") != NEXT_UNION_RESULT_ROWS
    or round6_merge_audit.get("status")
    != "PASS_complete_pending_round6_union_selection_audit"
    or round6_merge_audit.get("round6_plan_sha256") != REVIEWED_ROUND6_PLAN_SHA256
    or round6_merge_audit.get("round6_shard_set_sha256")
    != REVIEWED_ROUND6_SHARD_SET_SHA256
    or round6_merge_audit.get("round5_union_selection_audit_sha256")
    != REVIEWED_ROUND5_UNION_SELECTION_AUDIT_SHA256
    or round6_merge_audit.get("round6_incremental_result_rows")
    != EXPECTED_ROUND6_ROWS
    or round6_merge_audit.get("round6_incremental_candidate_count")
    != EXPECTED_ROUND6_CANDIDATES
    or round6_merge_audit.get("round6_incremental_operator_updates")
    != EXPECTED_ROUND6_OPERATOR_UPDATES
    or round6_merge_audit.get("source_shards") != 741
    or round6_merge_audit.get("source_cases") != 8_892
    or round6_merge_audit.get("next_union_expected_result_rows")
    != NEXT_UNION_RESULT_ROWS
    or round6_merge_audit.get("empty_candidate_domain_verified") is not True
    or round6_merge_audit.get("zero_row_source_shard_count") != 45
    or round6_merge_audit.get("physical_union_created") is not False
    or round6_merge_audit.get("round6_union_selection_audit_created") is not False
    or round6_merge_audit.get("frozen_hyperparameters_created") is not False
    or round6_merge_audit.get("test_locked_access") != "prohibited"
    or not line_ending_equivalent_sha256(
      protocol_path, str(round5_union_audit.get("protocol_sha256")),
    )
    or file_sha256(config_path) != round5_union_audit.get("config_sha256")
    or frozen_path.exists()
  ):
    raise ValueError("round-6 union protocol or reviewed chain is inconsistent")

  source_artifacts = []
  for csv_path in source_paths:
    source_artifacts.extend((
      csv_path,
      csv_path.with_suffix(csv_path.suffix + ".manifest.json"),
      csv_path.parent / "merge_record.json",
      csv_path.parent / "merge_SUCCESS.json",
    ))
  if not all(path.is_file() for path in source_artifacts):
    raise FileNotFoundError("one or more reviewed seven-source artifacts are missing")
  round6_manifest = round6_path.with_suffix(round6_path.suffix + ".manifest.json")
  round6_record = round6_path.parent / "merge_record.json"
  round6_success = round6_path.parent / "merge_SUCCESS.json"
  round6_manifest_json = _json(round6_manifest)
  round6_record_json = _json(round6_record)
  round6_success_json = _json(round6_success)
  prior_labels = ("original", "round1", "round2", "round3", "round4", "round5")
  prior_csv_hashes = tuple(
    round5_union_audit[f"{label}_merged_csv_sha256"] for label in prior_labels
  )
  if (
    tuple(file_sha256(path) for path in source_paths[:6]) != prior_csv_hashes
    or tuple(
      round6_merge_audit.get(f"{label}_merged_csv_sha256")
      for label in prior_labels
    ) != prior_csv_hashes
    or round6_merge_audit.get("round6_merged_csv_sha256")
    != file_sha256(round6_path)
    or file_sha256(round6_path) != EXPECTED_ROUND6_MERGED_CSV_SHA256
    or round6_merge_audit.get("round6_merged_manifest_sha256")
    != file_sha256(round6_manifest)
    or round6_merge_audit.get("merge_success_sha256") != file_sha256(round6_success)
    or round6_manifest_json.get("artifact_sha256") != file_sha256(round6_path)
    or round6_manifest_json.get("row_count") != EXPECTED_ROUND6_ROWS
    or round6_manifest_json.get("columns") != SWEEP_FIELDS
    or round6_manifest_json.get("zero_row_source_shard_count") != 45
    or round6_record_json.get("round6_merged_csv_sha256") != file_sha256(round6_path)
    or round6_record_json.get("round6_shard_set_sha256")
    != REVIEWED_ROUND6_SHARD_SET_SHA256
    or round6_record_json.get("empty_candidate_domain_verified") is not True
    or round6_success_json.get("merge_record_sha256") != file_sha256(round6_record)
    or round6_success_json.get("round6_merged_manifest_sha256")
    != file_sha256(round6_manifest)
    or round6_success_json.get("round6_incremental_result_rows")
    != EXPECTED_ROUND6_ROWS
    or round6_success_json.get("zero_row_source_shard_count") != 45
  ):
    raise ValueError("seven-source artifacts differ from their reviewed audits")
  for csv_path in source_paths[1:]:
    actual_files = {path.name for path in csv_path.parent.iterdir() if path.is_file()}
    if actual_files != {
      csv_path.name, f"{csv_path.name}.manifest.json",
      "merge_record.json", "merge_SUCCESS.json",
    }:
      raise ValueError(f"merge namespace does not contain exactly four files: {csv_path}")

  selection = stream_subject_balanced_round6_union_selection(
    *source_paths, protocol, *plans, contract.run_id,
  )
  prior_winners = {
    (item["dataset"], int(item["count_level"]), item["method"]):
      _selection_parameters(item)
    for item in round5_union_audit["selected_hyperparameters"]
  }
  round6_groups = {
    (group["dataset"], int(group["count_level"]), group["method"])
    for group in round6_plan["groups"]
  }
  reviewed_round6_groups = {
    (
      item["selection"]["dataset"], int(item["selection"]["count_level"]),
      item["selection"]["method"],
    )
    for item in round5_union_audit["expanded_boundary_selections"]
  }
  if round6_groups != reviewed_round6_groups:
    raise ValueError("round-6 groups differ from reviewed six-source boundaries")
  for group in round6_plan["groups"]:
    identity = (group["dataset"], int(group["count_level"]), group["method"])
    if _selection_parameters(group["source_selected_parameters"]) != prior_winners[identity]:
      raise ValueError(f"round-6 source differs from prior union winner: {identity}")
  unchanged_exact = 0
  source_boundary_winners = 0
  for item in selection["selected_hyperparameters"]:
    identity = (item["dataset"], int(item["count_level"]), item["method"])
    parameters = _selection_parameters(item)
    if identity not in prior_winners:
      raise ValueError(f"seven-source selection produced an undeclared group: {identity}")
    if identity not in round6_groups:
      if parameters != prior_winners[identity]:
        raise ValueError(f"unchanged prior union selection changed: {identity}")
      unchanged_exact += 1
    elif parameters == prior_winners[identity]:
      source_boundary_winners += 1
  selection["unchanged_prior_union_selection_exact_match_count"] = unchanged_exact
  selection["source_boundary_winner_count"] = source_boundary_winners

  expected_round6_count_rows = _expected_incremental_count_method_rows(
    protocol, contract.count_levels, round6_plan,
  )
  expected_union_method_counts = _sum_method_counts(
    round5_union_audit["union_method_row_counts"],
    round6_merge_audit["method_row_counts"],
  )
  prior_sources = round5_union_audit["source_method_row_counts"]
  prior_count_sources = round5_union_audit["source_dataset_count_method_row_counts"]
  expected_rows = (
    EXPECTED_ORIGINAL_ROWS, EXPECTED_ROUND1_ROWS, EXPECTED_ROUND2_ROWS,
    EXPECTED_ROUND3_ROWS, EXPECTED_ROUND4_ROWS, EXPECTED_ROUND5_ROWS,
  )
  if (
    tuple(selection[f"{label}_rows_streamed"] for label in (
      "original", "round1_incremental", "round2_incremental",
      "round3_incremental", "round4_incremental", "round5_incremental",
    )) != expected_rows
    or selection["round6_incremental_rows_streamed"] != EXPECTED_ROUND6_ROWS
    or selection["union_rows_streamed"] != NEXT_UNION_RESULT_ROWS
    or selection["original_subject_configuration_aggregates"]
    != EXPECTED_ORIGINAL_SUBJECT_AGGREGATES
    or selection["round1_incremental_subject_configuration_aggregates"]
    != EXPECTED_INCREMENTAL_SUBJECT_AGGREGATES
    or selection["round2_incremental_subject_configuration_aggregates"]
    != EXPECTED_ROUND2_SUBJECT_AGGREGATES
    or selection["round3_incremental_subject_configuration_aggregates"]
    != EXPECTED_ROUND3_SUBJECT_AGGREGATES
    or selection["round4_incremental_subject_configuration_aggregates"]
    != EXPECTED_ROUND4_SUBJECT_AGGREGATES
    or selection["round5_incremental_subject_configuration_aggregates"]
    != EXPECTED_ROUND5_SUBJECT_AGGREGATES
    or selection["round6_incremental_subject_configuration_aggregates"]
    != EXPECTED_ROUND6_SUBJECT_AGGREGATES
    or selection["union_subject_configuration_aggregates"]
    != EXPECTED_SEVEN_SOURCE_SUBJECT_AGGREGATES
    or selection["dataset_subject_counts"] != {"brainweb": 4, "hecktor": 45}
    or selection["dataset_count_levels"] != {
      dataset: sorted(contract.count_levels) for dataset in ("brainweb", "hecktor")
    }
    or any(
      selection["source_method_row_counts"][source] != prior_sources[source]
      for source in prior_sources
    )
    or selection["source_method_row_counts"]["round6_incremental"]
    != round6_merge_audit["method_row_counts"]
    or any(
      selection["source_dataset_count_method_row_counts"][source]
      != prior_count_sources[source]
      for source in prior_count_sources
    )
    or selection["source_dataset_count_method_row_counts"]["round6_incremental"]
    != expected_round6_count_rows
    or selection["union_method_row_counts"] != expected_union_method_counts
    or selection["selection_count"] != EXPECTED_SELECTIONS
    or selection["dataset_selection_counts"] != {"brainweb": 24, "hecktor": 24}
    or selection["reviewed_boundary_group_count"] != 3
    or selection["unchanged_prior_union_selection_count"] != 45
    or selection["unchanged_prior_union_selection_exact_match_count"] != 45
    or selection["source_boundary_winner_count"]
    + selection["round6_incremental_candidate_selected_count"] != 3
  ):
    raise ValueError("seven-source selection counts differ from reviewed inputs")

  boundary_required = selection["expanded_boundary_selection_count"] > 0
  result = {
    "artifact_kind": "validation_hyperparameter_round6_union_selection_audit",
    "schema_version": 1,
    "status": (
      "BLOCKED_expanded_boundary_requires_additional_shell"
      if boundary_required else "PASS_ready_for_hyperparameter_freeze"
    ),
    "read_only": True,
    "audit_git_commit": audit_commit,
    "config_sha256": file_sha256(config_path),
    "protocol_sha256": file_sha256(protocol_path),
    "round5_union_selection_audit_sha256": (
      REVIEWED_ROUND5_UNION_SELECTION_AUDIT_SHA256
    ),
    "round6_expansion_plan_sha256": REVIEWED_ROUND6_PLAN_SHA256,
    "round6_merge_audit_sha256": REVIEWED_ROUND6_MERGE_AUDIT_SHA256,
    **{
      f"{label}_merged_csv_sha256": file_sha256(path)
      for label, path in zip(
        ("original", "round1", "round2", "round3", "round4", "round5", "round6"),
        source_paths,
      )
    },
    "round6_merged_manifest_sha256": file_sha256(round6_manifest),
    "round6_merge_record_sha256": file_sha256(round6_record),
    "round6_merge_success_sha256": file_sha256(round6_success),
    "round6_shard_set_sha256": REVIEWED_ROUND6_SHARD_SET_SHA256,
    "prior_registration": registration,
    "selection_unit": "subject",
    "selection_primary": "mean_nrmse",
    "selection_tiebreaker": "mean_ssim",
    "selection_domain": (
      "reviewed_original_grid_union_round1_through_round6_local_shells"
    ),
    "streaming_aggregation": "kahan_per_subject_then_fsum_across_subjects",
    "adaptive_stopping_rule": round6_plan["adaptive_stopping_rule"],
    "expanded_boundary_policy": "only_round6_new_outward_values_trigger_another_shell",
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
