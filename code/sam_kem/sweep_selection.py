"""Read-only, streaming hyperparameter selection from an audited sweep merge."""

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
from .sweep_evaluation import SWEEP_FIELDS, SweepGrid, parse_sweep_grid


REVIEWED_MERGE_AUDIT_SHA256 = (
  "7cd8928b8bb989cc9f7282e9a3b71cb738baba810e33e4ff8e5364f8fa0a9376"
)
EXPECTED_ROWS = 5_157_360
EXPECTED_SELECTIONS = 48


class _CompensatedPair:
  """Two Kahan accumulators plus a shared observation count."""

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


def _json(path: Path) -> dict:
  value = json.loads(path.read_text(encoding="utf-8"))
  if not isinstance(value, dict):
    raise ValueError(f"expected a JSON object: {path}")
  return value


def _clean_commit(root: Path) -> str:
  status = subprocess.run(
    ["git", "status", "--porcelain"], cwd=root, check=True,
    capture_output=True, text=True,
  ).stdout.strip()
  if status:
    raise RuntimeError(
      f"hyperparameter selection audit requires a clean worktree: {status.splitlines()}"
    )
  return subprocess.run(
    ["git", "rev-parse", "HEAD"], cwd=root, check=True,
    capture_output=True, text=True,
  ).stdout.strip()


def _parameter_key(row: dict[str, str]) -> tuple[float, int, int, int, float]:
  return (
    float(row["sigma"]), int(row["radius"]), int(row["neighbours"]),
    int(row["iterations"]), float(row["alpha"]),
  )


def _expected_parameters(method: str, grid: SweepGrid) -> set[tuple]:
  if method == "MLEM":
    return {(0.0, 0, 0, value, 0.0) for value in grid.mlem_iterations}
  if method == "OSEM":
    return {(0.0, 0, 0, value, 0.0) for value in grid.osem_cycles}
  if method == "Bowsher-KEM":
    return {
      (0.0, radius, neighbours, iterations, 0.0)
      for radius in grid.bowsher_radii
      for neighbours in grid.bowsher_neighbours
      for iterations in grid.iterations
    }
  alphas = grid.alphas if method.startswith("SAM") else (0.0,)
  return {
    (sigma, radius, 0, iterations, alpha)
    for sigma in grid.sigmas
    for radius in grid.radii
    for iterations in grid.iterations
    for alpha in alphas
  }


def _boundary_axes(selection: dict, grid: SweepGrid) -> list[str]:
  axes = []
  method = selection["method"]
  iteration_grid = (
    grid.mlem_iterations if method == "MLEM" else
    grid.osem_cycles if method == "OSEM" else grid.iterations
  )
  if selection["iterations"] == max(iteration_grid):
    axes.append("iterations")
  elif selection["iterations"] == min(iteration_grid) and min(iteration_grid) > 1:
    axes.append("iterations_lower")
  if method in {"MLEM", "OSEM"}:
    return axes
  if method == "Bowsher-KEM":
    if selection["neighbours"] == max(grid.bowsher_neighbours):
      axes.append("bowsher_neighbours")
    elif (
      selection["neighbours"] == min(grid.bowsher_neighbours)
      and min(grid.bowsher_neighbours) > 1
    ):
      axes.append("bowsher_neighbours_lower")
    if selection["radius"] == max(grid.bowsher_radii):
      axes.append("bowsher_radius_upper")
    return axes
  if selection["sigma"] in {min(grid.sigmas), max(grid.sigmas)}:
    axes.append("sigma")
  if selection["radius"] == max(grid.radii):
    axes.append("radius_upper")
  if method.startswith("SAM") and selection["alpha"] == max(grid.alphas):
    axes.append("alpha_upper")
  return axes


def stream_subject_balanced_selection(
  sweep_path: Path, protocol, run_id: str,
) -> dict:
  """Stream rows and reproduce subject-balanced NRMSE/SSIM selection."""
  grid = parse_sweep_grid(protocol)
  allowed_methods = {
    "brainweb": tuple(method for method in protocol.methods if method != "CT-KEM"),
    "hecktor": tuple(method for method in protocol.methods if method != "MR-KEM"),
  }
  expected_by_method = {
    method: _expected_parameters(method, grid)
    for methods in allowed_methods.values() for method in methods
  }
  subject_totals: dict[tuple, _CompensatedPair] = {}
  method_row_counts: dict[str, Counter] = defaultdict(Counter)
  count_method_row_counts: dict[tuple[str, int, str], int] = defaultdict(int)
  dataset_subjects: dict[str, set[str]] = defaultdict(set)
  count_levels: dict[str, set[int]] = defaultdict(set)
  rows_streamed = 0
  with sweep_path.open("r", encoding="utf-8", newline="") as handle:
    reader = csv.DictReader(handle)
    if list(reader.fieldnames or ()) != SWEEP_FIELDS:
      raise ValueError("merged sweep CSV columns differ from the frozen sweep schema")
    for row in reader:
      rows_streamed += 1
      dataset = row["dataset"]
      method = row["method"]
      if (
        row["run_id"] != run_id or row["split"] != "validation"
        or dataset not in allowed_methods or method not in allowed_methods[dataset]
      ):
        raise ValueError(f"invalid sweep row identity at data row {rows_streamed}")
      count = int(row["count_level"])
      parameters = _parameter_key(row)
      if parameters not in expected_by_method[method]:
        raise ValueError(f"undeclared parameter tuple at data row {rows_streamed}")
      nrmse = float(row["nrmse"])
      ssim = float(row["ssim"])
      if not math.isfinite(nrmse) or not math.isfinite(ssim):
        raise ValueError(f"non-finite validation metric at data row {rows_streamed}")
      subject = row["subject_id"]
      key = (dataset, count, method, *parameters, subject)
      aggregate = subject_totals.get(key)
      if aggregate is None:
        aggregate = subject_totals[key] = _CompensatedPair()
      aggregate.add(nrmse, ssim)
      method_row_counts[dataset][method] += 1
      count_method_row_counts[(dataset, count, method)] += 1
      dataset_subjects[dataset].add(subject)
      count_levels[dataset].add(count)

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
        expected_parameters = expected_by_method[method]
        candidates = {
          key[3:]: values for key, values in candidate_subjects.items()
          if key[:3] == (dataset, count, method)
        }
        if set(candidates) != expected_parameters:
          raise ValueError(f"candidate grid is incomplete for {(dataset, count, method)}")
        scored = []
        for parameters, subject_values in candidates.items():
          if set(subject_values) != subjects:
            raise ValueError(
              f"subject coverage differs for {(dataset, count, method, parameters)}"
            )
          mean_nrmse = math.fsum(value[0] for value in subject_values.values()) / len(subjects)
          mean_ssim = math.fsum(value[1] for value in subject_values.values()) / len(subjects)
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

  boundaries = []
  axis_counts: Counter = Counter()
  for selection in selected:
    axes = _boundary_axes(selection, grid)
    if axes:
      boundaries.append({"selection": selection, "boundary_axes": axes})
      axis_counts.update(axes)
  return {
    "rows_streamed": rows_streamed,
    "subject_configuration_aggregates": len(subject_totals),
    "dataset_subject_counts": {
      dataset: len(subjects) for dataset, subjects in sorted(dataset_subjects.items())
    },
    "dataset_count_levels": {
      dataset: sorted(values) for dataset, values in sorted(count_levels.items())
    },
    "method_row_counts": {
      dataset: dict(sorted(counts.items()))
      for dataset, counts in sorted(method_row_counts.items())
    },
    "dataset_count_method_row_counts": {
      dataset: {
        str(count): {
          method: count_method_row_counts[(dataset, count, method)]
          for method in allowed_methods[dataset]
        }
        for count in sorted(count_levels[dataset])
      }
      for dataset in sorted(allowed_methods)
    },
    "selection_count": len(selected),
    "dataset_selection_counts": dict(sorted(selection_counts.items())),
    "selected_hyperparameters": selected,
    "boundary_selection_count": len(boundaries),
    "boundary_axis_counts": dict(sorted(axis_counts.items())),
    "boundary_selections": boundaries,
  }


def audit_streaming_hyperparameter_selection(
  root: Path, config_path: Path, protocol_path: Path, merge_audit_path: Path,
  output_path: Path,
) -> dict:
  """Verify the merge and audit selection without creating frozen parameters."""
  root = root.resolve()
  config_path = config_path.resolve()
  protocol_path = protocol_path.resolve()
  merge_audit_path = merge_audit_path.resolve()
  output_path = output_path.resolve()
  if output_path.exists():
    raise FileExistsError(f"refusing to overwrite selection audit: {output_path}")
  audit_commit = _clean_commit(root)
  if file_sha256(merge_audit_path) != REVIEWED_MERGE_AUDIT_SHA256:
    raise ValueError("merge audit is not the researcher-reviewed artifact")
  merge_audit = _json(merge_audit_path)
  cfg, contract = load_contract(config_path)
  require_development_execution(cfg)
  registration = validate_registered_validation_prior_configuration(root, config_path, cfg)
  protocol = load_protocol(protocol_path)
  sweep_path = root / str(cfg["outputs"]["hyperparameter_sweep"])
  frozen_path = root / str(cfg["outputs"]["frozen_hyperparameters"])
  manifest_path = sweep_path.with_suffix(sweep_path.suffix + ".manifest.json")
  record_path = sweep_path.parent / "merge_record.json"
  success_path = sweep_path.parent / "merge_SUCCESS.json"
  if (
    protocol.kind != "hyperparameter_sweep" or protocol.split != "validation"
    or protocol.raw.get("selection_unit") != "subject"
    or protocol.raw.get("selection_primary") != "mean_nrmse"
    or protocol.raw.get("selection_tiebreaker") != "mean_ssim"
    or protocol.raw.get("test_locked_access") != "prohibited"
    or not protocol.raw.get("expand_all_nonphysical_boundaries", False)
    or sweep_path != root / protocol.output / "validation_sweep.csv"
    or frozen_path.exists()
  ):
    raise ValueError("selection protocol or frozen-output state is inconsistent")
  if not all(path.is_file() for path in (sweep_path, manifest_path, record_path, success_path)):
    raise FileNotFoundError("one or more formal merge artifacts are missing")
  manifest = _json(manifest_path)
  record = _json(record_path)
  success = _json(success_path)
  if (
    merge_audit.get("status") != "PASS_complete_pending_hyperparameter_freeze"
    or merge_audit.get("merged_result_rows") != EXPECTED_ROWS
    or merge_audit.get("merged_csv_sha256") != file_sha256(sweep_path)
    or merge_audit.get("merged_manifest_sha256") != file_sha256(manifest_path)
    or merge_audit.get("merge_success_sha256") != file_sha256(success_path)
    or manifest.get("artifact_sha256") != merge_audit.get("merged_csv_sha256")
    or manifest.get("row_count") != EXPECTED_ROWS
    or manifest.get("columns") != SWEEP_FIELDS
    or record.get("merged_csv_sha256") != merge_audit.get("merged_csv_sha256")
    or record.get("shard_set_sha256") != merge_audit.get("shard_set_sha256")
    or success.get("merge_record_sha256") != file_sha256(record_path)
    or success.get("merged_manifest_sha256") != file_sha256(manifest_path)
    or success.get("row_count") != EXPECTED_ROWS
    or merge_audit.get("test_locked_access") != "prohibited"
  ):
    raise ValueError("formal merge artifacts differ from the reviewed merge audit")
  selection = stream_subject_balanced_selection(sweep_path, protocol, contract.run_id)
  expected_case_counts = {"brainweb": 2_088, "hecktor": 135}
  grid = parse_sweep_grid(protocol)
  expected_count_method_rows = {
    dataset: {
      str(count): {
        method: expected_case_counts[dataset] * len(_expected_parameters(method, grid))
        for method in protocol.methods
        if not (dataset == "brainweb" and method == "CT-KEM")
        and not (dataset == "hecktor" and method == "MR-KEM")
      }
      for count in contract.count_levels
    }
    for dataset in ("brainweb", "hecktor")
  }
  if (
    selection["rows_streamed"] != EXPECTED_ROWS
    or selection["selection_count"] != EXPECTED_SELECTIONS
    or selection["dataset_selection_counts"] != {"brainweb": 24, "hecktor": 24}
    or selection["dataset_subject_counts"] != {"brainweb": 4, "hecktor": 45}
    or selection["dataset_count_levels"] != {
      dataset: sorted(contract.count_levels) for dataset in ("brainweb", "hecktor")
    }
    or selection["dataset_count_method_row_counts"] != expected_count_method_rows
    or selection["method_row_counts"] != merge_audit.get("method_row_counts")
  ):
    raise ValueError("streamed selection counts differ from the audited merge")
  boundary_required = bool(selection["boundary_selections"])
  result = {
    "artifact_kind": "validation_hyperparameter_selection_audit",
    "schema_version": 1,
    "status": (
      "BLOCKED_grid_boundary_expansion_required"
      if boundary_required else "PASS_ready_for_hyperparameter_freeze"
    ),
    "read_only": True,
    "audit_git_commit": audit_commit,
    "config_sha256": file_sha256(config_path),
    "protocol_sha256": file_sha256(protocol_path),
    "merge_audit_sha256": REVIEWED_MERGE_AUDIT_SHA256,
    "merged_csv_sha256": file_sha256(sweep_path),
    "merged_manifest_sha256": file_sha256(manifest_path),
    "merge_record_sha256": file_sha256(record_path),
    "merge_success_sha256": file_sha256(success_path),
    "shard_set_sha256": merge_audit["shard_set_sha256"],
    "prior_registration": registration,
    "selection_unit": "subject",
    "selection_primary": "mean_nrmse",
    "selection_tiebreaker": "mean_ssim",
    "streaming_aggregation": "kahan_per_subject_then_fsum_across_subjects",
    "boundary_policy": "expand_all_nonphysical_boundaries",
    **selection,
    "frozen_hyperparameters_created": False,
    "test_locked_access": "prohibited",
  }
  atomic_write_json(output_path, result)
  return result
