"""Validation-only SAM-KEM hyperparameter sweep and freezing."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np

from .config import load_contract, require_development_execution
from .experiments import load_case_manifest, validate_case_contract
from .operator_factory import build_discrete_radon
from .protocols import load_protocol
from .prior_registration import validate_registered_validation_prior_configuration
from .reconstruction import operator_support_mask
from .runmeta import write_run_metadata
from .artifacts import atomic_csv_writer, finalize_csv, write_success_marker
from .sweep_evaluation import SWEEP_FIELDS, evaluate_validation_case, parse_sweep_grid


def _mean_by_subject(records: list[dict], field: str) -> float:
  values: dict[str, list[float]] = defaultdict(list)
  for record in records:
    values[record["subject_id"]].append(float(record[field]))
  return float(np.mean([np.mean(subject_values) for subject_values in values.values()]))


def run_validation_sweep(root: Path, config_path: Path, protocol_path: Path) -> tuple[Path, Path]:
  cfg, contract = load_contract(config_path)
  require_development_execution(cfg)
  validate_registered_validation_prior_configuration(root, config_path, cfg)
  protocol = load_protocol(protocol_path)
  if protocol.kind != "hyperparameter_sweep" or protocol.split != "validation":
    raise ValueError("sweep runner accepts validation hyperparameter_sweep protocol only")
  cases = load_case_manifest(
    root, root / str(cfg["outputs"]["validation_manifest"]), split="validation"
  )
  if not cases or any(case.sam_prior_path is None for case in cases):
    raise ValueError("every validation case must have a real cached SAM prior")
  validate_case_contract(cases, contract)
  grid = parse_sweep_grid(protocol)
  sigmas = grid.sigmas
  radii = grid.radii
  iteration_candidates = grid.iterations
  alphas = grid.alphas
  bowsher_neighbours = grid.bowsher_neighbours
  bowsher_radii = grid.bowsher_radii
  mlem_candidates = grid.mlem_iterations
  osem_candidates = grid.osem_cycles
  operator = build_discrete_radon(
    repo_root=root, size=int(cfg["forward_model"]["image_size"]), angles=contract.angles,
    operator_id=contract.operator_id,
    expected_implementation_sha256=cfg["forward_model"]["implementation_sha256"],
  )
  operator_support = operator_support_mask(operator)
  output_dir = root / protocol.output
  output_dir.mkdir(parents=True, exist_ok=True)
  run_metadata = write_run_metadata(output_dir, root=root, config_path=config_path, protocol_path=protocol_path)
  raw_path = output_dir / "validation_sweep.csv"
  records: list[dict] = []
  with atomic_csv_writer(raw_path, SWEEP_FIELDS) as writer:
    for case in cases:
      case_records = evaluate_validation_case(
        case, operator, operator_support, cfg, contract.run_id, protocol.methods, grid,
      )
      writer.writerows(case_records)
      records.extend(case_records)

  grouped: dict[tuple, list[dict]] = defaultdict(list)
  for record in records:
    key = (
      record["dataset"], record["count_level"], record["method"], record["sigma"],
      record["radius"], record["neighbours"], record["iterations"], record["alpha"],
    )
    grouped[key].append(record)
  selected = []
  for dataset, count, method in sorted({(r["dataset"], r["count_level"], r["method"]) for r in records}):
    candidates = []
    for key, group in grouped.items():
      if key[:3] == (dataset, count, method):
        candidates.append((_mean_by_subject(group, "nrmse"), _mean_by_subject(group, "ssim"), key))
    score, mean_ssim, key = min(candidates, key=lambda item: (item[0], -item[1], item[2][3:]))
    selected.append({
      "run_id": contract.run_id, "dataset": dataset,
      "count_level": count, "method": method, "sigma": key[3],
      "radius": key[4], "neighbours": key[5], "iterations": key[6], "alpha": key[7],
      "validation_subject_mean_nrmse": score,
      "validation_subject_mean_ssim": mean_ssim,
    })
  boundary = []
  for row in selected:
    axes = []
    method_iteration_grid = (
      mlem_candidates if row["method"] == "MLEM" else
      osem_candidates if row["method"] == "OSEM" else iteration_candidates
    )
    if row["iterations"] == max(method_iteration_grid):
      axes.append("iterations")
    elif row["iterations"] == min(method_iteration_grid) and min(method_iteration_grid) > 1:
      axes.append("iterations_lower")
    if row["method"] in {"MLEM", "OSEM"}:
      pass
    elif row["method"] == "Bowsher-KEM":
      if row["neighbours"] == max(bowsher_neighbours):
        axes.append("bowsher_neighbours")
      elif row["neighbours"] == min(bowsher_neighbours) and min(bowsher_neighbours) > 1:
        axes.append("bowsher_neighbours_lower")
      if row["radius"] == max(bowsher_radii):
        axes.append("bowsher_radius_upper")
    else:
      if row["sigma"] in {min(sigmas), max(sigmas)}:
        axes.append("sigma")
      if row["radius"] == max(radii):
        axes.append("radius_upper")
    if row["method"].startswith("SAM") and row["alpha"] == max(alphas):
      axes.append("alpha_upper")
    if axes:
      boundary.append({"selection": row, "boundary_axes": axes})
  if boundary and protocol.raw.get("expand_all_nonphysical_boundaries", False):
    raise RuntimeError(
      "validation optimum is on a searchable grid boundary; extend the named axes "
      f"and rerun before freezing: {boundary}"
    )
  frozen_path = root / str(cfg["outputs"]["frozen_hyperparameters"])
  frozen_path.parent.mkdir(parents=True, exist_ok=True)
  with atomic_csv_writer(frozen_path, list(selected[0])) as writer:
    writer.writerows(selected)
  raw_manifest = finalize_csv(
    raw_path, schema_name="validation_hyperparameter_sweep_v2",
    primary_key=[
      "run_id", "dataset", "subject_id", "slice_id", "realization_id", "count_level",
      "method", "sigma", "radius", "neighbours", "iterations", "alpha",
    ], split="validation",
  )
  frozen_manifest = finalize_csv(
    frozen_path, schema_name="frozen_hyperparameters_v2",
    primary_key=["run_id", "dataset", "count_level", "method"], split="validation",
  )
  write_success_marker(output_dir, [
    run_metadata, output_dir / "environment_freeze.txt", raw_path, raw_manifest,
    frozen_path, frozen_manifest,
  ])
  return raw_path, frozen_path
