"""Validation-only SAM-KEM hyperparameter sweep and freezing."""

from __future__ import annotations

from collections import defaultdict
from itertools import product
from pathlib import Path

import numpy as np

from .config import load_contract, require_development_execution
from .experiments import load_case_manifest, validate_case_contract
from .metrics import nrmse, ssim
from .operator_factory import build_discrete_radon
from .protocols import load_protocol
from .prior_registration import validate_registered_validation_prior_configuration
from .reconstruction import bowsher_kernel, kem, mlem, normalize_kernel_feature, operator_support_mask, osem, rbf_kernel
from .runmeta import write_run_metadata
from .artifacts import atomic_csv_writer, finalize_csv, write_success_marker


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
  search = protocol.raw["search_space"]
  sigmas = [float(x) for x in search["sigma"]]
  radii = [int(x) for x in search["radius"]]
  iteration_candidates = [int(x) for x in search["iterations"]]
  alphas = [float(x) for x in search["alpha"]]
  bowsher_neighbours = [int(x) for x in search["bowsher_neighbours"]]
  bowsher_radii = [int(x) for x in search["bowsher_radius"]]
  mlem_candidates = [int(x) for x in search["mlem_iterations"]]
  osem_candidates = [int(x) for x in search["osem_cycles"]]
  if (
    any(value <= 0 for value in sigmas)
    or any(value < 0 for value in radii)
    or any(value <= 0 for value in iteration_candidates + mlem_candidates + osem_candidates)
    or any(not 0.0 <= value <= 1.0 for value in alphas)
    or any(value <= 0 for value in bowsher_neighbours)
    or any(value <= 0 for value in bowsher_radii)
  ):
    raise ValueError("hyperparameter grids contain values outside their physical domains")
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
  fields = [
    "run_id", "dataset", "subject_id", "slice_id", "realization_id", "count_level", "method", "split",
    "sigma", "radius", "neighbours", "iterations", "osem_subsets",
    "effective_em_updates", "alpha", "nrmse", "ssim",
  ]
  records: list[dict] = []
  with atomic_csv_writer(raw_path, fields) as writer:
    for case in cases:
      measured = np.load(case.sinogram_path)
      target = np.load(case.target_path)
      if case.mask_path is not None:
        raise ValueError("external evaluation masks are prohibited in the clean protocol")
      mask = operator_support
      prior = np.load(case.sam_prior_path)
      physical_prior = np.clip(
        prior, cfg["normalization"]["expected_target_range"][0],
        cfg["normalization"]["expected_target_range"][1],
      )
      gain = np.full_like(measured, case.projection_scale, dtype=np.float64)
      mlem_count = int(cfg["reconstruction"]["prompt_mlem_iterations"])
      mlem_image = mlem(measured, operator, iterations=mlem_count, multiplicative=gain)[mlem_count]
      for method in protocol.methods:
        if method == "MR-KEM" and case.dataset != "brainweb": continue
        if method == "CT-KEM" and case.dataset != "hecktor": continue
        if method in {"MLEM", "OSEM"}:
          baseline_candidates = mlem_candidates if method == "MLEM" else osem_candidates
          baseline_images = (
            mlem(
              measured, operator, iterations=max(baseline_candidates),
              save_iterations=baseline_candidates, multiplicative=gain,
            )
            if method == "MLEM" else
            osem(
              measured, operator, iterations=max(baseline_candidates),
              subsets=int(cfg["reconstruction"]["osem_subsets"]),
              save_iterations=baseline_candidates, multiplicative=gain,
            )
          )
          for iterations in baseline_candidates:
            image = baseline_images[iterations]
            record = {
              "run_id": contract.run_id, "dataset": case.dataset,
              "subject_id": case.subject_id,
              "slice_id": case.slice_id, "realization_id": case.realization_id,
              "count_level": case.count_level, "method": method, "split": "validation",
              "sigma": 0.0, "radius": 0, "neighbours": 0,
              "iterations": iterations, "alpha": 0.0,
              "osem_subsets": int(cfg["reconstruction"]["osem_subsets"]) if method == "OSEM" else 1,
              "effective_em_updates": (
                iterations * int(cfg["reconstruction"]["osem_subsets"])
                if method == "OSEM" else iterations
              ),
              "nrmse": nrmse(target, image, mask),
              "ssim": ssim(target, image, data_range=1.0, mask=mask),
            }
            writer.writerow(record); records.append(record)
          continue
        if method == "SelfKEM":
          feature = normalize_kernel_feature(mlem_image)
        elif method in {"MR-KEM", "CT-KEM", "Bowsher-KEM"}:
          if case.anatomy_path is None: raise FileNotFoundError(f"missing anatomy baseline input: {case}")
          feature = normalize_kernel_feature(np.load(case.anatomy_path))
        elif method == "SAM-KEM-MedSAM2-Qwen3-full":
          feature = normalize_kernel_feature(prior)
        else:
          raise ValueError(f"unsupported sweep method: {method}")
        method_alphas = alphas if method.startswith("SAM") else [0.0]
        kernel_settings = (
          [(0.0, radius, neighbours) for radius, neighbours in product(bowsher_radii, bowsher_neighbours)]
          if method == "Bowsher-KEM"
          else [(sigma, radius, 0) for sigma, radius in product(sigmas, radii)]
        )
        for sigma, radius, neighbours in kernel_settings:
          kernel = (
            bowsher_kernel(
              feature, neighbours=neighbours, radius=radius, support=operator_support,
            )
            if method == "Bowsher-KEM"
            else rbf_kernel(
              feature, sigma=sigma, radius=radius, support=operator_support,
            )
          )
          iterates = kem(
            measured, operator, kernel, iterations=max(iteration_candidates),
            save_iterations=iteration_candidates, multiplicative=gain,
          )
          for iterations, alpha in product(iteration_candidates, method_alphas):
            image = alpha * physical_prior + (1.0 - alpha) * iterates[iterations] if method.startswith("SAM") else iterates[iterations]
            tolerance = float(cfg["evaluation"]["nonnegative_tolerance"])
            if not np.all(np.isfinite(image)) or float(np.nanmin(image)) < -tolerance:
              raise ValueError(
                f"invalid negative/non-finite PET estimate during validation: "
                f"{method}, alpha={alpha}, min={float(np.nanmin(image)):.6g}; "
                "the raw-prior/post-fusion definition must be resolved before selection"
              )
            image = np.maximum(image, 0.0)
            record = {
              "run_id": contract.run_id, "dataset": case.dataset,
              "subject_id": case.subject_id,
              "slice_id": case.slice_id, "realization_id": case.realization_id,
              "count_level": case.count_level, "method": method, "split": "validation",
              "sigma": sigma, "radius": radius, "iterations": iterations,
              "neighbours": neighbours, "osem_subsets": 1,
              "effective_em_updates": iterations, "alpha": alpha,
              "nrmse": nrmse(target, image, mask),
              "ssim": ssim(target, image, data_range=1.0, mask=mask),
            }
            writer.writerow(record)
            records.append(record)

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
