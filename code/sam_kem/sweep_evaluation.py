"""One exact validation-case evaluator shared by sequential and sharded sweeps."""

from __future__ import annotations

from dataclasses import dataclass
from collections import defaultdict
from itertools import product

import numpy as np

from .experiments import Case
from .metrics import nrmse, ssim
from .reconstruction import (
  bowsher_kernel,
  kem,
  mlem,
  normalize_kernel_feature,
  osem,
  rbf_kernel,
)


SWEEP_FIELDS = [
  "run_id", "dataset", "subject_id", "slice_id", "realization_id", "count_level",
  "method", "split", "sigma", "radius", "neighbours", "iterations",
  "osem_subsets", "effective_em_updates", "alpha", "nrmse", "ssim",
]


@dataclass(frozen=True)
class SweepGrid:
  sigmas: tuple[float, ...]
  radii: tuple[int, ...]
  iterations: tuple[int, ...]
  alphas: tuple[float, ...]
  bowsher_neighbours: tuple[int, ...]
  bowsher_radii: tuple[int, ...]
  mlem_iterations: tuple[int, ...]
  osem_cycles: tuple[int, ...]


def parse_sweep_grid(protocol) -> SweepGrid:
  search = protocol.raw["search_space"]
  grid = SweepGrid(
    sigmas=tuple(float(value) for value in search["sigma"]),
    radii=tuple(int(value) for value in search["radius"]),
    iterations=tuple(int(value) for value in search["iterations"]),
    alphas=tuple(float(value) for value in search["alpha"]),
    bowsher_neighbours=tuple(int(value) for value in search["bowsher_neighbours"]),
    bowsher_radii=tuple(int(value) for value in search["bowsher_radius"]),
    mlem_iterations=tuple(int(value) for value in search["mlem_iterations"]),
    osem_cycles=tuple(int(value) for value in search["osem_cycles"]),
  )
  if (
    any(value <= 0 for value in grid.sigmas)
    or any(value < 0 for value in grid.radii)
    or any(value <= 0 for value in (
      *grid.iterations, *grid.mlem_iterations, *grid.osem_cycles,
    ))
    or any(not 0.0 <= value <= 1.0 for value in grid.alphas)
    or any(value <= 0 for value in (*grid.bowsher_neighbours, *grid.bowsher_radii))
  ):
    raise ValueError("hyperparameter grids contain values outside their physical domains")
  return grid


def _base_record(case: Case, run_id: str, method: str) -> dict:
  return {
    "run_id": run_id, "dataset": case.dataset, "subject_id": case.subject_id,
    "slice_id": case.slice_id, "realization_id": case.realization_id,
    "count_level": case.count_level, "method": method, "split": "validation",
  }


def evaluate_validation_case(
  case: Case, operator, operator_support: np.ndarray, cfg: dict, run_id: str,
  methods: tuple[str, ...], grid: SweepGrid,
) -> list[dict]:
  """Evaluate every declared method/grid point for one immutable validation case."""
  measured = np.load(case.sinogram_path, allow_pickle=False)
  target = np.load(case.target_path, allow_pickle=False)
  if case.mask_path is not None:
    raise ValueError("external evaluation masks are prohibited in the clean protocol")
  if case.sam_prior_path is None:
    raise FileNotFoundError(f"SAM prior missing for validation case: {case}")
  prior = np.load(case.sam_prior_path, allow_pickle=False)
  physical_prior = np.clip(
    prior, cfg["normalization"]["expected_target_range"][0],
    cfg["normalization"]["expected_target_range"][1],
  )
  gain = np.full_like(measured, case.projection_scale, dtype=np.float64)
  prompt_iterations = int(cfg["reconstruction"]["prompt_mlem_iterations"])
  mlem_image = mlem(
    measured, operator, iterations=prompt_iterations, multiplicative=gain,
  )[prompt_iterations]
  records = []
  for method in methods:
    if method == "MR-KEM" and case.dataset != "brainweb":
      continue
    if method == "CT-KEM" and case.dataset != "hecktor":
      continue
    if method in {"MLEM", "OSEM"}:
      candidates = grid.mlem_iterations if method == "MLEM" else grid.osem_cycles
      images = (
        mlem(
          measured, operator, iterations=max(candidates),
          save_iterations=candidates, multiplicative=gain,
        )
        if method == "MLEM" else
        osem(
          measured, operator, iterations=max(candidates),
          subsets=int(cfg["reconstruction"]["osem_subsets"]),
          save_iterations=candidates, multiplicative=gain,
        )
      )
      for iterations in candidates:
        image = images[iterations]
        records.append({
          **_base_record(case, run_id, method),
          "sigma": 0.0, "radius": 0, "neighbours": 0,
          "iterations": iterations, "alpha": 0.0,
          "osem_subsets": (
            int(cfg["reconstruction"]["osem_subsets"]) if method == "OSEM" else 1
          ),
          "effective_em_updates": (
            iterations * int(cfg["reconstruction"]["osem_subsets"])
            if method == "OSEM" else iterations
          ),
          "nrmse": nrmse(target, image, operator_support),
          "ssim": ssim(target, image, data_range=1.0, mask=operator_support),
        })
      continue
    if method == "SelfKEM":
      feature = normalize_kernel_feature(mlem_image)
    elif method in {"MR-KEM", "CT-KEM", "Bowsher-KEM"}:
      if case.anatomy_path is None:
        raise FileNotFoundError(f"missing anatomy baseline input: {case}")
      feature = normalize_kernel_feature(np.load(case.anatomy_path, allow_pickle=False))
    elif method == "SAM-KEM-MedSAM2-Qwen3-full":
      feature = normalize_kernel_feature(prior)
    else:
      raise ValueError(f"unsupported sweep method: {method}")
    method_alphas = grid.alphas if method.startswith("SAM") else (0.0,)
    kernel_settings = (
      [(0.0, radius, neighbours) for radius, neighbours in product(
        grid.bowsher_radii, grid.bowsher_neighbours,
      )]
      if method == "Bowsher-KEM" else
      [(sigma, radius, 0) for sigma, radius in product(grid.sigmas, grid.radii)]
    )
    for sigma, radius, neighbours in kernel_settings:
      kernel = (
        bowsher_kernel(
          feature, neighbours=neighbours, radius=radius, support=operator_support,
        )
        if method == "Bowsher-KEM" else
        rbf_kernel(feature, sigma=sigma, radius=radius, support=operator_support)
      )
      iterates = kem(
        measured, operator, kernel, iterations=max(grid.iterations),
        save_iterations=grid.iterations, multiplicative=gain,
      )
      for iterations, alpha in product(grid.iterations, method_alphas):
        image = (
          alpha * physical_prior + (1.0 - alpha) * iterates[iterations]
          if method.startswith("SAM") else iterates[iterations]
        )
        tolerance = float(cfg["evaluation"]["nonnegative_tolerance"])
        if not np.all(np.isfinite(image)) or float(np.nanmin(image)) < -tolerance:
          raise ValueError(
            f"invalid negative/non-finite PET estimate during validation: "
            f"{method}, alpha={alpha}, min={float(np.nanmin(image)):.6g}"
          )
        image = np.maximum(image, 0.0)
        records.append({
          **_base_record(case, run_id, method),
          "sigma": sigma, "radius": radius, "iterations": iterations,
          "neighbours": neighbours, "osem_subsets": 1,
          "effective_em_updates": iterations, "alpha": alpha,
          "nrmse": nrmse(target, image, operator_support),
          "ssim": ssim(target, image, data_range=1.0, mask=operator_support),
        })
  return records


def evaluate_incremental_validation_case(
  case: Case, operator, operator_support: np.ndarray, cfg: dict, run_id: str,
  candidates: list[dict],
) -> list[dict]:
  """Evaluate only predeclared KEM candidates from a boundary-expansion plan."""
  measured = np.load(case.sinogram_path, allow_pickle=False)
  target = np.load(case.target_path, allow_pickle=False)
  if case.mask_path is not None:
    raise ValueError("external evaluation masks are prohibited in the clean protocol")
  if case.sam_prior_path is None:
    raise FileNotFoundError(f"SAM prior missing for validation case: {case}")
  prior = np.load(case.sam_prior_path, allow_pickle=False)
  physical_prior = np.clip(
    prior, cfg["normalization"]["expected_target_range"][0],
    cfg["normalization"]["expected_target_range"][1],
  )
  gain = np.full_like(measured, case.projection_scale, dtype=np.float64)
  prompt_iterations = int(cfg["reconstruction"]["prompt_mlem_iterations"])
  mlem_image = mlem(
    measured, operator, iterations=prompt_iterations, multiplicative=gain,
  )[prompt_iterations]
  by_method: dict[str, list[dict]] = defaultdict(list)
  for candidate in candidates:
    by_method[str(candidate["method"])].append(candidate)
  records = []
  for method, method_candidates in by_method.items():
    if method == "SelfKEM":
      feature = normalize_kernel_feature(mlem_image)
    elif method in {"MR-KEM", "CT-KEM", "Bowsher-KEM"}:
      if case.anatomy_path is None:
        raise FileNotFoundError(f"missing anatomy baseline input: {case}")
      if method == "MR-KEM" and case.dataset != "brainweb":
        raise ValueError("MR-KEM expansion is permitted only for BrainWeb")
      if method == "CT-KEM" and case.dataset != "hecktor":
        raise ValueError("CT-KEM expansion is permitted only for HECKTOR")
      feature = normalize_kernel_feature(np.load(case.anatomy_path, allow_pickle=False))
    elif method == "SAM-KEM-MedSAM2-Qwen3-full":
      feature = normalize_kernel_feature(prior)
    else:
      raise ValueError(f"unsupported incremental sweep method: {method}")
    by_kernel: dict[tuple[float, int, int], list[dict]] = defaultdict(list)
    for candidate in method_candidates:
      by_kernel[
        (float(candidate["sigma"]), int(candidate["radius"]), int(candidate["neighbours"]))
      ].append(candidate)
    for (sigma, radius, neighbours), kernel_candidates in by_kernel.items():
      kernel = (
        bowsher_kernel(
          feature, neighbours=neighbours, radius=radius, support=operator_support,
        )
        if method == "Bowsher-KEM" else
        rbf_kernel(feature, sigma=sigma, radius=radius, support=operator_support)
      )
      save_iterations = tuple(sorted({int(row["iterations"]) for row in kernel_candidates}))
      iterates = kem(
        measured, operator, kernel, iterations=max(save_iterations),
        save_iterations=save_iterations, multiplicative=gain,
      )
      for candidate in kernel_candidates:
        iterations = int(candidate["iterations"])
        alpha = float(candidate["alpha"])
        image = (
          alpha * physical_prior + (1.0 - alpha) * iterates[iterations]
          if method.startswith("SAM") else iterates[iterations]
        )
        tolerance = float(cfg["evaluation"]["nonnegative_tolerance"])
        if not np.all(np.isfinite(image)) or float(np.nanmin(image)) < -tolerance:
          raise ValueError(
            f"invalid negative/non-finite PET estimate during expansion: "
            f"{method}, alpha={alpha}, min={float(np.nanmin(image)):.6g}"
          )
        image = np.maximum(image, 0.0)
        records.append({
          **_base_record(case, run_id, method),
          "sigma": sigma, "radius": radius, "neighbours": neighbours,
          "iterations": iterations, "osem_subsets": 1,
          "effective_em_updates": iterations, "alpha": alpha,
          "nrmse": nrmse(target, image, operator_support),
          "ssim": ssim(target, image, data_range=1.0, mask=operator_support),
        })
  return records
