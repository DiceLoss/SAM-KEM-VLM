"""Shared runner for primary, comparison, ablation and reporting protocols."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .config import load_contract, require_locked_execution
from .metrics import mae, nrmse, poisson_deviance, psnr, ssim
from .operator_factory import build_discrete_radon
from .protocols import Protocol, load_protocol
from .reconstruction import (
  bowsher_kernel, kem, mlem, normalize_kernel_feature, osem, rbf_kernel,
)
from .runmeta import write_run_metadata
from .artifacts import atomic_csv_writer, finalize_csv, write_success_marker


@dataclass(frozen=True)
class Case:
  dataset: str
  subject_id: str
  slice_id: str
  realization_id: int
  count_level: int
  split: str
  seed: int
  scale_constant_C: float
  normalization_id: str
  normalization_offset: float
  normalization_clip: bool
  operator_id: str
  projection_scale: float
  target_path: Path
  sinogram_path: Path
  mlem10_input_path: Path
  anatomy_path: Path | None
  sam_prior_path: Path | None
  mask_path: Path | None


def _optional_path(root: Path, value: str) -> Path | None:
  return root / value if value.strip() else None


def load_case_manifest(root: Path, path: Path, *, split: str) -> list[Case]:
  with path.open("r", encoding="utf-8", newline="") as handle:
    rows = list(csv.DictReader(handle))
  required = {
    "dataset", "subject_id", "slice_id", "realization_id", "count_level",
    "split", "target_path", "sinogram_path", "anatomy_path", "sam_prior_path",
    "evaluation_mask_path", "mlem10_input_path", "seed", "normalization_id",
    "normalization_offset", "normalization_clip", "scale_constant_C", "operator_id",
    "projection_scale",
  }
  if not rows or not required.issubset(rows[0]):
    raise ValueError(f"case manifest missing columns: {sorted(required - set(rows[0] if rows else {}))}")
  cases = [Case(
    dataset=row["dataset"], subject_id=row["subject_id"], slice_id=row["slice_id"],
    realization_id=int(row["realization_id"]), count_level=int(row["count_level"]),
    split=row["split"], seed=int(row["seed"]),
    normalization_id=row["normalization_id"],
    normalization_offset=float(row["normalization_offset"]),
    normalization_clip=row["normalization_clip"].lower() == "true",
    scale_constant_C=float(row["scale_constant_C"]), operator_id=row["operator_id"],
    projection_scale=float(row["projection_scale"]),
    target_path=root / row["target_path"],
    sinogram_path=root / row["sinogram_path"], mlem10_input_path=root / row["mlem10_input_path"],
    anatomy_path=_optional_path(root, row["anatomy_path"]),
    sam_prior_path=_optional_path(root, row["sam_prior_path"]),
    mask_path=_optional_path(root, row["evaluation_mask_path"]),
  ) for row in rows if row["split"] == split]
  keys = [(c.dataset, c.subject_id, c.slice_id, c.realization_id, c.count_level) for c in cases]
  if len(keys) != len(set(keys)):
    raise ValueError("case manifest contains duplicate case keys")
  return cases


def load_frozen_hyperparameters(path: Path) -> dict[tuple[str, int, str], dict[str, float | int]]:
  rows = list(csv.DictReader(path.open("r", encoding="utf-8", newline="")))
  frozen: dict[tuple[str, int, str], dict[str, float | int]] = {}
  for row in rows:
    key = (row["dataset"], int(row["count_level"]), row["method"])
    if key in frozen:
      raise ValueError(f"duplicate frozen hyperparameter key: {key}")
    frozen[key] = {
      "sigma": float(row["sigma"]), "radius": int(row["radius"]),
      "neighbours": int(row["neighbours"]),
      "iterations": int(row["iterations"]), "alpha": float(row["alpha"]),
    }
  return frozen


def frozen_for_method(frozen: dict, dataset: str, count: int, method: str, cfg: dict) -> dict[str, float | int]:
  if method == "SAM-prior-MedSAM2-Qwen3":
    return {
      "sigma": 0.0, "radius": 0, "neighbours": 0,
      "iterations": int(cfg["reconstruction"]["prompt_mlem_iterations"]), "alpha": 0.0,
    }
  if method == "SAM-KEM-MedSAM2-Qwen3-no-post":
    parameters = dict(frozen[(dataset, count, "SAM-KEM-MedSAM2-Qwen3-full")])
    parameters["alpha"] = 0.0
    return parameters
  return frozen[(dataset, count, method)]


def validate_case_contract(cases: list[Case], contract) -> None:
  for case in cases:
    if case.operator_id != contract.operator_id:
      raise ValueError(f"case operator mismatch for {case.subject_id}/{case.slice_id}")
    if case.scale_constant_C != contract.scale_constants[case.dataset]:
      raise ValueError(f"case physical scale mismatch for {case.subject_id}/{case.slice_id}")
    if (
      case.normalization_id != contract.normalization_id
      or case.normalization_offset != contract.normalization_offsets[case.dataset]
      or case.normalization_clip != contract.normalization_clip
    ):
      raise ValueError(f"case normalization mismatch for {case.subject_id}/{case.slice_id}")


def compute_mlem_input(case: Case, measured: np.ndarray, operator, cfg) -> np.ndarray:
  if not case.mlem10_input_path.is_file():
    raise FileNotFoundError(
      f"canonical MLEM-10 prompt/reconstruction missing: {case.mlem10_input_path}"
    )
  iterations = int(cfg["reconstruction"]["prompt_mlem_iterations"])
  gain = np.full_like(measured, case.projection_scale, dtype=np.float64)
  recomputed = mlem(
    measured, operator, iterations=iterations, multiplicative=gain,
  )[iterations]
  cached = np.load(case.mlem10_input_path).astype(np.float32)
  if cached.shape != recomputed.shape or not np.allclose(
    cached, recomputed, rtol=2e-5, atol=2e-7,
  ):
    maximum_error = (
      float("inf") if cached.shape != recomputed.shape
      else float(np.max(np.abs(cached - recomputed)))
    )
    raise ValueError(
      "cached MLEM input is stale or was produced by a mismatched operator/"
      f"observation model: {case.mlem10_input_path} (max error {maximum_error:.6g})"
    )
  return recomputed


def method_image(
  method: str, case: Case, measured: np.ndarray, operator, cfg, hp, *,
  prior_path: Path | None = None, mlem_image: np.ndarray | None = None,
) -> np.ndarray | None:
  gain = np.full_like(measured, case.projection_scale, dtype=np.float64)
  if mlem_image is None:
    mlem_image = compute_mlem_input(case, measured, operator, cfg)
  if method == "MLEM":
    iterations = int(hp["iterations"])
    return mlem(
      measured, operator, iterations=iterations, multiplicative=gain,
    )[iterations]
  if method == "OSEM":
    iterations = int(hp["iterations"])
    return osem(measured, operator, iterations=iterations, subsets=int(cfg["reconstruction"]["osem_subsets"]), multiplicative=gain)[iterations]
  if method in {"MR-KEM", "CT-KEM", "Bowsher-KEM"} and case.anatomy_path is None:
    return None
  active_prior_path = prior_path or case.sam_prior_path
  if method.startswith("SAM") and active_prior_path is None:
    raise FileNotFoundError(f"SAM prior missing for {case}")
  if method == "SelfKEM":
    feature = normalize_kernel_feature(mlem_image)
  elif method in {"MR-KEM", "CT-KEM"}:
    feature = normalize_kernel_feature(np.load(case.anatomy_path))
  elif method == "Bowsher-KEM":
    feature = normalize_kernel_feature(np.load(case.anatomy_path))
    kernel = bowsher_kernel(
      feature, neighbours=int(hp["neighbours"]), radius=int(hp["radius"]),
    )
    return kem(measured, operator, kernel, iterations=int(hp["iterations"]), multiplicative=gain)[int(hp["iterations"])]
  elif method.startswith("SAM"):
    prior = np.load(active_prior_path).astype(np.float32)
    physical_prior = np.clip(
      prior, cfg["normalization"]["expected_target_range"][0],
      cfg["normalization"]["expected_target_range"][1],
    )
    if method == "SAM-prior-MedSAM2-Qwen3":
      return physical_prior
    feature = normalize_kernel_feature(prior)
  else:
    raise ValueError(f"unimplemented method: {method}")
  kernel = rbf_kernel(feature, sigma=float(hp["sigma"]), radius=int(hp["radius"]))
  reconstruction = kem(measured, operator, kernel, iterations=int(hp["iterations"]), multiplicative=gain)[int(hp["iterations"])]
  if method.endswith("-full"):
    alpha = float(hp["alpha"])
    reconstruction = alpha * physical_prior + (1.0 - alpha) * reconstruction
  return reconstruction.astype(np.float32)


def run_protocol(
  root: Path, config_path: Path, protocol_path: Path, *, finalize_run: bool = True,
) -> Path:
  cfg, contract = load_contract(config_path)
  require_locked_execution(cfg)
  protocol: Protocol = load_protocol(protocol_path)
  if protocol.kind in {"training", "hyperparameter_sweep", "ablation", "convergence", "cross_count", "runtime"}:
    raise ValueError(f"{protocol.kind} uses its dedicated runner; generic evaluation runner refused")
  manifest = root / str(cfg["outputs"]["raw_array_manifest"])
  cases = load_case_manifest(root, manifest, split=protocol.split)
  allowed_datasets = set(protocol.raw.get("datasets", []))
  allowed_counts = {int(value) for value in protocol.raw.get("count_levels", contract.count_levels)}
  allowed_realizations = {int(value) for value in protocol.raw.get("realizations", contract.realizations)}
  cases = [
    case for case in cases
    if case.dataset in allowed_datasets
    and case.count_level in allowed_counts
    and case.realization_id in allowed_realizations
  ]
  if not cases:
    raise ValueError(f"no {protocol.split} cases in {manifest}")
  validate_case_contract(cases, contract)
  frozen = load_frozen_hyperparameters(root / str(cfg["outputs"]["frozen_hyperparameters"]))
  operator = build_discrete_radon(
    repo_root=root, size=int(cfg["forward_model"]["image_size"]), angles=contract.angles,
    operator_id=contract.operator_id,
    expected_implementation_sha256=cfg["forward_model"]["implementation_sha256"],
  )
  output_dir = root / protocol.output
  output_dir.mkdir(parents=True, exist_ok=True)
  run_metadata = write_run_metadata(output_dir, root=root, config_path=config_path, protocol_path=protocol_path)
  output_path = output_dir / "per_case_metrics.csv"
  fields = [
    "protocol", "dataset", "subject_id", "slice_id", "realization_id", "count_level",
    "method", "split", "psnr_db", "ssim", "nrmse", "mae", "poisson_deviance",
    "operator_id", "normalization_id", "normalization_offset", "normalization_clip",
    "scale_constant_C", "sigma", "radius", "neighbours", "iterations", "alpha",
    "osem_subsets", "effective_em_updates", "seed", "checkpoint_sha256",
  ]
  with atomic_csv_writer(output_path, fields) as writer:
    for case in cases:
      measured = np.load(case.sinogram_path)
      target = np.load(case.target_path)
      mask = np.load(case.mask_path) if case.mask_path is not None else None
      mlem_image = compute_mlem_input(case, measured, operator, cfg)
      for method in protocol.methods:
        if method == "MR-KEM" and case.dataset != "brainweb":
          continue
        if method == "CT-KEM" and case.dataset != "hecktor":
          continue
        hp = frozen_for_method(frozen, case.dataset, case.count_level, method, cfg)
        image = method_image(method, case, measured, operator, cfg, hp, mlem_image=mlem_image)
        if image is None:
          continue
        negative_tolerance = float(cfg["evaluation"]["nonnegative_tolerance"])
        if not np.all(np.isfinite(image)) or float(np.nanmin(image)) < -negative_tolerance:
          raise ValueError(
            f"{method} produced a non-finite or negative PET estimate for "
            f"{case.subject_id}/{case.slice_id}: min={float(np.nanmin(image)):.6g}"
          )
        image = np.maximum(image, 0.0)
        expected = case.projection_scale * operator.forward(image)
        writer.writerow({
          "protocol": protocol.name, "dataset": case.dataset,
          "subject_id": case.subject_id, "slice_id": case.slice_id,
          "realization_id": case.realization_id, "count_level": case.count_level,
          "method": method, "split": case.split,
          "psnr_db": psnr(target, image, data_range=float(cfg["evaluation"]["psnr_data_range_after_fixed_C_scaling"]), mask=mask),
          "ssim": ssim(target, image, data_range=float(cfg["evaluation"]["psnr_data_range_after_fixed_C_scaling"]), mask=mask),
          "nrmse": nrmse(target, image, mask), "mae": mae(target, image, mask),
          "poisson_deviance": poisson_deviance(measured, expected),
          "operator_id": contract.operator_id,
          "normalization_id": contract.normalization_id,
          "normalization_offset": contract.normalization_offsets[case.dataset],
          "normalization_clip": contract.normalization_clip,
          "scale_constant_C": contract.scale_constants[case.dataset], **hp,
          "osem_subsets": int(cfg["reconstruction"]["osem_subsets"]) if method == "OSEM" else 1,
          "effective_em_updates": (
            int(hp["iterations"]) * int(cfg["reconstruction"]["osem_subsets"])
            if method == "OSEM" else int(hp["iterations"])
          ),
          "seed": case.seed,
          "checkpoint_sha256": (
            cfg["medsam2"]["checkpoint_sha256"][case.dataset]
            if method.startswith("SAM") else "NA"
          ),
        })
  result_manifest = finalize_csv(
    output_path, schema_name="per_case_reconstruction_metrics_v1",
    primary_key=[
      "protocol", "dataset", "subject_id", "slice_id", "realization_id",
      "count_level", "method",
    ], split=protocol.split,
  )
  if finalize_run:
    write_success_marker(output_dir, [
      run_metadata, output_dir / "environment_freeze.txt", output_path, result_manifest,
    ])
  return output_path
