"""Descriptive convergence curves using already frozen hyperparameters."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .config import load_contract, require_locked_execution
from .experiments import frozen_for_method, load_case_manifest, load_frozen_hyperparameters, validate_case_contract
from .metrics import nrmse
from .operator_factory import build_discrete_radon
from .protocols import load_protocol
from .reconstruction import bowsher_kernel, kem, mlem, normalize_kernel_feature, operator_support_mask, osem, rbf_kernel
from .runmeta import write_run_metadata
from .artifacts import atomic_csv_writer, finalize_csv, write_success_marker


def run_convergence(root: Path, config_path: Path, protocol_path: Path) -> Path:
  cfg, contract = load_contract(config_path)
  require_locked_execution(cfg)
  protocol = load_protocol(protocol_path)
  if protocol.kind != "convergence" or protocol.raw.get("select_iteration_from_this_output") is not False:
    raise ValueError("convergence output must be explicitly descriptive, never selection data")
  iterations = sorted({int(x) for x in protocol.raw["saved_iterations"]})
  cases = load_case_manifest(root, root / str(cfg["outputs"]["raw_array_manifest"]), split=protocol.split)
  allowed_counts = {int(value) for value in protocol.raw.get("count_levels", contract.count_levels)}
  cases = [case for case in cases if case.count_level in allowed_counts]
  validate_case_contract(cases, contract)
  frozen = load_frozen_hyperparameters(
    root / str(cfg["outputs"]["frozen_hyperparameters"]),
    expected_run_id=contract.run_id,
  )
  operator = build_discrete_radon(
    repo_root=root, size=int(cfg["forward_model"]["image_size"]), angles=contract.angles,
    operator_id=contract.operator_id,
    expected_implementation_sha256=cfg["forward_model"]["implementation_sha256"],
  )
  operator_support = operator_support_mask(operator)
  output_dir = root / protocol.output; output_dir.mkdir(parents=True, exist_ok=True)
  run_metadata = write_run_metadata(output_dir, root=root, config_path=config_path, protocol_path=protocol_path)
  output = output_dir / "convergence.csv"
  fields = [
    "run_id", "dataset", "subject_id", "slice_id", "realization_id", "count_level",
    "method", "iteration", "subset_count", "effective_em_updates", "split",
    "nrmse", "selection_allowed",
  ]
  with atomic_csv_writer(output, fields) as writer:
    for case in cases:
      y = np.load(case.sinogram_path); target = np.load(case.target_path)
      if case.mask_path is not None:
        raise ValueError("external evaluation masks are prohibited in the clean protocol")
      mask = operator_support
      gain = np.full_like(y, case.projection_scale, dtype=np.float64)
      for method in protocol.methods:
        if method == "MR-KEM" and case.dataset != "brainweb" or method == "CT-KEM" and case.dataset != "hecktor":
          continue
        hp = frozen_for_method(frozen, case.dataset, case.count_level, method, cfg)
        if method == "MLEM":
          images = mlem(y, operator, iterations=max(iterations), save_iterations=iterations, multiplicative=gain)
        elif method == "OSEM":
          images = osem(
            y, operator, iterations=max(iterations),
            subsets=int(cfg["reconstruction"]["osem_subsets"]),
            save_iterations=iterations, multiplicative=gain,
          )
        else:
          if method == "SelfKEM":
            init_count = int(cfg["reconstruction"]["prompt_mlem_iterations"])
            init = mlem(y, operator, iterations=init_count, multiplicative=gain)[init_count]
            feature = normalize_kernel_feature(init)
          elif method in {"MR-KEM", "CT-KEM", "Bowsher-KEM"}:
            if case.anatomy_path is None: continue
            feature = normalize_kernel_feature(np.load(case.anatomy_path))
          elif method.startswith("SAM"):
            if case.sam_prior_path is None: raise FileNotFoundError(f"missing SAM prior: {case}")
            feature = normalize_kernel_feature(np.load(case.sam_prior_path))
          else:
            raise ValueError(f"unsupported convergence method: {method}")
          kernel = (
            bowsher_kernel(
              feature, neighbours=int(hp["neighbours"]), radius=int(hp["radius"]),
              support=operator_support,
            )
            if method == "Bowsher-KEM" else
            rbf_kernel(
              feature, sigma=float(hp["sigma"]), radius=int(hp["radius"]),
              support=operator_support,
            )
          )
          images = kem(y, operator, kernel, iterations=max(iterations), save_iterations=iterations, multiplicative=gain)
        for iteration, image in images.items():
          if method.endswith("-full"):
            raw_prior = np.load(case.sam_prior_path)
            physical_prior = np.clip(
              raw_prior, cfg["normalization"]["expected_target_range"][0],
              cfg["normalization"]["expected_target_range"][1],
            )
            alpha = float(hp["alpha"]); image = alpha * physical_prior + (1 - alpha) * image
          tolerance = float(cfg["evaluation"]["nonnegative_tolerance"])
          if not np.all(np.isfinite(image)) or float(np.nanmin(image)) < -tolerance:
            raise ValueError(
              f"{method} convergence estimate is invalid at iteration {iteration}: "
              f"min={float(np.nanmin(image)):.6g}"
            )
          image = np.maximum(image, 0.0)
          writer.writerow({
            "run_id": contract.run_id, "dataset": case.dataset,
            "subject_id": case.subject_id, "slice_id": case.slice_id,
            "realization_id": case.realization_id, "count_level": case.count_level,
            "method": method, "iteration": iteration, "split": case.split,
            "subset_count": int(cfg["reconstruction"]["osem_subsets"]) if method == "OSEM" else 1,
            "effective_em_updates": (
              iteration * int(cfg["reconstruction"]["osem_subsets"])
              if method == "OSEM" else iteration
            ),
            "nrmse": nrmse(target, image, mask), "selection_allowed": "false",
          })
  result_manifest = finalize_csv(
    output, schema_name="descriptive_convergence_metrics_v2",
    primary_key=[
      "run_id", "dataset", "subject_id", "slice_id", "realization_id", "count_level",
      "method", "iteration",
    ], split=protocol.split,
  )
  write_success_marker(output_dir, [
    run_metadata, output_dir / "environment_freeze.txt", output, result_manifest,
  ])
  return output
