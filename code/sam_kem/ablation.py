"""True one-factor-at-a-time ablation runner."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from .config import load_contract, require_locked_execution
from .experiments import compute_mlem_input, frozen_for_method, load_case_manifest, load_frozen_hyperparameters, method_image, validate_case_contract
from .metrics import nrmse, psnr
from .operator_factory import build_discrete_radon
from .protocols import load_protocol
from .runmeta import write_run_metadata
from .artifacts import atomic_csv_writer, finalize_csv, write_success_marker


def run_ablation(root: Path, config_path: Path, protocol_path: Path) -> Path:
  cfg, contract = load_contract(config_path)
  require_locked_execution(cfg)
  protocol = load_protocol(protocol_path)
  if protocol.kind != "ablation":
    raise ValueError("ablation runner requires an ablation protocol")
  manifest_path = root / str(cfg["outputs"]["raw_array_manifest"])
  cases = load_case_manifest(root, manifest_path, split=protocol.split)
  allowed_counts = {int(value) for value in protocol.raw.get("count_levels", contract.count_levels)}
  cases = [case for case in cases if case.count_level in allowed_counts]
  validate_case_contract(cases, contract)
  rows = list(csv.DictReader(manifest_path.open("r", encoding="utf-8", newline="")))
  variant_paths = {
    (r["dataset"], r["subject_id"], r["slice_id"], int(r["realization_id"]), int(r["count_level"])): r
    for r in rows if r["split"] == protocol.split
  }
  for column in ("sam_prior_centroid_path", "sam_prior_released_path"):
    if rows and column not in rows[0]:
      raise ValueError(f"real ablation requires manifest column {column}")
  frozen = load_frozen_hyperparameters(root / str(cfg["outputs"]["frozen_hyperparameters"]))
  operator = build_discrete_radon(
    repo_root=root, size=int(cfg["forward_model"]["image_size"]), angles=contract.angles,
    operator_id=contract.operator_id,
    expected_implementation_sha256=cfg["forward_model"]["implementation_sha256"],
  )
  output_dir = root / protocol.output; output_dir.mkdir(parents=True, exist_ok=True)
  run_metadata = write_run_metadata(output_dir, root=root, config_path=config_path, protocol_path=protocol_path)
  output = output_dir / "per_case_ablation.csv"
  fields = ["dataset", "subject_id", "slice_id", "realization_id", "count_level", "variant", "split", "psnr_db", "nrmse"]
  with atomic_csv_writer(output, fields) as writer:
    for case in cases:
      key = (case.dataset, case.subject_id, case.slice_id, case.realization_id, case.count_level)
      row = variant_paths[key]
      prior_paths = {
        "full_qwen_trained_decoder_post_fusion": case.sam_prior_path,
        "centroid_box_only": root / row["sam_prior_centroid_path"],
        "released_decoder_only": root / row["sam_prior_released_path"],
        "no_post_fusion": case.sam_prior_path,
        "prior_only": case.sam_prior_path,
      }
      method_names = {
        "full_qwen_trained_decoder_post_fusion": "SAM-KEM-MedSAM2-Qwen3-full",
        "centroid_box_only": "SAM-KEM-MedSAM2-Qwen3-full",
        "released_decoder_only": "SAM-KEM-MedSAM2-Qwen3-full",
        "no_post_fusion": "SAM-KEM-MedSAM2-Qwen3-no-post",
        "prior_only": "SAM-prior-MedSAM2-Qwen3",
      }
      measured = np.load(case.sinogram_path); target = np.load(case.target_path)
      mask = np.load(case.mask_path) if case.mask_path else None
      mlem_image = compute_mlem_input(case, measured, operator, cfg)
      hp = frozen_for_method(
        frozen, case.dataset, case.count_level,
        "SAM-KEM-MedSAM2-Qwen3-full", cfg,
      )
      for variant in protocol.methods:
        prior = prior_paths[variant]
        if prior is None or not prior.is_file():
          raise FileNotFoundError(f"missing actual prior for ablation {variant}: {prior}")
        image = method_image(
          method_names[variant], case, measured, operator, cfg, hp,
          prior_path=prior, mlem_image=mlem_image,
        )
        tolerance = float(cfg["evaluation"]["nonnegative_tolerance"])
        if not np.all(np.isfinite(image)) or float(np.nanmin(image)) < -tolerance:
          raise ValueError(
            f"ablation {variant} produced an invalid PET estimate: "
            f"min={float(np.nanmin(image)):.6g}"
          )
        image = np.maximum(image, 0.0)
        writer.writerow({
          "dataset": case.dataset, "subject_id": case.subject_id, "slice_id": case.slice_id,
          "realization_id": case.realization_id, "count_level": case.count_level,
          "variant": variant, "split": case.split,
          "psnr_db": psnr(target, image, data_range=1.0, mask=mask), "nrmse": nrmse(target, image, mask),
        })
  result_manifest = finalize_csv(
    output, schema_name="component_ablation_metrics_v1",
    primary_key=[
      "dataset", "subject_id", "slice_id", "realization_id", "count_level", "variant",
    ], split=protocol.split,
  )
  write_success_marker(output_dir, [
    run_metadata, output_dir / "environment_freeze.txt", output, result_manifest,
  ])
  return output
