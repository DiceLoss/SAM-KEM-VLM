"""Read-only GPU smoke test for registered validation-prior generation."""

from __future__ import annotations

import csv
import gc
from pathlib import Path

import numpy as np

from .artifacts import file_sha256, verify_csv_artifact
from .boxes import BoxCache, BoxKey
from .checkpoint_registration import validate_registered_decoder_configuration
from .config import load_contract, require_development_execution
from .medsam import CheckpointSpec, predict_continuous_prior, predictor_factory, sha256_file
from .noise import array_sha256


def select_validation_smoke_rows(
  rows: list[dict[str, str]], *, datasets: tuple[str, ...], count_level: int,
  realization_id: int,
) -> list[dict[str, str]]:
  """Select one case per dataset by a fixed lexicographic rule."""
  selected: list[dict[str, str]] = []
  for dataset in datasets:
    candidates = [
      row for row in rows
      if row.get("dataset") == dataset
      and row.get("split") == "validation"
      and int(row.get("count_level", -1)) == count_level
      and int(row.get("realization_id", -1)) == realization_id
    ]
    if not candidates:
      raise ValueError(
        f"no validation smoke candidate for {dataset}, count={count_level}, "
        f"realization={realization_id}"
      )
    selected.append(min(
      candidates,
      key=lambda row: (row["subject_id"], row["slice_id"]),
    ))
  return selected


def run_validation_prior_preflight(
  root: Path, config_path: Path, manifest_path: Path, *, count_level: int = 200000,
) -> dict:
  """Run one decoder forward per dataset without writing formal priors."""
  root = root.resolve()
  config_path = config_path.resolve()
  manifest_path = manifest_path.resolve()
  cfg, contract = load_contract(config_path)
  require_development_execution(cfg)
  validate_registered_decoder_configuration(root, config_path, cfg)
  if count_level not in contract.count_levels:
    raise ValueError(f"smoke count level is outside the frozen acquisition levels: {count_level}")

  manifest_record = verify_csv_artifact(manifest_path)
  if manifest_record.get("schema_name") != "canonical_case_manifest_v3":
    raise ValueError("validation-prior preflight requires a canonical v3 case manifest")
  with manifest_path.open("r", encoding="utf-8", newline="") as handle:
    rows = list(csv.DictReader(handle))
  required = {
    "run_id", "split_id", "split_sha256", "dataset", "subject_id", "slice_id",
    "count_level", "realization_id", "split", "mlem10_input_path",
  }
  if not rows or not required.issubset(rows[0]):
    raise ValueError(f"case manifest missing columns: {sorted(required - set(rows[0] if rows else {}))}")
  selected = select_validation_smoke_rows(
    rows, datasets=("brainweb", "hecktor"), count_level=count_level,
    realization_id=int(cfg["prompts"]["anchor_realization"]),
  )
  if any(
    row["run_id"] != contract.run_id
    or row["split_id"] != contract.split_id
    or row["split_sha256"] != contract.split_sha256
    for row in selected
  ):
    raise ValueError("selected smoke cases do not match the registered run/split identity")

  import torch

  if not torch.cuda.is_available():
    raise RuntimeError("validation-prior preflight requires CUDA")
  cache = BoxCache.read(root / str(cfg["prompts"]["training_box_cache"]))
  pretrained = root / str(cfg["medsam2"]["pretrained_checkpoint_path"])
  medsam_source = root / str(cfg["medsam2"]["source_checkout_path"])
  if sha256_file(pretrained) != cfg["medsam2"]["pretrained_checkpoint_sha256"]:
    raise ValueError("pretrained checkpoint SHA mismatch")
  build_predictor = predictor_factory(
    medsam_source=medsam_source, pretrained_checkpoint=pretrained,
  )

  sample_results = []
  for row in selected:
    dataset = row["dataset"]
    key = BoxKey(dataset, row["subject_id"], row["slice_id"], count_level)
    box_record = cache.get(key)
    if box_record is None or box_record.run_id != contract.run_id:
      raise ValueError(f"missing or mismatched box-cache record for {key}")
    mlem_path = root / row["mlem10_input_path"]
    mlem_image = np.load(mlem_path).astype(np.float32)
    if (
      mlem_image.shape != (128, 128)
      or not np.all(np.isfinite(mlem_image))
      or float(mlem_image.min()) < 0.0
      or float(mlem_image.max()) <= float(mlem_image.min())
    ):
      raise ValueError(f"invalid MLEM-10 smoke input: {mlem_path}")
    mlem_sha = array_sha256(mlem_image)
    if mlem_sha != box_record.source_mlem10_sha256:
      raise ValueError(f"box-cache/MLEM-10 identity mismatch for {key}")

    checkpoint = CheckpointSpec(
      dataset=dataset,
      path=root / str(cfg["medsam2"]["checkpoint_paths"][dataset]),
      sha256=str(cfg["medsam2"]["checkpoint_sha256"][dataset]),
      normalization_id=contract.normalization_id,
      run_id=contract.run_id,
    )
    predictor = build_predictor(checkpoint)
    with torch.inference_mode():
      prior = predict_continuous_prior(
        predictor, mlem_image, box_record.box, output_shape=(128, 128),
      )
    if (
      prior.shape != (128, 128)
      or not np.all(np.isfinite(prior))
      or float(prior.max()) <= float(prior.min())
    ):
      raise RuntimeError(f"invalid continuous prior in smoke case: {key}")
    sample_results.append({
      "dataset": dataset,
      "subject_id": row["subject_id"],
      "slice_id": row["slice_id"],
      "count_level": count_level,
      "realization_id": int(row["realization_id"]),
      "mlem10_path": row["mlem10_input_path"],
      "mlem10_array_sha256": mlem_sha,
      "box_xyxy": [
        box_record.box.xmin, box_record.box.ymin,
        box_record.box.xmax, box_record.box.ymax,
      ],
      "box_source": box_record.source,
      "decoder_checkpoint_sha256": checkpoint.sha256,
      "prior_shape": list(prior.shape),
      "prior_array_sha256": array_sha256(prior),
      "prior_min": float(prior.min()),
      "prior_max": float(prior.max()),
      "prior_mean": float(prior.mean()),
      "negative_pixel_fraction": float(np.mean(prior < 0.0)),
    })
    del predictor
    gc.collect()
    torch.cuda.empty_cache()

  return {
    "status": "PASS",
    "artifact_kind": "validation_prior_gpu_preflight",
    "schema_version": 1,
    "write_policy": "read_only_no_formal_prior_artifacts",
    "selection_rule": (
      "lexicographically_first_subject_slice_per_dataset_at_"
      "validation_anchor_realization_and_declared_count"
    ),
    "registered_config_sha256": file_sha256(config_path),
    "manifest_sha256": file_sha256(manifest_path),
    "count_level": count_level,
    "anchor_realization": int(cfg["prompts"]["anchor_realization"]),
    "cuda_device": torch.cuda.get_device_name(0),
    "torch_version": torch.__version__,
    "cuda_runtime": torch.version.cuda,
    "samples": sample_results,
  }
