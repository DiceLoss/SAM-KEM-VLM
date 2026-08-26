"""Generate checkpoint-bound continuous MedSAM2 priors from a manifest."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from .boxes import BoxCache, BoxKey, centroid_fallback
from .config import load_contract, require_development_execution, require_test_release
from .medsam import CheckpointSpec, PredictorRegistry, build_released_predictor, predict_continuous_prior, predictor_factory, sha256_file
from .noise import array_sha256
from .artifacts import atomic_save_numpy, atomic_write_json, verify_csv_artifact


def generate_priors(
  root: Path, config_path: Path, manifest_path: Path, *,
  splits: set[str], include_ablation_variants: bool = True,
) -> int:
  cfg, _ = load_contract(config_path)
  if "test_locked" in splits:
    if splits != {"test_locked"}:
      raise ValueError("generate locked-test priors in a separate post-freeze invocation")
    require_test_release(root, cfg)
    cache_path = root / str(cfg["prompts"]["box_cache"])
  else:
    require_development_execution(cfg)
    cache_path = root / str(cfg["prompts"]["training_box_cache"])
  cache = BoxCache.read(cache_path)
  manifest_record = verify_csv_artifact(manifest_path)
  if manifest_record.get("schema_name") != "canonical_case_manifest_v3":
    raise ValueError("prior generation requires a canonical v3 case manifest")
  pretrained = root / str(cfg["medsam2"]["pretrained_checkpoint_path"])
  medsam_source = root / str(cfg["medsam2"]["source_checkout_path"])
  if sha256_file(pretrained) != cfg["medsam2"]["pretrained_checkpoint_sha256"]:
    raise ValueError("pretrained checkpoint SHA mismatch")
  registry = PredictorRegistry(predictor_factory(
    medsam_source=medsam_source, pretrained_checkpoint=pretrained,
  ))
  released_predictor = (
    build_released_predictor(
      medsam_source=medsam_source, pretrained_checkpoint=pretrained,
    )
    if include_ablation_variants else None
  )
  specs = {
    dataset: CheckpointSpec(
      dataset, root / cfg["medsam2"]["checkpoint_paths"][dataset],
      cfg["medsam2"]["checkpoint_sha256"][dataset],
      cfg["normalization"]["normalization_id"],
      cfg["run_id"],
    ) for dataset in ("brainweb", "hecktor")
  }
  rows = list(csv.DictReader(manifest_path.open("r", encoding="utf-8", newline="")))
  if not rows or any(
    row.get("run_id") != cfg["run_id"]
    or row.get("split_id") != cfg["data"]["split_id"]
    or row.get("split_sha256") != cfg["data"]["split_sha256"]
    for row in rows
  ):
    raise ValueError("prior-generation manifest has the wrong clean run/split identity")
  anchor_rows = {
    BoxKey(row["dataset"], row["subject_id"], row["slice_id"], int(row["count_level"])): row
    for row in rows if int(row["realization_id"]) == int(cfg["prompts"]["anchor_realization"])
  }
  fallback_size = tuple(int(value) for value in cfg["prompts"]["fallback_box_size"])
  centroid_boxes = {}
  written = 0
  for row in rows:
    if row["split"] not in splits:
      continue
    key = BoxKey(row["dataset"], row["subject_id"], row["slice_id"], int(row["count_level"]))
    box_record = cache.get(key)
    if box_record is None:
      raise ValueError(f"box cache missing {key}")
    if box_record.run_id != cfg["run_id"]:
      raise ValueError(f"box cache run_id mismatch for {key}")
    mlem_path = root / row["mlem10_input_path"]
    mlem_image = np.load(mlem_path)
    trained_spec = specs[row["dataset"]]
    outputs = {
      "sam_prior.npy": (
        registry.get(trained_spec), box_record.box, box_record.source,
        trained_spec.sha256,
      )
    }
    if include_ablation_variants and row["split"] == "test_locked" and int(row["count_level"]) == 200000:
      if key not in centroid_boxes:
        anchor = anchor_rows.get(key)
        if anchor is None:
          raise ValueError(f"missing r=0 centroid anchor for {key}")
        anchor_image = np.load(root / anchor["mlem10_input_path"])
        centroid_boxes[key] = centroid_fallback(
          anchor_image, box_width=fallback_size[0], box_height=fallback_size[1],
          relative_threshold=float(cfg["prompts"]["fallback_relative_threshold"]),
        )
      outputs["sam_prior_centroid.npy"] = (
        registry.get(trained_spec), centroid_boxes[key], "centroid_ablation_r0_anchor",
        trained_spec.sha256,
      )
      outputs["sam_prior_released.npy"] = (
        released_predictor, box_record.box, box_record.source,
        cfg["medsam2"]["pretrained_checkpoint_sha256"],
      )
    for filename, (predictor, box, box_source, checkpoint_sha) in outputs.items():
      output = mlem_path.parent / filename
      metadata_path = output.with_suffix(".json")
      if output.exists() or metadata_path.exists():
        raise FileExistsError(f"refusing to overwrite prior: {output}")
      prior = predict_continuous_prior(
        predictor, mlem_image, box,
        output_shape=(128, 128),
      )
      atomic_save_numpy(output, prior)
      metadata = {
        "schema_version": 2, "artifact_kind": "medsam2_continuous_prior",
        "run_id": cfg["run_id"],
        "normalization_id": cfg["normalization"]["normalization_id"],
        "dataset": row["dataset"], "subject_id": row["subject_id"],
        "slice_id": row["slice_id"], "count_level": int(row["count_level"]),
        "realization_id": int(row["realization_id"]),
        "input_mlem10_sha256": array_sha256(mlem_image),
        "box_cache_key": {
          "dataset": key.dataset, "subject_id": key.subject_id,
          "slice_id": key.slice_id, "count_level": key.count_level,
        },
        "box": box.__dict__, "box_source": box_source,
        "box_anchor_realization": int(cfg["prompts"]["anchor_realization"]),
        "decoder_checkpoint_sha256": checkpoint_sha,
        "output_semantics": "continuous_raw_decoder_output_fixed_target_scale",
        "prior_sha256": array_sha256(prior),
        "prior_min": float(prior.min()), "prior_max": float(prior.max()),
        "negative_pixel_fraction": float(np.mean(prior < 0)),
      }
      try:
        atomic_write_json(metadata_path, metadata)
      except BaseException:
        output.unlink(missing_ok=True)
        raise
      written += 1
  return written
