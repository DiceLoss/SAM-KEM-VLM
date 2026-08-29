"""Generate checkpoint-bound continuous MedSAM2 priors from a manifest."""

from __future__ import annotations

import csv
import gc
import json
from pathlib import Path

import numpy as np

from .boxes import BoxCache, BoxKey, centroid_fallback
from .config import load_contract, require_development_execution, require_test_release
from .medsam import CheckpointSpec, build_released_predictor, predict_continuous_prior, predictor_factory, sha256_file
from .noise import array_sha256
from .artifacts import (
  atomic_csv_writer, atomic_save_numpy, atomic_write_json, file_sha256, finalize_csv,
  verify_csv_artifact, verify_success_marker, write_success_marker,
)
from .checkpoint_registration import validate_registered_decoder_configuration


PRIOR_MANIFEST_FIELDS = [
  "run_id", "dataset", "subject_id", "slice_id", "realization_id",
  "count_level", "split", "prior_variant", "prior_path", "prior_file_sha256",
  "prior_array_sha256", "metadata_path", "metadata_file_sha256",
  "input_mlem10_sha256", "box_source", "decoder_checkpoint_sha256",
]


def _relative(path: Path, root: Path) -> str:
  try:
    return path.resolve().relative_to(root.resolve()).as_posix()
  except ValueError:
    return path.resolve().as_posix()


def _verify_existing_prior(
  output: Path, metadata_path: Path, *, expected: dict,
) -> tuple[np.ndarray, dict]:
  if output.is_file() != metadata_path.is_file():
    raise ValueError(f"incomplete prior/metadata pair cannot be resumed: {output}")
  if not output.is_file():
    raise FileNotFoundError(output)
  metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
  mismatches = {
    key: (metadata.get(key), value)
    for key, value in expected.items() if metadata.get(key) != value
  }
  if mismatches:
    raise ValueError(f"existing prior metadata identity mismatch for {output}: {mismatches}")
  prior = np.load(output, allow_pickle=False).astype(np.float32)
  if prior.shape != (128, 128) or not np.all(np.isfinite(prior)):
    raise ValueError(f"existing prior is non-finite or has the wrong shape: {output}")
  if metadata.get("prior_sha256") != array_sha256(prior):
    raise ValueError(f"existing prior array checksum mismatch: {output}")
  return prior, metadata


def _verify_prior_manifest_files(root: Path, manifest_path: Path) -> int:
  record = verify_csv_artifact(manifest_path)
  if record.get("schema_name") != "medsam2_prior_manifest_v1":
    raise ValueError(f"invalid prior completion manifest: {manifest_path}")
  with manifest_path.open("r", encoding="utf-8", newline="") as handle:
    rows = list(csv.DictReader(handle))
  for row in rows:
    prior_path = root / row["prior_path"]
    metadata_path = root / row["metadata_path"]
    if (
      not prior_path.is_file()
      or file_sha256(prior_path) != row["prior_file_sha256"]
      or not metadata_path.is_file()
      or file_sha256(metadata_path) != row["metadata_file_sha256"]
    ):
      raise ValueError(f"prior completion manifest member mismatch: {prior_path}")
  return len(rows)


def generate_priors(
  root: Path, config_path: Path, manifest_path: Path, *,
  splits: set[str], include_ablation_variants: bool = True,
  resume_verified: bool = False, output_dir: Path | None = None,
) -> dict:
  cfg, _ = load_contract(config_path)
  validate_registered_decoder_configuration(root, config_path, cfg)
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
  output_dir = (
    output_dir.resolve() if output_dir is not None
    else (root / "runs" / cfg["run_id"] / "validation_priors").resolve()
  )
  output_dir.mkdir(parents=True, exist_ok=True)
  completion_marker = output_dir / "_SUCCESS.json"
  generation_record_path = output_dir / "generation_record.json"
  prior_manifest_path = output_dir / "prior_manifest.csv"
  if completion_marker.is_file():
    if not resume_verified:
      raise FileExistsError(f"prior batch is already complete: {completion_marker}")
    verify_success_marker(output_dir)
    result = json.loads(generation_record_path.read_text(encoding="utf-8"))
    verified_count = _verify_prior_manifest_files(root, prior_manifest_path)
    if (
      result.get("status") != "complete"
      or result.get("registered_config_sha256") != file_sha256(config_path)
      or int(result.get("prior_artifacts", -1)) != verified_count
    ):
      raise ValueError("completed prior batch identity differs from the registered config")
    result["already_complete"] = True
    result["verified_prior_artifacts"] = verified_count
    return result
  if any([
    generation_record_path.exists(), prior_manifest_path.exists(),
    prior_manifest_path.with_suffix(".csv.manifest.json").exists(),
  ]):
    raise FileExistsError(
      f"incomplete finalization artifacts exist without _SUCCESS: {output_dir}"
    )
  pretrained = root / str(cfg["medsam2"]["pretrained_checkpoint_path"])
  medsam_source = root / str(cfg["medsam2"]["source_checkout_path"])
  if sha256_file(pretrained) != cfg["medsam2"]["pretrained_checkpoint_sha256"]:
    raise ValueError("pretrained checkpoint SHA mismatch")
  build_predictor = predictor_factory(
    medsam_source=medsam_source, pretrained_checkpoint=pretrained,
  )
  released_predictor = (
    build_released_predictor(
      medsam_source=medsam_source, pretrained_checkpoint=pretrained,
    )
    if include_ablation_variants and splits == {"test_locked"} else None
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
  config_sha = file_sha256(config_path)
  case_manifest_sha = file_sha256(manifest_path)
  box_cache_sha = file_sha256(cache_path)
  written = 0
  reused = 0
  manifest_rows = []
  active_dataset = None
  trained_predictor = None
  eligible_rows = sorted(
    (row for row in rows if row["split"] in splits),
    key=lambda row: (
      row["dataset"], row["subject_id"], row["slice_id"],
      int(row["count_level"]), int(row["realization_id"]),
    ),
  )
  for row in eligible_rows:
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
        "trained", box_record.box, box_record.source,
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
        "trained", centroid_boxes[key], "centroid_ablation_r0_anchor",
        trained_spec.sha256,
      )
      outputs["sam_prior_released.npy"] = (
        "released", box_record.box, box_record.source,
        cfg["medsam2"]["pretrained_checkpoint_sha256"],
      )
    for filename, (predictor_kind, box, box_source, checkpoint_sha) in outputs.items():
      output = mlem_path.parent / filename
      metadata_path = output.with_suffix(".json")
      expected_metadata = {
        "schema_version": 2, "artifact_kind": "medsam2_continuous_prior",
        "run_id": cfg["run_id"],
        "split": row["split"],
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
        "registered_config_sha256": config_sha,
        "case_manifest_sha256": case_manifest_sha,
        "box_cache_sha256": box_cache_sha,
      }
      if output.exists() or metadata_path.exists():
        if not resume_verified:
          raise FileExistsError(f"refusing to overwrite prior: {output}")
        prior, metadata = _verify_existing_prior(
          output, metadata_path, expected=expected_metadata,
        )
        reused += 1
      else:
        if predictor_kind == "released":
          predictor = released_predictor
        else:
          if active_dataset != row["dataset"]:
            if trained_predictor is not None:
              del trained_predictor
              gc.collect()
              try:
                import torch
                torch.cuda.empty_cache()
              except (ImportError, RuntimeError):
                pass
            trained_predictor = build_predictor(trained_spec)
            active_dataset = row["dataset"]
          predictor = trained_predictor
        prior = predict_continuous_prior(
          predictor, mlem_image, box,
          output_shape=(128, 128),
        )
        atomic_save_numpy(output, prior)
        metadata = {
          **expected_metadata,
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
      manifest_rows.append({
        "run_id": cfg["run_id"], "dataset": row["dataset"],
        "subject_id": row["subject_id"], "slice_id": row["slice_id"],
        "realization_id": int(row["realization_id"]),
        "count_level": int(row["count_level"]), "split": row["split"],
        "prior_variant": filename.removesuffix(".npy"),
        "prior_path": _relative(output, root),
        "prior_file_sha256": file_sha256(output),
        "prior_array_sha256": metadata["prior_sha256"],
        "metadata_path": _relative(metadata_path, root),
        "metadata_file_sha256": file_sha256(metadata_path),
        "input_mlem10_sha256": expected_metadata["input_mlem10_sha256"],
        "box_source": box_source,
        "decoder_checkpoint_sha256": checkpoint_sha,
      })

  with atomic_csv_writer(prior_manifest_path, PRIOR_MANIFEST_FIELDS) as writer:
    writer.writerows(manifest_rows)
  prior_manifest_sidecar = finalize_csv(
    prior_manifest_path, schema_name="medsam2_prior_manifest_v1",
    primary_key=[
      "run_id", "dataset", "subject_id", "slice_id", "realization_id",
      "count_level", "prior_variant",
    ], split=next(iter(splits)) if len(splits) == 1 else None,
  )
  result = {
    "artifact_kind": "medsam2_prior_generation_batch",
    "schema_version": 1,
    "status": "complete",
    "run_id": cfg["run_id"],
    "splits": sorted(splits),
    "registered_config_sha256": config_sha,
    "case_manifest_sha256": case_manifest_sha,
    "box_cache_sha256": box_cache_sha,
    "eligible_case_rows": len(eligible_rows),
    "prior_artifacts": len(manifest_rows),
    "written": written,
    "reused_verified": reused,
    "prior_manifest_path": _relative(prior_manifest_path, root),
    "prior_manifest_sha256": file_sha256(prior_manifest_path),
  }
  atomic_write_json(generation_record_path, result)
  marker = write_success_marker(
    output_dir, [prior_manifest_path, prior_manifest_sidecar, generation_record_path],
  )
  result["success_marker_path"] = _relative(marker, root)
  return result
