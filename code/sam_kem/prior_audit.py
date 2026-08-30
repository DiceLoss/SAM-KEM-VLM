"""Independent integrity and distribution audit for generated validation priors."""

from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from .artifacts import (
  atomic_csv_writer, atomic_write_json, file_sha256, finalize_csv,
  verify_csv_artifact, verify_success_marker,
)
from .boxes import BoxCache, BoxKey
from .checkpoint_registration import validate_registered_decoder_configuration
from .config import load_contract, require_development_execution
from .noise import array_sha256


DIAGNOSTIC_FIELDS = [
  "run_id", "dataset", "subject_id", "slice_id", "realization_id",
  "count_level", "split", "box_source", "decoder_checkpoint_sha256",
  "mlem10_array_sha256", "prior_path", "prior_file_sha256",
  "prior_array_sha256", "prior_min", "prior_max", "prior_mean",
  "negative_pixel_fraction", "above_one_pixel_fraction",
]


def summarize_values(values: list[float]) -> dict:
  if not values:
    return {"n": 0}
  array = np.asarray(values, dtype=np.float64)
  return {
    "n": len(values),
    "mean": float(array.mean()),
    "quantiles": {
      str(percent): float(np.percentile(array, percent))
      for percent in (0, 1, 5, 25, 50, 75, 95, 99, 100)
    },
  }


def _read_csv(path: Path, *, expected_schema: str) -> list[dict[str, str]]:
  record = verify_csv_artifact(path)
  if record.get("schema_name") != expected_schema:
    raise ValueError(f"unexpected CSV schema for {path}: {record.get('schema_name')}")
  with path.open("r", encoding="utf-8", newline="") as handle:
    return list(csv.DictReader(handle))


def audit_validation_priors(
  root: Path, config_path: Path, case_manifest_path: Path, prior_dir: Path,
  diagnostics_path: Path, summary_path: Path,
) -> dict:
  root = root.resolve()
  config_path = config_path.resolve()
  case_manifest_path = case_manifest_path.resolve()
  prior_dir = prior_dir.resolve()
  diagnostics_path = diagnostics_path.resolve()
  summary_path = summary_path.resolve()
  if diagnostics_path.exists() or summary_path.exists():
    raise FileExistsError("refusing to overwrite validation-prior audit outputs")
  cfg, contract = load_contract(config_path)
  require_development_execution(cfg)
  validate_registered_decoder_configuration(root, config_path, cfg)
  success = verify_success_marker(prior_dir)
  prior_manifest_path = prior_dir / "prior_manifest.csv"
  generation_record_path = prior_dir / "generation_record.json"
  prior_rows = _read_csv(
    prior_manifest_path, expected_schema="medsam2_prior_manifest_v1",
  )
  case_rows = _read_csv(
    case_manifest_path, expected_schema="canonical_case_manifest_v3",
  )
  generation = json.loads(generation_record_path.read_text(encoding="utf-8"))
  config_sha = file_sha256(config_path)
  case_manifest_sha = file_sha256(case_manifest_path)
  cache_path = root / str(cfg["prompts"]["training_box_cache"])
  cache_sha = file_sha256(cache_path)
  if (
    success.get("status") != "complete"
    or generation.get("status") != "complete"
    or generation.get("splits") != ["validation"]
    or generation.get("registered_config_sha256") != config_sha
    or generation.get("case_manifest_sha256") != case_manifest_sha
    or generation.get("box_cache_sha256") != cache_sha
    or int(generation.get("prior_artifacts", -1)) != len(prior_rows)
    or generation.get("prior_manifest_sha256") != file_sha256(prior_manifest_path)
  ):
    raise ValueError("prior completion records have inconsistent frozen identities")

  expected_cases = {
    (
      row["dataset"], row["subject_id"], row["slice_id"],
      int(row["realization_id"]), int(row["count_level"]),
    ): row
    for row in case_rows if row["split"] == "validation"
  }
  actual_cases = {
    (
      row["dataset"], row["subject_id"], row["slice_id"],
      int(row["realization_id"]), int(row["count_level"]),
    ): row
    for row in prior_rows if row["prior_variant"] == "sam_prior"
  }
  if len(actual_cases) != len(prior_rows) or set(actual_cases) != set(expected_cases):
    raise ValueError("prior manifest does not exactly cover validation cases once")

  cache = BoxCache.read(cache_path)
  diagnostics = []
  fallback_keys = set()
  for case_key in sorted(actual_cases):
    dataset, subject_id, slice_id, realization_id, count_level = case_key
    prior_row = actual_cases[case_key]
    case_row = expected_cases[case_key]
    box_key = BoxKey(dataset, subject_id, slice_id, count_level)
    box_record = cache.get(box_key)
    if box_record is None:
      raise ValueError(f"box cache missing validation key: {box_key}")
    expected_checkpoint = str(cfg["medsam2"]["checkpoint_sha256"][dataset])
    mlem_path = root / case_row["mlem10_input_path"]
    expected_prior_path = mlem_path.parent / "sam_prior.npy"
    prior_path = root / prior_row["prior_path"]
    metadata_path = root / prior_row["metadata_path"]
    if prior_path.resolve() != expected_prior_path.resolve():
      raise ValueError(f"prior path differs from canonical case path: {prior_path}")
    if not prior_path.is_file() or not metadata_path.is_file():
      raise FileNotFoundError(f"prior or metadata missing: {prior_path}")
    if (
      file_sha256(prior_path) != prior_row["prior_file_sha256"]
      or file_sha256(metadata_path) != prior_row["metadata_file_sha256"]
      or prior_row["decoder_checkpoint_sha256"] != expected_checkpoint
      or prior_row["box_source"] != box_record.source
    ):
      raise ValueError(f"prior-manifest file/checkpoint/box mismatch: {prior_path}")

    mlem = np.load(mlem_path, allow_pickle=False)
    mlem_sha = array_sha256(mlem)
    prior = np.load(prior_path, allow_pickle=False).astype(np.float32)
    prior_sha = array_sha256(prior)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    expected_metadata = {
      "schema_version": 2,
      "artifact_kind": "medsam2_continuous_prior",
      "run_id": contract.run_id,
      "split": "validation",
      "normalization_id": contract.normalization_id,
      "dataset": dataset,
      "subject_id": subject_id,
      "slice_id": slice_id,
      "count_level": count_level,
      "realization_id": realization_id,
      "input_mlem10_sha256": mlem_sha,
      "box_cache_key": {
        "dataset": dataset, "subject_id": subject_id,
        "slice_id": slice_id, "count_level": count_level,
      },
      "box": box_record.box.__dict__,
      "box_source": box_record.source,
      "box_anchor_realization": int(cfg["prompts"]["anchor_realization"]),
      "decoder_checkpoint_sha256": expected_checkpoint,
      "output_semantics": "continuous_raw_decoder_output_fixed_target_scale",
      "registered_config_sha256": config_sha,
      "case_manifest_sha256": case_manifest_sha,
      "box_cache_sha256": cache_sha,
      "prior_sha256": prior_sha,
    }
    mismatches = {
      key: (metadata.get(key), value)
      for key, value in expected_metadata.items() if metadata.get(key) != value
    }
    if mismatches:
      raise ValueError(f"prior metadata mismatch for {prior_path}: {mismatches}")
    if (
      prior.shape != (128, 128)
      or not np.all(np.isfinite(prior))
      or float(prior.max()) <= float(prior.min())
      or prior_row["prior_array_sha256"] != prior_sha
      or prior_row["input_mlem10_sha256"] != mlem_sha
      or not np.isclose(float(metadata["prior_min"]), float(prior.min()))
      or not np.isclose(float(metadata["prior_max"]), float(prior.max()))
      or not np.isclose(
        float(metadata["negative_pixel_fraction"]), float(np.mean(prior < 0.0)),
      )
    ):
      raise ValueError(f"invalid prior values or recorded statistics: {prior_path}")
    if realization_id == int(cfg["prompts"]["anchor_realization"]):
      if box_record.source_mlem10_sha256 != mlem_sha:
        raise ValueError(f"anchor box and MLEM identity mismatch: {box_key}")
    if box_record.source != "qwen3-vl-plus":
      fallback_keys.add(box_key)
    diagnostics.append({
      "run_id": contract.run_id, "dataset": dataset,
      "subject_id": subject_id, "slice_id": slice_id,
      "realization_id": realization_id, "count_level": count_level,
      "split": "validation", "box_source": box_record.source,
      "decoder_checkpoint_sha256": expected_checkpoint,
      "mlem10_array_sha256": mlem_sha,
      "prior_path": prior_row["prior_path"],
      "prior_file_sha256": prior_row["prior_file_sha256"],
      "prior_array_sha256": prior_sha,
      "prior_min": float(prior.min()), "prior_max": float(prior.max()),
      "prior_mean": float(prior.mean()),
      "negative_pixel_fraction": float(np.mean(prior < 0.0)),
      "above_one_pixel_fraction": float(np.mean(prior > 1.0)),
    })

  diagnostics_path.parent.mkdir(parents=True, exist_ok=True)
  with atomic_csv_writer(diagnostics_path, DIAGNOSTIC_FIELDS) as writer:
    writer.writerows(diagnostics)
  diagnostics_sidecar = finalize_csv(
    diagnostics_path, schema_name="validation_prior_diagnostics_v1",
    primary_key=[
      "run_id", "dataset", "subject_id", "slice_id", "realization_id",
      "count_level",
    ], split="validation",
  )
  grouped = defaultdict(list)
  for row in diagnostics:
    grouped[(row["dataset"], row["count_level"])].append(row)
  group_summary = {}
  for (dataset, count_level), group in sorted(grouped.items()):
    group_summary[f"{dataset}/c{count_level}"] = {
      "records": len(group),
      "box_source_counts": dict(sorted(Counter(
        row["box_source"] for row in group
      ).items())),
      "prior_min": summarize_values([row["prior_min"] for row in group]),
      "prior_max": summarize_values([row["prior_max"] for row in group]),
      "prior_mean": summarize_values([row["prior_mean"] for row in group]),
      "negative_pixel_fraction": summarize_values([
        row["negative_pixel_fraction"] for row in group
      ]),
      "above_one_pixel_fraction": summarize_values([
        row["above_one_pixel_fraction"] for row in group
      ]),
    }
  summary = {
    "artifact_kind": "validation_prior_integrity_audit",
    "schema_version": 1,
    "status": "PASS",
    "run_id": contract.run_id,
    "split": "validation",
    "records": len(diagnostics),
    "registered_config_sha256": config_sha,
    "case_manifest_sha256": case_manifest_sha,
    "box_cache_sha256": cache_sha,
    "prior_manifest_sha256": file_sha256(prior_manifest_path),
    "prior_success_marker_sha256": file_sha256(prior_dir / "_SUCCESS.json"),
    "generation_record_sha256": file_sha256(generation_record_path),
    "diagnostics_sha256": file_sha256(diagnostics_path),
    "diagnostics_manifest_sha256": file_sha256(diagnostics_sidecar),
    "dataset_counts": dict(sorted(Counter(row["dataset"] for row in diagnostics).items())),
    "count_level_counts": {
      str(key): value for key, value in sorted(Counter(
        row["count_level"] for row in diagnostics
      ).items())
    },
    "box_source_counts": dict(sorted(Counter(
      row["box_source"] for row in diagnostics
    ).items())),
    "fallback_cache_key_count": len(fallback_keys),
    "fallback_cache_keys": [
      {
        "dataset": key.dataset, "subject_id": key.subject_id,
        "slice_id": key.slice_id, "count_level": key.count_level,
      }
      for key in sorted(fallback_keys)
    ],
    "priors_with_negative_pixels": sum(
      row["negative_pixel_fraction"] > 0 for row in diagnostics
    ),
    "priors_with_pixels_above_one": sum(
      row["above_one_pixel_fraction"] > 0 for row in diagnostics
    ),
    "raw_output_interpretation": (
      "continuous_decoder_output; values outside [0,1] are retained here, "
      "then per-prior min-max is used only for kernel features and [0,1] "
      "clipping is used only for the physical fusion branch"
    ),
    "groups": group_summary,
  }
  atomic_write_json(summary_path, summary)
  return summary
