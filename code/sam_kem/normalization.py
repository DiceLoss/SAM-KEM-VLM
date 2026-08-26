"""Train-only normalization calibration and validation diagnostics."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .artifacts import atomic_csv_writer, finalize_csv, write_success_marker
from .config import load_yaml_strict
from .noise import array_sha256
from .protocols import load_protocol
from .runmeta import write_run_metadata


ACTIVITY_COLUMNS = (
  "dataset", "subject_id", "slice_id", "split", "activity_path",
  "activity_sha256", "activity_unit",
)


@dataclass(frozen=True)
class ActivityCase:
  dataset: str
  subject_id: str
  slice_id: str
  split: str
  path: Path
  expected_sha256: str
  unit: str


@dataclass(frozen=True)
class NormalizationCandidate:
  name: str
  offset: float | None
  scale: float | None
  clip: bool
  paper_role: str
  fitted_from: str


def build_normalization_activity_manifest(
  root: Path, arrays_root: Path, output_path: Path,
) -> Path:
  """Index verified train/validation canonical activities exactly once."""
  import json

  root = root.resolve()
  rows = []
  for activity_path in sorted(arrays_root.glob("*/*/*/raw_activity.npy")):
    sidecar_path = activity_path.with_suffix(".json")
    if not sidecar_path.is_file():
      raise FileNotFoundError(f"canonical activity sidecar is missing: {sidecar_path}")
    record = json.loads(sidecar_path.read_text(encoding="utf-8"))
    if record.get("split") == "test_locked":
      raise ValueError("normalization manifest must never include test_locked activity")
    if record.get("split") not in {"train", "validation"}:
      raise ValueError(f"unexpected canonical activity split: {record.get('split')!r}")
    value = np.load(activity_path, allow_pickle=False)
    actual_sha = array_sha256(value)
    if (
      record.get("schema_version") != 1
      or record.get("artifact_kind") != "canonical_activity"
      or record.get("activity_sha256") != actual_sha
      or record.get("support_policy") != "complete_square_no_content_crop_or_circular_mask"
    ):
      raise ValueError(f"canonical activity provenance mismatch: {activity_path}")
    try:
      portable = str(activity_path.resolve().relative_to(root))
    except ValueError:
      portable = str(activity_path.resolve())
    rows.append({
      "dataset": record["dataset"], "subject_id": record["subject_id"],
      "slice_id": record["slice_id"], "split": record["split"],
      "activity_path": portable, "activity_sha256": actual_sha,
      "activity_unit": record["activity_unit"],
    })
  if not rows:
    raise ValueError(f"no canonical activities found under {arrays_root}")
  with atomic_csv_writer(output_path, list(ACTIVITY_COLUMNS)) as writer:
    writer.writerows(rows)
  finalize_csv(
    output_path, schema_name="normalization_activity_manifest_v1",
    primary_key=["dataset", "subject_id", "slice_id"],
  )
  return output_path


def load_activity_cases(root: Path, manifest: Path) -> list[ActivityCase]:
  with manifest.open("r", encoding="utf-8", newline="") as handle:
    reader = csv.DictReader(handle)
    missing = set(ACTIVITY_COLUMNS) - set(reader.fieldnames or ())
    if missing:
      raise ValueError(f"normalization manifest missing columns: {sorted(missing)}")
    rows = list(reader)
  keys = [(row["dataset"], row["subject_id"], row["slice_id"]) for row in rows]
  if len(keys) != len(set(keys)):
    raise ValueError("normalization manifest contains duplicate dataset/subject/slice keys")
  cases = []
  for row in rows:
    if row["split"] not in {"train", "validation", "test_locked"}:
      raise ValueError(f"unknown split {row['split']!r}")
    cases.append(ActivityCase(
      dataset=row["dataset"], subject_id=row["subject_id"],
      slice_id=row["slice_id"], split=row["split"],
      path=(root / row["activity_path"]).resolve(),
      expected_sha256=row["activity_sha256"], unit=row["activity_unit"],
    ))
  return cases


def load_verified_activity(case: ActivityCase) -> np.ndarray:
  value = np.load(case.path)
  if value.shape != (128, 128):
    raise ValueError(f"{case.path} must be a pre-registered 128x128 activity slice")
  if not np.all(np.isfinite(value)) or np.any(value < 0):
    raise ValueError(f"{case.path} must be finite and non-negative")
  if array_sha256(value) != case.expected_sha256:
    raise ValueError(f"activity SHA mismatch: {case.path}")
  return np.asarray(value, dtype=np.float64)


def fit_candidates(
  training_arrays: list[np.ndarray], *, configured_fixed_C: float | None,
) -> list[NormalizationCandidate]:
  if not training_arrays:
    raise ValueError("normalization calibration has no training arrays")
  train_min = min(float(value.min()) for value in training_arrays)
  train_max = max(float(value.max()) for value in training_arrays)
  if train_max <= train_min:
    raise ValueError("training activity has no usable dynamic range")
  candidates = []
  if configured_fixed_C is not None:
    if configured_fixed_C <= 0:
      raise ValueError("configured physical C must be positive")
    candidates.append(NormalizationCandidate(
      "configured_physical_C", 0.0, float(configured_fixed_C), False,
      "primary_if_source_provenance_is_verified", "external_physical_metadata",
    ))
  candidates.extend((
    NormalizationCandidate(
      "train_max_fixed_C", 0.0, train_max, False,
      "data_derived_sensitivity_not_physical_scale", "train_only_global_max",
    ),
    NormalizationCandidate(
      "train_global_minmax", train_min, train_max - train_min, True,
      "sensitivity_analysis_only", "train_only_global_min_and_max",
    ),
    NormalizationCandidate(
      "per_image_minmax", None, None, True,
      "diagnostic_only_never_primary", "each_evaluated_image",
    ),
  ))
  return candidates


def apply_candidate(array: np.ndarray, candidate: NormalizationCandidate) -> tuple[np.ndarray, float, float]:
  value = np.asarray(array, dtype=np.float64)
  if candidate.name == "per_image_minmax":
    offset = float(value.min()); scale = float(value.max()) - offset
  else:
    offset = float(candidate.offset); scale = float(candidate.scale)
  if scale <= 0:
    raise ValueError(f"normalization candidate {candidate.name} has non-positive scale")
  transformed = (value - offset) / scale
  low_fraction = float(np.mean(transformed < 0.0))
  high_fraction = float(np.mean(transformed > 1.0))
  if candidate.clip:
    transformed = np.clip(transformed, 0.0, 1.0)
  return transformed, low_fraction, high_fraction


def run_normalization_calibration(
  root: Path, config_path: Path, protocol_path: Path,
) -> tuple[Path, Path]:
  cfg = load_yaml_strict(config_path)
  protocol = load_protocol(protocol_path)
  if protocol.kind != "normalization_calibration":
    raise ValueError("normalization calibration requires its dedicated protocol")
  source_manifest = root / str(protocol.raw["activity_manifest"])
  cases = load_activity_cases(root, source_manifest)
  output_dir = root / protocol.output
  run_metadata = write_run_metadata(
    output_dir, root=root, config_path=config_path, protocol_path=protocol_path,
  )
  candidate_output = output_dir / "normalization_candidates.csv"
  diagnostic_output = output_dir / "validation_diagnostics.csv"
  candidate_fields = [
    "run_id", "dataset", "candidate", "available", "offset", "scale", "clip",
    "activity_unit", "fitted_from", "paper_role", "train_cases",
    "validation_cases", "validation_max_clipped_fraction",
    "validation_mean_clipped_fraction",
  ]
  diagnostic_fields = [
    "run_id", "dataset", "subject_id", "slice_id", "candidate", "activity_unit",
    "raw_min", "raw_max", "normalized_min", "normalized_max",
    "normalized_mean", "normalized_std", "fraction_below_zero",
    "fraction_above_one", "paper_role",
  ]
  candidate_rows = []
  diagnostic_rows = []
  configured = cfg.get("normalization", {}).get("scale_constants", {})
  for dataset in protocol.raw["datasets"]:
    training_cases = [case for case in cases if case.dataset == dataset and case.split == "train"]
    validation_cases = [case for case in cases if case.dataset == dataset and case.split == "validation"]
    if not training_cases or not validation_cases:
      raise ValueError(f"{dataset} requires both train and validation activity cases")
    units = {case.unit for case in training_cases + validation_cases}
    if len(units) != 1:
      raise ValueError(f"{dataset} mixes activity units: {sorted(units)}")
    # Locked-test paths are deliberately never opened by this calibration.
    training = [load_verified_activity(case) for case in training_cases]
    validation = [(case, load_verified_activity(case)) for case in validation_cases]
    fixed = configured.get(dataset)
    candidates = fit_candidates(
      training, configured_fixed_C=None if fixed is None else float(fixed),
    )
    requested = set(protocol.methods)
    unknown = requested - {candidate.name for candidate in candidates} - {"configured_physical_C"}
    if unknown:
      raise ValueError(f"unknown normalization candidates: {sorted(unknown)}")
    for candidate_name in protocol.methods:
      matching = [candidate for candidate in candidates if candidate.name == candidate_name]
      if not matching:
        candidate_rows.append({
          "run_id": cfg["run_id"], "dataset": dataset,
          "candidate": candidate_name, "available": False,
          "offset": "", "scale": "", "clip": False,
          "activity_unit": next(iter(units)), "fitted_from": "external_physical_metadata",
          "paper_role": "unavailable_until_physical_C_is_supplied",
          "train_cases": len(training_cases), "validation_cases": len(validation_cases),
          "validation_max_clipped_fraction": "",
          "validation_mean_clipped_fraction": "",
        })
        continue
      candidate = matching[0]
      clipped = []
      for case, raw in validation:
        transformed, below, above = apply_candidate(raw, candidate)
        clipped.append(below + above)
        diagnostic_rows.append({
          "run_id": cfg["run_id"], "dataset": dataset,
          "subject_id": case.subject_id,
          "slice_id": case.slice_id, "candidate": candidate.name,
          "activity_unit": case.unit, "raw_min": float(raw.min()),
          "raw_max": float(raw.max()), "normalized_min": float(transformed.min()),
          "normalized_max": float(transformed.max()),
          "normalized_mean": float(transformed.mean()),
          "normalized_std": float(transformed.std()),
          "fraction_below_zero": below, "fraction_above_one": above,
          "paper_role": candidate.paper_role,
        })
      candidate_rows.append({
        "run_id": cfg["run_id"], "dataset": dataset,
        "candidate": candidate.name, "available": True,
        "offset": "" if candidate.offset is None else candidate.offset,
        "scale": "" if candidate.scale is None else candidate.scale,
        "clip": candidate.clip, "activity_unit": next(iter(units)),
        "fitted_from": candidate.fitted_from, "paper_role": candidate.paper_role,
        "train_cases": len(training_cases), "validation_cases": len(validation_cases),
        "validation_max_clipped_fraction": max(clipped),
        "validation_mean_clipped_fraction": float(np.mean(clipped)),
      })
  with atomic_csv_writer(candidate_output, candidate_fields) as writer:
    writer.writerows(candidate_rows)
  with atomic_csv_writer(diagnostic_output, diagnostic_fields) as writer:
    writer.writerows(diagnostic_rows)
  candidate_manifest = finalize_csv(
    candidate_output, schema_name="normalization_candidates_v2",
    primary_key=["run_id", "dataset", "candidate"], split="validation",
  )
  diagnostic_manifest = finalize_csv(
    diagnostic_output, schema_name="normalization_validation_diagnostics_v2",
    primary_key=["run_id", "dataset", "subject_id", "slice_id", "candidate"],
    split="validation",
  )
  write_success_marker(output_dir, [
    run_metadata, output_dir / "environment_freeze.txt", candidate_output,
    candidate_manifest, diagnostic_output, diagnostic_manifest, source_manifest,
  ])
  return candidate_output, diagnostic_output
