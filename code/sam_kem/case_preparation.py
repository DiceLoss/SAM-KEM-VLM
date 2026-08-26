"""Batch construction of matched projections, Poisson cases, and MLEM inputs."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .artifacts import file_sha256
from .config import require_test_release
from .mlem_cache import write_mlem_input
from .noise import array_sha256, write_low_count_simulation
from .pet_simulation import ProjectionContext, prepare_noise_free_projection
from .reproducibility import CaseKey, case_seed


def _group_state(paths: tuple[Path, ...], *, resume_verified: bool) -> bool:
  exists = [path.is_file() for path in paths]
  if any(exists) and not all(exists):
    raise RuntimeError(f"partial immutable artifact group requires quarantine: {paths}")
  if all(exists) and not resume_verified:
    raise FileExistsError(f"artifacts already exist; use --resume-verified: {paths[0]}")
  return all(exists)


def _verify_projection(
  raw_path: Path, target_path: Path, projection_path: Path, context, dataset: str,
) -> None:
  raw = np.load(raw_path, allow_pickle=False)
  target = np.load(target_path, allow_pickle=False)
  projection = np.load(projection_path, allow_pickle=False)
  record = json.loads(projection_path.with_suffix(".json").read_text(encoding="utf-8"))
  ingestion_path = raw_path.with_suffix(".json")
  if (
    record.get("schema_version") != 2
    or record.get("artifact_kind") != "noise_free_pet_projection"
    or record.get("raw_activity_sha256") != array_sha256(raw)
    or record.get("target_sha256") != array_sha256(target)
    or record.get("noise_free_projection_sha256") != array_sha256(projection)
    or record.get("ingestion_metadata_sha256") != file_sha256(ingestion_path)
    or record.get("operator_id") != context.contract.operator_id
    or record.get("dataset") != dataset
    or float(record.get("normalization_scale_C", -1))
    != context.contract.scale_constants[dataset]
  ):
    raise ValueError(f"existing projection group failed verification: {projection_path}")


def _verify_sinogram(path: Path, projection: np.ndarray, key: CaseKey, context) -> float:
  measured = np.load(path, allow_pickle=False)
  record = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
  expected_seed = case_seed(key, experiment_seed=context.contract.experiment_seed)
  if (
    record.get("schema_version") != 2
    or record.get("artifact_kind") != "poisson_sinogram"
    or record.get("sinogram_sha256") != array_sha256(measured)
    or record.get("noise_free_projection_sha256") != array_sha256(projection)
    or int(record.get("seed", -1)) != expected_seed
    or int(record.get("count_level", -1)) != key.count_level
    or int(record.get("realization_id", -1)) != key.realization_id
    or record.get("dataset") != key.dataset
    or record.get("subject_id") != key.subject_id
    or record.get("slice_id") != key.slice_id
  ):
    raise ValueError(f"existing sinogram group failed verification: {path}")
  return float(record["projection_scale"])


def _verify_mlem(path: Path, sinogram: np.ndarray, context) -> None:
  image = np.load(path, allow_pickle=False)
  record = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
  if (
    record.get("schema_version") != 2
    or record.get("artifact_kind") != "mlem_prompt_input"
    or record.get("mlem10_sha256") != array_sha256(image)
    or record.get("sinogram_sha256") != array_sha256(sinogram)
    or record.get("operator_id") != context.contract.operator_id
    or int(record.get("iterations", -1)) != context.contract.prompt_mlem_iterations
  ):
    raise ValueError(f"existing MLEM group failed verification: {path}")


def prepare_cases(
  root: Path, config_path: Path, arrays_root: Path, *, splits: set[str],
  resume_verified: bool = False,
) -> dict[str, int]:
  context = ProjectionContext.build(root, config_path)
  if "test_locked" in splits:
    require_test_release(root, context.cfg)
  raw_paths = sorted(arrays_root.glob("*/*/*/raw_activity.npy"))
  selected = []
  for raw_path in raw_paths:
    ingestion = json.loads(raw_path.with_suffix(".json").read_text(encoding="utf-8"))
    relative = raw_path.relative_to(arrays_root)
    if len(relative.parts) != 4 or relative.name != "raw_activity.npy":
      raise ValueError(f"invalid canonical activity layout: {raw_path}")
    path_identity = relative.parts[:3]
    sidecar_identity = (
      str(ingestion.get("dataset")), str(ingestion.get("subject_id")),
      str(ingestion.get("slice_id")),
    )
    if (
      ingestion.get("schema_version") != 1
      or ingestion.get("artifact_kind") != "canonical_activity"
      or path_identity != sidecar_identity
      or ingestion.get("activity_sha256")
      != array_sha256(np.load(raw_path, allow_pickle=False))
    ):
      raise ValueError(f"canonical activity path/sidecar mismatch: {raw_path}")
    if ingestion.get("split") in splits:
      selected.append((raw_path, ingestion))
  if not selected:
    raise ValueError(f"no ingested activity cases match splits {sorted(splits)}")
  summary = {"activity_cases": 0, "poisson_cases": 0, "reused_groups": 0}
  for raw_path, ingestion in selected:
    dataset = str(ingestion["dataset"])
    subject = str(ingestion["subject_id"])
    slice_id = str(ingestion["slice_id"])
    case_dir = raw_path.parent
    target_path = case_dir / "target.npy"
    projection_path = case_dir / "noise_free_projection.npy"
    projection_group = (
      target_path, projection_path, projection_path.with_suffix(".json"),
    )
    if _group_state(projection_group, resume_verified=resume_verified):
      _verify_projection(raw_path, target_path, projection_path, context, dataset)
      summary["reused_groups"] += 1
    else:
      prepare_noise_free_projection(
        root, config_path, dataset=dataset, raw_activity_path=raw_path,
        normalized_target_path=target_path, projection_path=projection_path,
        context=context,
      )
    projection = np.load(projection_path, allow_pickle=False)
    summary["activity_cases"] += 1
    for count in context.contract.count_levels:
      for realization in context.contract.realizations:
        output_dir = case_dir / f"c{count}" / f"r{realization}"
        sinogram_path = output_dir / "sinogram.npy"
        sinogram_group = (sinogram_path, sinogram_path.with_suffix(".json"))
        key = CaseKey(dataset, subject, slice_id, count, realization)
        if _group_state(sinogram_group, resume_verified=resume_verified):
          scale = _verify_sinogram(sinogram_path, projection, key, context)
          summary["reused_groups"] += 1
        else:
          write_low_count_simulation(
            projection_path, sinogram_path, expected_true_counts=count,
            case=key, experiment_seed=context.contract.experiment_seed,
          )
          scale = count / float(projection.sum())
        mlem_path = output_dir / "mlem10.npy"
        mlem_group = (mlem_path, mlem_path.with_suffix(".json"))
        if _group_state(mlem_group, resume_verified=resume_verified):
          _verify_mlem(
            mlem_path, np.load(sinogram_path, allow_pickle=False), context,
          )
          summary["reused_groups"] += 1
        else:
          write_mlem_input(
            root, config_path, sinogram_path, mlem_path,
            projection_scale=scale, context=context,
          )
        summary["poisson_cases"] += 1
  return summary
