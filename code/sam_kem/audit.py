"""Repository-level scientific audit. A warning can never produce PASS."""

from __future__ import annotations

import csv
import hashlib
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .boxes import BoxCache, BoxKey
from .config import ConfigurationError, ScientificContract, load_yaml_strict
from .splits import load_configured_subject_splits, split_counts
from .reproducibility import CaseKey, case_seed
from .noise import array_sha256
from .artifacts import file_sha256, verify_csv_artifact, verify_success_marker
from .ingestion import load_source_manifest


@dataclass(frozen=True)
class Issue:
  code: str
  severity: str
  message: str


def _read_csv(path: Path) -> list[dict[str, str]]:
  with path.open("r", encoding="utf-8", newline="") as handle:
    return list(csv.DictReader(handle))


def _matches_expected_split_counts(
  actual: dict[str, int], expected: dict[str, int],
) -> bool:
  """Compare configured split counts while treating an absent split as zero."""
  split_names = set(actual) | set(expected)
  return all(int(actual.get(name, 0)) == int(expected.get(name, 0)) for name in split_names)


def audit_repository(root: Path, config_path: Path) -> list[Issue]:
  issues: list[Issue] = []
  contract: ScientificContract | None = None
  try:
    cfg = load_yaml_strict(config_path)
  except (OSError, ConfigurationError, ValueError) as error:
    return [Issue("CONFIG", "fatal", str(error))]
  try:
    contract = ScientificContract.from_mapping(cfg)
  except (ConfigurationError, ValueError) as error:
    issues.append(Issue("CONTRACT", "fatal", str(error)))
  if cfg.get("locked") is not True or cfg.get("status") != "locked_for_execution":
    issues.append(Issue("CONFIG_UNLOCKED", "fatal", "set locked=true and status=locked_for_execution only after resolving provenance"))
  run_id = str(cfg.get("run_id", ""))
  namespaced_paths = [
    cfg.get("data", {}).get("split_file", ""),
    cfg.get("data", {}).get("source_activity_manifest", ""),
    cfg.get("data", {}).get("canonical_activity_root", ""),
    cfg.get("prompts", {}).get("box_cache", ""),
    cfg.get("prompts", {}).get("training_box_cache", ""),
    *cfg.get("medsam2", {}).get("checkpoint_paths", {}).values(),
    cfg.get("outputs", {}).get("per_case_metrics", ""),
    cfg.get("outputs", {}).get("hyperparameter_sweep", ""),
    cfg.get("outputs", {}).get("frozen_hyperparameters", ""),
    cfg.get("outputs", {}).get("pretraining_manifest", ""),
    cfg.get("outputs", {}).get("validation_manifest", ""),
    cfg.get("outputs", {}).get("box_input_manifest", ""),
    cfg.get("outputs", {}).get("raw_array_manifest", ""),
  ]
  missing_namespace = [path for path in namespaced_paths if run_id not in str(path)]
  if not run_id or missing_namespace:
    issues.append(Issue(
      "RUN_NAMESPACE", "fatal",
      f"clean run_id {run_id!r} is absent from artifact paths: {missing_namespace}",
    ))
  protocol_namespace_errors = []
  for protocol_path in sorted((root / "code/configs/protocols").glob("*.yaml")):
    protocol_mapping = load_yaml_strict(protocol_path)
    if run_id not in str(protocol_mapping.get("output", "")):
      protocol_namespace_errors.append(str(protocol_path.relative_to(root)))
  if protocol_namespace_errors:
    issues.append(Issue(
      "PROTOCOL_NAMESPACE", "fatal",
      f"protocol outputs omit run_id {run_id!r}: {protocol_namespace_errors}",
    ))

  try:
    splits = load_configured_subject_splits(root, cfg)
  except (OSError, ValueError) as error:
    issues.append(Issue("SPLIT", "fatal", str(error)))
    splits = []
  if splits:
    actual = split_counts(splits)
    expected = cfg["data"].get("expected_subject_counts", {})
    for dataset, split_expectation in expected.items():
      if not _matches_expected_split_counts(actual.get(dataset, {}), split_expectation):
        issues.append(Issue(
          "SPLIT_COUNTS", "fatal",
          f"{dataset} subject counts {actual.get(dataset, {})} != {split_expectation}",
        ))

  source_manifest_path = root / str(cfg.get("data", {}).get("source_activity_manifest", ""))
  try:
    source_cases = load_source_manifest(root, source_manifest_path)
  except (OSError, ValueError) as error:
    issues.append(Issue("SOURCE_MANIFEST", "fatal", str(error)))
  else:
    split_index = {
      (row["dataset"], row["subject_id"]): row["split"] for row in splits
    }
    allowed_transforms = cfg["data"].get("allowed_spatial_transforms", {})
    bad_sources = []
    for case in source_cases:
      identity = (case.dataset, case.subject_id)
      if (
        split_index.get(identity) != case.split
        or case.spatial_transform not in set(allowed_transforms.get(case.dataset, ()))
      ):
        bad_sources.append((identity, case.slice_id))
    if bad_sources:
      issues.append(Issue(
        "SOURCE_CONTRACT", "fatal",
        f"{len(bad_sources)} source rows violate split/transform contract; first={bad_sources[0]}",
      ))

  operator_path = root / str(cfg["forward_model"].get("implementation_path", ""))
  expected_operator_sha = cfg["forward_model"].get("implementation_sha256")
  if not operator_path.is_file():
    issues.append(Issue("OPERATOR_MISSING", "fatal", f"missing {operator_path}"))
  elif hashlib.sha256(operator_path.read_bytes()).hexdigest() != expected_operator_sha:
    issues.append(Issue("OPERATOR_SHA", "fatal", "forward-operator implementation SHA mismatch"))

  clean_sources = list((root / "code/sam_kem").glob("*.py")) + list(
    (root / "code/experiments").glob("*.py")
  )
  forbidden_legacy_references = {
    "src/original", "code/src/original", "import recon  #", "import EM  #",
    "from discrete_radon import",
  }
  legacy_links = []
  for source in clean_sources:
    if source.resolve() == Path(__file__).resolve():
      continue
    text = source.read_text(encoding="utf-8")
    matches = sorted(token for token in forbidden_legacy_references if token in text)
    if matches:
      legacy_links.append((str(source.relative_to(root)), matches))
  if legacy_links:
    issues.append(Issue(
      "LEGACY_CODE_DEPENDENCY", "fatal",
      f"clean pipeline imports retired code: {legacy_links}",
    ))

  checkpoint_cfg = cfg.get("medsam2", {})
  if not checkpoint_cfg.get("source_repository") or not checkpoint_cfg.get("source_commit"):
    issues.append(Issue("MEDSAM2_SOURCE", "fatal", "MedSAM2 source repository/commit is unresolved"))
  source_checkout_value = checkpoint_cfg.get("source_checkout_path")
  if not source_checkout_value:
    issues.append(Issue("MEDSAM2_CHECKOUT", "fatal", "MedSAM2 source_checkout_path is unresolved"))
  else:
    source_checkout = root / str(source_checkout_value)
    try:
      source_head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=source_checkout, check=True,
        capture_output=True, text=True,
      ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
      issues.append(Issue("MEDSAM2_CHECKOUT", "fatal", f"not a readable Git checkout: {source_checkout}"))
    else:
      if source_head != checkpoint_cfg.get("source_commit"):
        issues.append(Issue(
          "MEDSAM2_CHECKOUT", "fatal",
          f"checkout HEAD {source_head} != configured commit {checkpoint_cfg.get('source_commit')}",
        ))
  pretrained_path = root / str(checkpoint_cfg.get("pretrained_checkpoint_path", ""))
  pretrained_sha = checkpoint_cfg.get("pretrained_checkpoint_sha256")
  if not pretrained_path.is_file() or not pretrained_sha:
    issues.append(Issue("MEDSAM2_PRETRAINED", "fatal", "pretrained MedSAM2 checkpoint/path SHA is unresolved"))
  elif hashlib.sha256(pretrained_path.read_bytes()).hexdigest() != pretrained_sha:
    issues.append(Issue("MEDSAM2_PRETRAINED", "fatal", "pretrained MedSAM2 checkpoint SHA mismatch"))
  for dataset in ("brainweb", "hecktor"):
    if not checkpoint_cfg.get("checkpoint_archive_uri", {}).get(dataset):
      issues.append(Issue("CHECKPOINT_ARCHIVE", "fatal", f"immutable checkpoint archive URI missing for {dataset}"))
    expected_sha = checkpoint_cfg.get("checkpoint_sha256", {}).get(dataset)
    relative_path = checkpoint_cfg.get("checkpoint_paths", {}).get(dataset)
    if not expected_sha or not isinstance(expected_sha, str) or len(expected_sha) != 64:
      issues.append(Issue("CHECKPOINT_SHA", "fatal", f"unresolved checkpoint SHA for {dataset}"))
      continue
    checkpoint_path = root / str(relative_path)
    if not checkpoint_path.is_file():
      issues.append(Issue("CHECKPOINT_MISSING", "fatal", f"missing {checkpoint_path}"))
      continue
    actual_sha = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
    if actual_sha != expected_sha:
      issues.append(Issue("CHECKPOINT_MISMATCH", "fatal", f"{dataset}: {actual_sha} != {expected_sha}"))
    provenance_path = checkpoint_path.with_suffix(".json")
    if not provenance_path.is_file():
      issues.append(Issue("TRAINING_PROVENANCE", "fatal", f"missing {provenance_path}"))
      continue
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    if provenance.get("checkpoint_sha256") != actual_sha:
      issues.append(Issue("TRAINING_PROVENANCE", "fatal", f"sidecar SHA mismatch for {dataset}"))
    if (
      provenance.get("checkpoint_scope") != "mask_decoder_only"
      or provenance.get("selection_policy")
      != checkpoint_cfg.get("checkpoint_selection_policy")
    ):
      issues.append(Issue("CHECKPOINT_FORMAT", "fatal", f"wrong checkpoint scope/selection claim for {dataset}"))
    resume_path = checkpoint_path.with_suffix(".resume.pt")
    metrics_path = checkpoint_path.with_suffix(".training.csv")
    try:
      metrics_record = verify_csv_artifact(metrics_path)
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as error:
      issues.append(Issue("TRAINING_METRICS", "fatal", f"{dataset}: {error}"))
    else:
      if (
        provenance.get("training_metrics_sha256") != file_sha256(metrics_path)
        or int(metrics_record.get("row_count", -1)) != int(checkpoint_cfg["training"]["epochs"])
      ):
        issues.append(Issue("TRAINING_METRICS", "fatal", f"metrics provenance/epoch count mismatch for {dataset}"))
    if (
      not resume_path.is_file()
      or provenance.get("resume_checkpoint_sha256") != (
        file_sha256(resume_path) if resume_path.is_file() else None
      )
    ):
      issues.append(Issue("RESUME_CHECKPOINT", "fatal", f"missing or changed resume state for {dataset}"))
    try:
      import torch
      payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    except (ImportError, OSError, RuntimeError, ValueError) as error:
      issues.append(Issue("CHECKPOINT_PAYLOAD", "fatal", f"cannot inspect {dataset} checkpoint: {error}"))
    else:
      if (
        payload.get("schema_version") != 1
        or payload.get("artifact_kind") != "inference_mask_decoder"
        or payload.get("dataset") != dataset
        or payload.get("run_id") != run_id
        or contract is None
        or payload.get("normalization_id") != contract.normalization_id
        or payload.get("pretrained_checkpoint_sha256") != pretrained_sha
        or "mask_decoder_state" not in payload or "model_state" in payload
      ):
        issues.append(Issue("CHECKPOINT_PAYLOAD", "fatal", f"invalid decoder-only payload for {dataset}"))
    if provenance.get("non_mask_decoder_trainable_names"):
      issues.append(Issue("ENCODER_NOT_FROZEN", "fatal", f"non-decoder trainable parameters for {dataset}"))
    if provenance.get("run_id") != run_id:
      issues.append(Issue("CHECKPOINT_RUN_ID", "fatal", f"wrong run_id for {dataset}"))
    if contract is not None and provenance.get("normalization_id") != contract.normalization_id:
      issues.append(Issue("CHECKPOINT_NORMALIZATION", "fatal", f"wrong normalization identity for {dataset}"))
    if provenance.get("loss") != "l1_only_raw_decoder_output":
      issues.append(Issue("TRAINING_LOSS", "fatal", f"unexpected loss for {dataset}"))
    counts = {int(key): int(value) for key, value in provenance.get("training_count_samples", {}).items()}
    if set(counts) != set(cfg["acquisition"]["count_levels"]) or len(set(counts.values())) != 1:
      issues.append(Issue("TRAINING_IMBALANCE", "fatal", f"unbalanced/missing count samples for {dataset}: {counts}"))
    subject_samples = {
      str(key): int(value)
      for key, value in provenance.get("training_subject_samples", {}).items()
    }
    if (
      not subject_samples
      or max(subject_samples.values()) - min(subject_samples.values())
      > len(cfg["acquisition"]["count_levels"])
    ):
      issues.append(Issue(
        "TRAINING_SUBJECT_IMBALANCE", "fatal",
        f"subject-hierarchical training balance failed for {dataset}: {subject_samples}",
      ))
    if splits:
      allowed_train = {
        row["subject_id"] for row in splits
        if row["dataset"] == dataset and row["split"] == "train"
      }
      used = set(provenance.get("train_subject_ids", []))
      if used != allowed_train:
        issues.append(Issue(
          "TRAINING_SUBJECT_COVERAGE", "fatal",
          f"training subjects differ for {dataset}: "
          f"missing={sorted(allowed_train - used)}, extra={sorted(used - allowed_train)}",
        ))
    if provenance.get("medsam2_source_commit") != checkpoint_cfg.get("source_commit"):
      issues.append(Issue("TRAINING_SOURCE", "fatal", f"wrong MedSAM2 source commit for {dataset}"))
    declared_training = checkpoint_cfg.get("training", {})
    if (
      provenance.get("optimizer") != declared_training.get("optimizer")
      or float(provenance.get("weight_decay", -1)) != float(declared_training.get("weight_decay", -2))
      or int(provenance.get("batch_size", -1)) != int(declared_training.get("batch_size", -2))
      or provenance.get("sampling_unit") != declared_training.get("sampling_unit")
    ):
      issues.append(Issue("TRAINING_PROTOCOL", "fatal", f"optimizer/batch protocol mismatch for {dataset}"))

  cache_path = root / str(cfg["prompts"]["box_cache"])
  try:
    cache_record = verify_csv_artifact(cache_path)
    if cache_record.get("schema_name") != "qwen_anchor_box_cache_v2":
      raise ValueError("unexpected box-cache schema")
    cache = BoxCache.read(cache_path)
  except (OSError, ValueError, KeyError, json.JSONDecodeError) as error:
    issues.append(Issue("BOX_CACHE", "fatal", str(error)))
    cache = None
  training_cache_path = root / str(cfg["prompts"]["training_box_cache"])
  try:
    training_cache_record = verify_csv_artifact(training_cache_path)
    if training_cache_record.get("schema_name") != "qwen_anchor_box_cache_v2":
      raise ValueError("unexpected training box-cache schema")
    training_cache = BoxCache.read(training_cache_path)
  except (OSError, ValueError, KeyError, json.JSONDecodeError) as error:
    issues.append(Issue("TRAINING_BOX_CACHE", "fatal", str(error)))
    training_cache = None

  manifest_path = root / str(cfg["outputs"]["raw_array_manifest"])
  for manifest_name in ("pretraining_manifest", "validation_manifest", "box_input_manifest"):
    staged_manifest = root / str(cfg["outputs"][manifest_name])
    try:
      staged_record = verify_csv_artifact(staged_manifest)
      if staged_record.get("schema_name") != "canonical_case_manifest_v3":
        raise ValueError("unexpected case-manifest schema")
      staged_rows = _read_csv(staged_manifest)
      if not staged_rows or any(
        row.get("run_id") != run_id
        or row.get("split_id") != cfg["data"]["split_id"]
        or row.get("split_sha256") != cfg["data"]["split_sha256"]
        for row in staged_rows
      ):
        raise ValueError("manifest is empty or has wrong run/split identity")
      if manifest_name == "pretraining_manifest" and any(
        row.get("split") == "test_locked" for row in staged_rows
      ):
        raise ValueError("pretraining manifest contains locked-test rows")
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as error:
      issues.append(Issue(manifest_name.upper(), "fatal", str(error)))
  if not manifest_path.is_file():
    issues.append(Issue("MANIFEST_MISSING", "fatal", f"missing canonical case manifest: {manifest_path}"))
  else:
    try:
      manifest_record = verify_csv_artifact(manifest_path)
      if manifest_record.get("schema_name") != "canonical_case_manifest_v3":
        raise ValueError("unexpected canonical case-manifest schema")
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as error:
      issues.append(Issue("MANIFEST_ARTIFACT", "fatal", str(error)))
  if manifest_path.is_file() and cache is not None:
    manifest_rows = _read_csv(manifest_path)
    needed_boxes = {
      BoxKey(row["dataset"], row["subject_id"], row["slice_id"], int(row["count_level"]))
      for row in manifest_rows
    }
    missing_boxes = sorted(key for key in needed_boxes if cache.get(key) is None)
    if missing_boxes:
      issues.append(Issue("BOX_COVERAGE", "fatal", f"canonical cache misses {len(missing_boxes)} keys"))
    wrong_cache_run = [
      key for key in needed_boxes
      if cache.get(key) is not None and cache.get(key).run_id != run_id
    ]
    if wrong_cache_run:
      issues.append(Issue("BOX_RUN_ID", "fatal", f"{len(wrong_cache_run)} boxes have wrong run_id"))
    if training_cache is not None:
      training_needed = {
        BoxKey(row["dataset"], row["subject_id"], row["slice_id"], int(row["count_level"]))
        for row in manifest_rows if row["split"] in {"train", "validation"}
      }
      training_missing = [
        key for key in training_needed
        if training_cache.get(key) is None
        or training_cache.get(key).run_id != run_id
      ]
      if training_missing:
        issues.append(Issue(
          "TRAINING_BOX_COVERAGE", "fatal",
          f"training/validation cache misses {len(training_missing)} keys",
        ))
    invalid_qwen_records = [
      key for key in needed_boxes
      if cache.get(key) is not None
      and cache.get(key).source == "qwen3-vl-plus"
      and (
        cache.get(key).run_id != run_id
        or not cache.get(key).request_id or not cache.get(key).image_sha256
        or not cache.get(key).prompt_sha256
        or cache.get(key).model != cfg["prompts"]["model"]
      )
    ]
    if invalid_qwen_records:
      issues.append(Issue("BOX_PROVENANCE", "fatal", f"{len(invalid_qwen_records)} Qwen boxes lack request/image provenance"))
    anchor_errors = []
    for key in needed_boxes:
      anchor = [
        row for row in manifest_rows
        if row["dataset"] == key.dataset and row["subject_id"] == key.subject_id
        and row["slice_id"] == key.slice_id and int(row["count_level"]) == key.count_level
        and int(row["realization_id"]) == int(cfg["prompts"]["anchor_realization"])
      ]
      if len(anchor) != 1:
        anchor_errors.append(key)
        continue
      try:
        anchor_image = np.load(root / anchor[0]["mlem10_input_path"])
      except (OSError, ValueError):
        anchor_errors.append(key)
        continue
      if cache.get(key).source_mlem10_sha256 != array_sha256(anchor_image):
        anchor_errors.append(key)
    if anchor_errors:
      issues.append(Issue(
        "BOX_ANCHOR", "fatal",
        f"{len(anchor_errors)} box records are not bound to the declared anchor realization",
      ))
    if contract is not None:
      bad_manifest_rows = []
      for row in manifest_rows:
        key = CaseKey(
          row["dataset"], row["subject_id"], row["slice_id"],
          int(row["count_level"]), int(row["realization_id"]),
        )
        expected_seed = case_seed(key, experiment_seed=contract.experiment_seed)
        valid = (
          row.get("run_id") == contract.run_id
          and row.get("split_id") == contract.split_id
          and row.get("split_sha256") == contract.split_sha256
          and int(row["seed"]) == expected_seed
          and row["operator_id"] == contract.operator_id
          and float(row["scale_constant_C"]) == contract.scale_constants[row["dataset"]]
          and row.get("normalization_id") == contract.normalization_id
          and float(row.get("normalization_offset", "nan"))
          == contract.normalization_offsets[row["dataset"]]
          and (row.get("normalization_clip", "").lower() == "true")
          == contract.normalization_clip
          and float(row["projection_scale"]) > 0
        )
        if not valid:
          bad_manifest_rows.append(key)
      if bad_manifest_rows:
        issues.append(Issue("MANIFEST_CONTRACT", "fatal", f"{len(bad_manifest_rows)} manifest rows violate seed/scale/operator contract"))
      artifact_errors: list[str] = []
      for row in manifest_rows:
        try:
          if row.get("evaluation_mask_path"):
            raise ValueError("external evaluation mask is prohibited; use operator support")
          target_path = root / row["target_path"]
          projection_path = root / row["noise_free_projection_path"]
          projection_meta_path = root / row["projection_metadata_path"]
          sinogram_path = root / row["sinogram_path"]
          simulation_meta_path = root / row["simulation_metadata_path"]
          mlem_path = root / row["mlem10_input_path"]
          mlem_meta_path = root / row["mlem10_metadata_path"]
          for artifact in (
            target_path, projection_path, projection_meta_path,
            sinogram_path, simulation_meta_path, mlem_path, mlem_meta_path,
          ):
            if not artifact.is_file():
              raise FileNotFoundError(artifact)
          target = np.load(target_path)
          projection = np.load(projection_path)
          sinogram = np.load(sinogram_path)
          mlem_input = np.load(mlem_path)
          projection_meta = json.loads(projection_meta_path.read_text(encoding="utf-8"))
          simulation_meta = json.loads(simulation_meta_path.read_text(encoding="utf-8"))
          mlem_meta = json.loads(mlem_meta_path.read_text(encoding="utf-8"))
          expected_projection_shape = (contract.angles, contract.detector_bins)
          if target.shape != (128, 128) or mlem_input.shape != target.shape:
            raise ValueError("target/MLEM shape mismatch")
          if projection.shape != expected_projection_shape or sinogram.shape != projection.shape:
            raise ValueError("projection/sinogram shape mismatch")
          if (
            not np.all(np.isfinite(target)) or np.any(target < 0)
            or not np.all(np.isfinite(projection)) or np.any(projection < 0)
            or not np.all(np.isfinite(mlem_input)) or np.any(mlem_input < 0)
            or np.any(sinogram < 0) or not np.issubdtype(sinogram.dtype, np.integer)
          ):
            raise ValueError("invalid PET array range, finiteness, or count dtype")
          if projection_meta.get("target_sha256") != array_sha256(target):
            raise ValueError("target hash differs from projection sidecar")
          if (
            projection_meta.get("schema_version") != 2
            or projection_meta.get("artifact_kind") != "noise_free_pet_projection"
            or projection_meta.get("normalization_id") != contract.normalization_id
            or float(projection_meta.get("normalization_offset", "nan"))
            != contract.normalization_offsets[row["dataset"]]
            or bool(projection_meta.get("normalization_clip"))
            != contract.normalization_clip
          ):
            raise ValueError("projection sidecar has the wrong normalization identity")
          ingestion_meta_path = Path(str(projection_meta.get("ingestion_metadata_path", "")))
          if not ingestion_meta_path.is_absolute():
            ingestion_meta_path = root / ingestion_meta_path
          if (
            not ingestion_meta_path.is_file()
            or projection_meta.get("ingestion_metadata_sha256")
            != file_sha256(ingestion_meta_path)
          ):
            raise ValueError("projection is not bound to canonical ingestion provenance")
          ingestion_meta = json.loads(ingestion_meta_path.read_text(encoding="utf-8"))
          raw_activity_path = Path(str(projection_meta.get("raw_activity_path", "")))
          if not raw_activity_path.is_absolute():
            raw_activity_path = root / raw_activity_path
          if (
            ingestion_meta.get("artifact_kind") != "canonical_activity"
            or not raw_activity_path.is_file()
            or ingestion_meta.get("activity_sha256")
            != array_sha256(np.load(raw_activity_path, allow_pickle=False))
          ):
            raise ValueError("canonical ingested activity is missing or changed")
          if projection_meta.get("noise_free_projection_sha256") != array_sha256(projection):
            raise ValueError("projection hash differs from projection sidecar")
          if simulation_meta.get("noise_free_projection_sha256") != array_sha256(projection):
            raise ValueError("projection hash differs from simulation sidecar")
          if (
            simulation_meta.get("schema_version") != 2
            or simulation_meta.get("artifact_kind") != "poisson_sinogram"
          ):
            raise ValueError("sinogram sidecar has an obsolete schema")
          if simulation_meta.get("sinogram_sha256") != array_sha256(sinogram):
            raise ValueError("sinogram hash differs from simulation sidecar")
          if int(simulation_meta.get("expected_total_counts", -1)) != int(row["count_level"]):
            raise ValueError("expected count differs from case count level")
          if int(simulation_meta.get("realized_total_counts", -1)) != int(sinogram.sum()):
            raise ValueError("realized count differs from sinogram total")
          if (
            mlem_meta.get("schema_version") != 2
            or mlem_meta.get("artifact_kind") != "mlem_prompt_input"
            or mlem_meta.get("mlem10_sha256") != array_sha256(mlem_input)
            or mlem_meta.get("sinogram_sha256") != array_sha256(sinogram)
            or mlem_meta.get("operator_id") != contract.operator_id
            or int(mlem_meta.get("iterations", -1)) != contract.prompt_mlem_iterations
          ):
            raise ValueError("MLEM cache differs from provenance sidecar")
          expected_scale = int(row["count_level"]) / float(projection.sum())
          if not np.isclose(float(row["projection_scale"]), expected_scale, rtol=1e-12):
            raise ValueError("manifest projection scale differs from array")
          if row.get("anatomy_path"):
            anatomy = np.load(root / row["anatomy_path"])
            if (
              anatomy.shape != target.shape or not np.all(np.isfinite(anatomy))
              or row.get("anatomy_sha256") != array_sha256(anatomy)
            ):
              raise ValueError("anatomy baseline input is invalid or changed")
          for prior_name, metadata_name, released in (
            ("sam_prior_path", "sam_prior_metadata_path", False),
            ("sam_prior_centroid_path", "sam_prior_centroid_metadata_path", False),
            ("sam_prior_released_path", "sam_prior_released_metadata_path", True),
          ):
            if not row.get(prior_name):
              continue
            prior = np.load(root / row[prior_name])
            prior_meta = json.loads((root / row[metadata_name]).read_text(encoding="utf-8"))
            expected_checkpoint = (
              cfg["medsam2"]["pretrained_checkpoint_sha256"] if released
              else cfg["medsam2"]["checkpoint_sha256"][row["dataset"]]
            )
            if (
              prior.shape != target.shape or not np.all(np.isfinite(prior))
              or prior_meta.get("schema_version") != 2
              or prior_meta.get("artifact_kind") != "medsam2_continuous_prior"
              or prior_meta.get("run_id") != contract.run_id
              or prior_meta.get("prior_sha256") != array_sha256(prior)
              or prior_meta.get("input_mlem10_sha256") != array_sha256(mlem_input)
              or prior_meta.get("normalization_id") != contract.normalization_id
              or prior_meta.get("decoder_checkpoint_sha256") != expected_checkpoint
            ):
              raise ValueError(f"prior or its provenance changed: {prior_name}")
        except (KeyError, OSError, EOFError, ValueError, TypeError, json.JSONDecodeError) as error:
          artifact_errors.append(
            f"{row.get('dataset')}/{row.get('subject_id')}/{row.get('slice_id')}/"
            f"c{row.get('count_level')}/r{row.get('realization_id')}: {error}"
          )
      if artifact_errors:
        issues.append(Issue(
          "MANIFEST_ARTIFACT_PROVENANCE", "fatal",
          f"{len(artifact_errors)} manifest rows have missing/tampered/incompatible arrays; "
          f"first={artifact_errors[0]}",
        ))

  metrics_path = root / str(cfg["outputs"]["per_case_metrics"])
  if not cfg["outputs"]["artifact_policy"].get("immutable_archive_uri"):
    issues.append(Issue("RESULT_ARCHIVE", "fatal", "immutable result archive URI is unresolved"))
  run_metadata_path = metrics_path.parent / "run_metadata.json"
  if not run_metadata_path.is_file():
    issues.append(Issue("RUN_METADATA", "fatal", f"missing {run_metadata_path}"))
  else:
    run_metadata = json.loads(run_metadata_path.read_text(encoding="utf-8"))
    if run_metadata.get("git_worktree_dirty") is not False:
      issues.append(Issue("DIRTY_RUN", "fatal", "canonical results were produced from a dirty worktree"))
    if run_metadata.get("run_id") != run_id:
      issues.append(Issue("RUN_METADATA_ID", "fatal", "result run metadata has wrong run_id"))
    if run_metadata.get("config_sha256") != hashlib.sha256(config_path.read_bytes()).hexdigest():
      issues.append(Issue("RUN_CONFIG", "fatal", "result run config does not match current locked config"))
  try:
    verify_success_marker(metrics_path.parent)
  except (OSError, ValueError, KeyError, json.JSONDecodeError) as error:
    issues.append(Issue("RUN_INCOMPLETE", "fatal", str(error)))
  if metrics_path.is_file():
    try:
      result_record = verify_csv_artifact(metrics_path)
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as error:
      issues.append(Issue("RESULT_ARTIFACT", "fatal", str(error)))
      rows = _read_csv(metrics_path)
    else:
      rows = _read_csv(metrics_path)
      if result_record.get("schema_name") != "per_case_reconstruction_metrics_v2":
        issues.append(Issue("RESULT_SCHEMA", "fatal", "unexpected per-case result schema"))
    numeric_fields = (
      "psnr_db", "ssim", "nrmse", "mae", "poisson_deviance",
      "scale_constant_C", "sigma", "radius", "neighbours", "iterations", "alpha",
    )
    invalid_numeric = []
    for row in rows:
      try:
        values = [float(row[name]) for name in numeric_fields]
      except (KeyError, TypeError, ValueError):
        invalid_numeric.append(row); continue
      if not np.all(np.isfinite(values)):
        invalid_numeric.append(row)
    if invalid_numeric:
      issues.append(Issue("RESULT_NUMERIC", "fatal", f"{len(invalid_numeric)} result rows have missing/non-finite values"))
    wrong_normalization = [
      row for row in rows
      if row.get("run_id") != contract.run_id
      or row.get("normalization_id") != contract.normalization_id
      or float(row.get("normalization_offset", "nan"))
      != contract.normalization_offsets.get(row.get("dataset"), float("nan"))
      or (row.get("normalization_clip", "").lower() == "true")
      != contract.normalization_clip
    ] if contract is not None else []
    if wrong_normalization:
      issues.append(Issue(
        "RESULT_NORMALIZATION", "fatal",
        f"{len(wrong_normalization)} result rows have the wrong normalization identity",
      ))
    if splits:
      declared_test = {
        (row["dataset"], row["subject_id"]) for row in splits
        if row["split"] == "test_locked"
      }
      observed_test = {
        (row["dataset"], row["subject_id"]) for row in rows
        if row.get("split") == "test_locked"
      }
      missing_subjects = sorted(declared_test - observed_test)
      extra_subjects = sorted(observed_test - declared_test)
      if missing_subjects or extra_subjects:
        issues.append(Issue(
          "TEST_COVERAGE", "fatal",
          f"missing declared test subjects={missing_subjects[:5]} (n={len(missing_subjects)}); "
          f"undeclared test subjects={extra_subjects[:5]} (n={len(extra_subjects)})",
        ))
    primary = (
      "dataset", "subject_id", "slice_id", "realization_id", "count_level", "method"
    )
    keys = [tuple(row.get(name, "") for name in primary) for row in rows]
    duplicates = len(keys) - len(set(keys))
    if duplicates:
      issues.append(Issue("RESULT_DUPLICATES", "fatal", f"{duplicates} duplicate per-case keys"))
    grouped: dict[tuple[str, ...], set[str]] = {}
    for row in rows:
      case = tuple(row.get(name, "") for name in primary[:-1])
      grouped.setdefault(case, set()).add(row.get("method", ""))
    by_dataset = cfg["evaluation"].get("required_methods_by_dataset", {})
    incomplete = []
    for case, methods in grouped.items():
      expected_methods = set(by_dataset.get(case[0], cfg["evaluation"]["required_methods"]))
      if methods != expected_methods:
        incomplete.append((case, {"missing": sorted(expected_methods - methods), "extra": sorted(methods - expected_methods)}))
    if incomplete:
      issues.append(Issue(
        "RESULT_METHOD_COVERAGE", "fatal",
        f"{len(incomplete)} cases do not contain exactly the required methods; first={incomplete[0]}",
      ))
    zero_deviance = [row for row in rows if row.get("poisson_deviance") in {"0", "0.0", "0.000000"}]
    if zero_deviance:
      issues.append(Issue(
        "ZERO_DEVIANCE", "fatal",
        f"{len(zero_deviance)} rows report exactly zero Poisson deviance",
      ))
    checkpoint_mismatches = [
      row for row in rows
      if row.get("method", "").startswith("SAM")
      and row.get("checkpoint_sha256")
      != cfg["medsam2"]["checkpoint_sha256"].get(row.get("dataset"))
    ]
    if checkpoint_mismatches:
      issues.append(Issue(
        "RESULT_CHECKPOINT", "fatal",
        f"{len(checkpoint_mismatches)} SAM result rows have the wrong checkpoint SHA",
      ))
    if cache is not None:
      sam_cases = {
        BoxKey(row["dataset"], row["subject_id"], row["slice_id"], int(row["count_level"]))
        for row in rows if row.get("method", "").startswith("SAM")
      }
      missing = sorted(key for key in sam_cases if cache.get(key) is None)
      if missing:
        issues.append(Issue(
          "BOX_COVERAGE", "fatal",
          f"box cache misses {len(missing)}/{len(sam_cases)} evaluated SAM cases; first={missing[0]}",
        ))
  else:
    issues.append(Issue("RESULTS_MISSING", "fatal", f"missing {metrics_path}"))

  sweep_path = root / str(cfg["outputs"]["hyperparameter_sweep"])
  if sweep_path.is_file():
    try:
      verify_csv_artifact(sweep_path)
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as error:
      issues.append(Issue("SWEEP_ARTIFACT", "fatal", str(error)))
    sweep = _read_csv(sweep_path)
    wrong_run = [row for row in sweep if row.get("run_id") != run_id]
    if wrong_run:
      issues.append(Issue("SWEEP_RUN_ID", "fatal", f"{len(wrong_run)} sweep rows have wrong run_id"))
    wrong_split = [row for row in sweep if row.get("split") != "validation"]
    if wrong_split:
      issues.append(Issue("SWEEP_LEAKAGE", "fatal", f"{len(wrong_split)} sweep rows are not validation"))
    if splits:
      validation_subjects = {
        (row["dataset"], row["subject_id"]) for row in splits if row["split"] == "validation"
      }
      sweep_subjects = {(row["dataset"], row["subject_id"]) for row in sweep}
      if sweep_subjects != validation_subjects:
        issues.append(Issue(
          "SWEEP_SUBJECT_COVERAGE", "fatal",
          f"sweep subjects differ from validation assignment: missing={sorted(validation_subjects - sweep_subjects)[:5]}, "
          f"extra={sorted(sweep_subjects - validation_subjects)[:5]}",
        ))
    axes = {
      "sigma": {row.get("sigma") for row in sweep},
      "radius": {row.get("radius") for row in sweep},
      "iterations": {row.get("iterations") for row in sweep},
      "alpha": {row.get("alpha") for row in sweep},
    }
    for name, values in axes.items():
      if len(values - {"", None}) < 2:
        issues.append(Issue("DEGENERATE_SWEEP", "fatal", f"hyperparameter axis {name} has <2 values"))
  else:
    issues.append(Issue("SWEEP_MISSING", "fatal", f"missing {sweep_path}"))

  frozen_path = root / str(cfg["outputs"]["frozen_hyperparameters"])
  if not frozen_path.is_file():
    issues.append(Issue("FROZEN_HYPERPARAMETERS_MISSING", "fatal", f"missing {frozen_path}"))
  else:
    try:
      verify_csv_artifact(frozen_path)
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as error:
      issues.append(Issue("FROZEN_HYPERPARAMETERS_ARTIFACT", "fatal", str(error)))
    frozen_rows = _read_csv(frozen_path)
    if any(row.get("run_id") != run_id for row in frozen_rows):
      issues.append(Issue("FROZEN_RUN_ID", "fatal", "frozen hyperparameters have wrong run_id"))
    frozen_keys = {(row["dataset"], int(row["count_level"]), row["method"]) for row in frozen_rows}
    expected_frozen = set()
    for dataset in ("brainweb", "hecktor"):
      methods = [
        "MLEM", "OSEM", "Bowsher-KEM", "SelfKEM",
        "SAM-KEM-MedSAM2-Qwen3-full",
      ]
      methods.append("MR-KEM" if dataset == "brainweb" else "CT-KEM")
      for count in cfg["acquisition"]["count_levels"]:
        expected_frozen.update((dataset, int(count), method) for method in methods)
    if frozen_keys != expected_frozen:
      issues.append(Issue(
        "FROZEN_HYPERPARAMETERS_COVERAGE", "fatal",
        f"frozen keys mismatch: missing={sorted(expected_frozen - frozen_keys)[:5]}, extra={sorted(frozen_keys - expected_frozen)[:5]}",
      ))

  return issues


def audit_exit_code(issues: list[Issue]) -> int:
  return 1 if any(issue.severity == "fatal" for issue in issues) else 0
