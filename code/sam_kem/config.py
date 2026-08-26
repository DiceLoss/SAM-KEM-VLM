"""Strict configuration loading and scientific-contract validation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


class ConfigurationError(ValueError):
  pass


class _UniqueKeyLoader(yaml.SafeLoader):
  """YAML loader that rejects duplicate mapping keys."""


def _construct_mapping(loader: _UniqueKeyLoader, node: yaml.MappingNode, deep: bool = False) -> dict:
  mapping: dict[Any, Any] = {}
  for key_node, value_node in node.value:
    key = loader.construct_object(key_node, deep=deep)
    if key in mapping:
      mark = key_node.start_mark
      raise ConfigurationError(
        f"duplicate YAML key {key!r} at line {mark.line + 1}, column {mark.column + 1}"
      )
    mapping[key] = loader.construct_object(value_node, deep=deep)
  return mapping


_UniqueKeyLoader.add_constructor(
  yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
  _construct_mapping,
)


def load_yaml_strict(path: Path) -> dict[str, Any]:
  with path.open("r", encoding="utf-8") as handle:
    value = yaml.load(handle, Loader=_UniqueKeyLoader)
  if not isinstance(value, dict):
    raise ConfigurationError(f"top-level YAML value must be a mapping: {path}")
  return value


def _require(mapping: dict[str, Any], dotted: str) -> Any:
  value: Any = mapping
  for part in dotted.split("."):
    if not isinstance(value, dict) or part not in value:
      raise ConfigurationError(f"missing required configuration field: {dotted}")
    value = value[part]
  return value


@dataclass(frozen=True)
class ScientificContract:
  run_id: str
  split_id: str
  split_sha256: str
  experiment_seed: int
  count_levels: tuple[int, ...]
  realizations: tuple[int, ...]
  scale_constants: dict[str, float]
  normalization_offsets: dict[str, float]
  normalization_id: str
  normalization_clip: bool
  operator_id: str
  angles: int
  detector_bins: int
  prompt_mlem_iterations: int

  @classmethod
  def from_mapping(cls, cfg: dict[str, Any]) -> "ScientificContract":
    run_id = str(_require(cfg, "run_id"))
    if not run_id or not all(character.isalnum() or character in "_-" for character in run_id):
      raise ConfigurationError("run_id must contain only letters, digits, '_' or '-'")
    split_id = str(_require(cfg, "data.split_id"))
    split_sha = _require(cfg, "data.split_sha256")
    if not split_id or not isinstance(split_sha, str) or len(split_sha) != 64:
      raise ConfigurationError("clean split_id and 64-character split_sha256 are required")
    constants = _require(cfg, "normalization.scale_constants")
    if not isinstance(constants, dict):
      raise ConfigurationError("normalization.scale_constants must be a mapping")
    parsed_constants: dict[str, float] = {}
    parsed_offsets: dict[str, float] = {}
    constant_sources = _require(cfg, "normalization.scale_constant_sources")
    offsets = _require(cfg, "normalization.offsets")
    if not isinstance(constant_sources, dict) or not isinstance(offsets, dict):
      raise ConfigurationError("normalization scale sources and offsets must be mappings")
    normalization_id = str(_require(cfg, "normalization.normalization_id"))
    if not normalization_id or normalization_id == "unresolved":
      raise ConfigurationError("normalization.normalization_id must be explicit")
    for dataset in ("brainweb", "hecktor"):
      raw = constants.get(dataset)
      if raw is None:
        raise ConfigurationError(
          f"normalization.scale_constants.{dataset} is unresolved; obtain the physical C value"
        )
      value = float(raw)
      if value <= 0:
        raise ConfigurationError(f"scale constant for {dataset} must be positive")
      if not constant_sources.get(dataset):
        raise ConfigurationError(
          f"normalization.scale_constant_sources.{dataset} must identify the physical provenance"
        )
      parsed_constants[dataset] = value
      parsed_offsets[dataset] = float(offsets.get(dataset, 0.0))

    counts = tuple(int(x) for x in _require(cfg, "acquisition.count_levels"))
    realizations = tuple(int(x) for x in _require(cfg, "acquisition.realizations"))
    if len(set(counts)) != len(counts) or any(x <= 0 for x in counts):
      raise ConfigurationError("count levels must be unique positive integers")
    if len(set(realizations)) != len(realizations) or any(x < 0 for x in realizations):
      raise ConfigurationError("realization IDs must be unique non-negative integers")
    training_counts = tuple(int(x) for x in _require(cfg, "medsam2.training_count_levels"))
    if training_counts != counts:
      raise ConfigurationError("MedSAM2 joint-training count levels must equal acquisition counts")

    angles = int(_require(cfg, "forward_model.angles"))
    if angles != 180:
      raise ConfigurationError("canonical protocol requires 180 projection angles")
    if int(_require(cfg, "forward_model.image_size")) != 128:
      raise ConfigurationError("canonical protocol requires a 128x128 image grid")
    detector_bins = int(_require(cfg, "forward_model.detector_bins"))
    if detector_bins != 182:
      raise ConfigurationError(
        "canonical full-square protocol requires 182 detector bins for a 128x128 image"
      )
    if _require(cfg, "acquisition.observation_model") != "poisson_scaled_noise_free_projection":
      raise ConfigurationError("canonical observation must be scaled Poisson projection counts")
    if _require(cfg, "acquisition.independent_realizations_across_count_levels") is not True:
      raise ConfigurationError("count levels and realizations must use independent random draws")
    normalization_rule = _require(cfg, "normalization.rule")
    normalization_clip = _require(cfg, "normalization.clip_to_expected_target_range")
    if normalization_rule == "divide_by_fixed_dataset_physical_constant":
      if any(value != 0.0 for value in parsed_offsets.values()) or normalization_clip is not False:
        raise ConfigurationError("physical fixed-C scaling requires zero offset and no clipping")
    elif normalization_rule == "training_global_minmax_affine_clip01":
      if normalization_clip is not True:
        raise ConfigurationError("global min-max sensitivity requires explicit clipping")
    else:
      raise ConfigurationError(f"unsupported normalization rule: {normalization_rule}")
    forward = _require(cfg, "forward_model")
    expected_physics = {
      "geometry": "idealized_parallel_beam", "field_of_view": "full_square_image_grid",
      "axial_model": "independent_2d_ring_positions_no_axial_coincidence_model",
      "attenuation": "none", "scatter": "none", "randoms": "none",
      "detector_normalization": "none", "point_spread_function": "none",
      "time_of_flight": "none", "dead_time": "none",
    }
    mismatched = {
      name: (forward.get(name), expected)
      for name, expected in expected_physics.items() if forward.get(name) != expected
    }
    if mismatched:
      raise ConfigurationError(f"undeclared or changed PET forward physics: {mismatched}")
    if forward.get("same_operator_for_simulation_reconstruction_and_deviance") is not True:
      raise ConfigurationError("one matched operator is required throughout the experiment")
    target_range = tuple(float(x) for x in _require(cfg, "normalization.expected_target_range"))
    psnr_range = float(_require(cfg, "evaluation.psnr_data_range_after_fixed_C_scaling"))
    if target_range != (0.0, 1.0) or psnr_range != 1.0:
      raise ConfigurationError("fixed-C targets and PSNR must share the declared [0,1] scale")
    if _require(cfg, "reconstruction.bowsher_method_id") != "Bowsher-KEM":
      raise ConfigurationError("the anatomy-neighbour coefficient method must be named Bowsher-KEM")
    if (
      _require(cfg, "evaluation.locked_test_access_policy")
      != "only_after_validation_freeze_no_test_feedback"
    ):
      raise ConfigurationError(
        "locked-test access must occur only after validation freeze and without test feedback"
      )
    if (
      _require(cfg, "evaluation.region") != "operator_full_square_support"
      or _require(cfg, "evaluation.metric_support_source")
      != "positive_operator_sensitivity"
    ):
      raise ConfigurationError("metrics must use the anatomy-independent PET operator support")
    if _require(cfg, "medsam2.raw_prior_kernel_transform") != "per_prior_minmax":
      raise ConfigurationError("SAM kernel features require per-prior min-max")
    if (
      _require(cfg, "medsam2.physical_prior_output_constraint")
      != "clip_to_fixed_target_range_without_sigmoid_or_minmax"
    ):
      raise ConfigurationError("physical prior output must use the declared fixed-range clipping")
    artifact_policy = _require(cfg, "outputs.artifact_policy")
    if (
      artifact_policy.get("immutable") is not True
      or artifact_policy.get("atomic_write") is not True
      or artifact_policy.get("csv_manifest_required") is not True
      or artifact_policy.get("success_marker_required") is not True
      or artifact_policy.get("checksum") != "sha256"
      or artifact_policy.get("partial_run_eligible_for_paper") is not False
    ):
      raise ConfigurationError("result artifact policy must be immutable, atomic and SHA-bound")
    checkpoint_cfg = _require(cfg, "medsam2")
    training_cfg = _require(cfg, "medsam2.training")
    if (
      checkpoint_cfg.get("checkpoint_format")
      != "mask_decoder_only_inference_plus_separate_resumable_training_state"
      or checkpoint_cfg.get("checkpoint_selection_policy")
      != "fixed_budget_final_checkpoint_no_best_claim"
      or checkpoint_cfg.get("checkpoint_atomic_write") is not True
      or training_cfg.get("checkpoint_every_epochs") != 1
      or training_cfg.get("save_optimizer_and_rng_state") is not True
      or training_cfg.get("sampling_unit") != "count_then_subject_then_slice_realization"
    ):
      raise ConfigurationError(
        "decoder checkpoints must use the declared atomic decoder-only/fixed-budget bundle"
      )
    return cls(
      run_id=run_id,
      split_id=split_id,
      split_sha256=split_sha,
      experiment_seed=int(_require(cfg, "reproducibility.experiment_seed")),
      count_levels=counts,
      realizations=realizations,
      scale_constants=parsed_constants,
      normalization_offsets=parsed_offsets,
      normalization_id=normalization_id,
      normalization_clip=bool(normalization_clip),
      operator_id=str(_require(cfg, "forward_model.operator_id")),
      angles=angles,
      detector_bins=detector_bins,
      prompt_mlem_iterations=int(_require(cfg, "reconstruction.prompt_mlem_iterations")),
    )


def load_contract(path: Path) -> tuple[dict[str, Any], ScientificContract]:
  cfg = load_yaml_strict(path)
  return cfg, ScientificContract.from_mapping(cfg)


def require_locked_execution(cfg: dict[str, Any]) -> None:
  if cfg.get("locked") is not True or cfg.get("status") != "locked_for_execution":
    raise ConfigurationError(
      "evaluation requires locked=true and status=locked_for_execution"
    )


def require_test_release(root: Path, cfg: dict[str, Any]) -> None:
  """Require a completed validation sweep before any locked-test data access."""
  import csv
  from .artifacts import verify_csv_artifact, verify_success_marker

  require_locked_execution(cfg)
  frozen = root / str(_require(cfg, "outputs.frozen_hyperparameters"))
  sweep = root / str(_require(cfg, "outputs.hyperparameter_sweep"))
  try:
    frozen_record = verify_csv_artifact(frozen)
    sweep_record = verify_csv_artifact(sweep)
    success = verify_success_marker(sweep.parent)
  except (OSError, ValueError, KeyError) as error:
    raise ConfigurationError(
      "locked test is sealed until the validation sweep and frozen parameters are complete"
    ) from error
  if (
    frozen_record.get("schema_name") != "frozen_hyperparameters_v2"
    or sweep_record.get("schema_name") != "validation_hyperparameter_sweep_v2"
    or frozen_record.get("split") != "validation"
    or sweep_record.get("split") != "validation"
  ):
    raise ConfigurationError("locked-test release artifacts have unexpected schemas")
  completed_paths = {
    (sweep.parent / str(item["path"])).resolve()
    for item in success.get("artifacts", [])
  }
  required_paths = {
    sweep.resolve(), sweep.with_suffix(sweep.suffix + ".manifest.json").resolve(),
    frozen.resolve(), frozen.with_suffix(frozen.suffix + ".manifest.json").resolve(),
  }
  if not required_paths <= completed_paths:
    raise ConfigurationError(
      "validation completion marker does not bind the sweep and frozen parameters"
    )
  with frozen.open("r", encoding="utf-8", newline="") as handle:
    rows = list(csv.DictReader(handle))
  if not rows or any(row.get("run_id") != cfg["run_id"] for row in rows):
    raise ConfigurationError("frozen parameters do not belong to the configured run_id")
  with sweep.open("r", encoding="utf-8", newline="") as handle:
    sweep_rows = list(csv.DictReader(handle))
  if (
    not sweep_rows
    or any(row.get("run_id") != cfg["run_id"] for row in sweep_rows)
    or any(row.get("split") != "validation" for row in sweep_rows)
  ):
    raise ConfigurationError("validation sweep does not belong to the configured run_id/split")
