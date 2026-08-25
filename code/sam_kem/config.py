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
  experiment_seed: int
  count_levels: tuple[int, ...]
  realizations: tuple[int, ...]
  scale_constants: dict[str, float]
  operator_id: str
  angles: int
  prompt_mlem_iterations: int

  @classmethod
  def from_mapping(cls, cfg: dict[str, Any]) -> "ScientificContract":
    constants = _require(cfg, "normalization.scale_constants")
    if not isinstance(constants, dict):
      raise ConfigurationError("normalization.scale_constants must be a mapping")
    parsed_constants: dict[str, float] = {}
    for dataset in ("brainweb", "hecktor"):
      raw = constants.get(dataset)
      if raw is None:
        raise ConfigurationError(
          f"normalization.scale_constants.{dataset} is unresolved; obtain the physical C value"
        )
      value = float(raw)
      if value <= 0:
        raise ConfigurationError(f"scale constant for {dataset} must be positive")
      parsed_constants[dataset] = value

    counts = tuple(int(x) for x in _require(cfg, "acquisition.count_levels"))
    realizations = tuple(int(x) for x in _require(cfg, "acquisition.realizations"))
    if len(set(counts)) != len(counts) or any(x <= 0 for x in counts):
      raise ConfigurationError("count levels must be unique positive integers")
    if len(set(realizations)) != len(realizations) or any(x < 0 for x in realizations):
      raise ConfigurationError("realization IDs must be unique non-negative integers")

    angles = int(_require(cfg, "forward_model.angles"))
    if angles != 180:
      raise ConfigurationError("canonical protocol requires 180 projection angles")
    if int(_require(cfg, "forward_model.image_size")) != 128:
      raise ConfigurationError("canonical protocol requires a 128x128 image grid")
    if _require(cfg, "acquisition.observation_model") != "poisson_scaled_noise_free_projection":
      raise ConfigurationError("canonical observation must be scaled Poisson projection counts")
    if _require(cfg, "acquisition.independent_realizations_across_count_levels") is not True:
      raise ConfigurationError("count levels and realizations must use independent random draws")
    if _require(cfg, "normalization.rule") != "divide_by_fixed_dataset_physical_constant":
      raise ConfigurationError("inputs/targets must use fixed dataset C scaling")
    forward = _require(cfg, "forward_model")
    expected_physics = {
      "geometry": "idealized_parallel_beam", "field_of_view": "inscribed_circle",
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
    ):
      raise ConfigurationError(
        "decoder checkpoints must use the declared atomic decoder-only/fixed-budget bundle"
      )
    return cls(
      experiment_seed=int(_require(cfg, "reproducibility.experiment_seed")),
      count_levels=counts,
      realizations=realizations,
      scale_constants=parsed_constants,
      operator_id=str(_require(cfg, "forward_model.operator_id")),
      angles=angles,
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
