"""Validation of experiment-specific protocols."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import ConfigurationError, load_yaml_strict


EXPERIMENT_KINDS = {
  "training", "hyperparameter_sweep", "primary", "comparison",
  "ablation", "convergence", "cross_count",
  "runtime", "normalization_calibration",
  "stage7a_prompt_planning",
  "stage7a_v2_prompt_planning",
  "stage7a_v3_prompt_planning",
  "stage7a_v4_prompt_planning",
}


@dataclass(frozen=True)
class Protocol:
  name: str
  kind: str
  split: str
  methods: tuple[str, ...]
  output: Path
  raw: dict[str, Any]


def load_protocol(path: Path) -> Protocol:
  raw = load_yaml_strict(path)
  kind = str(raw.get("kind", ""))
  split = str(raw.get("split", ""))
  if kind not in EXPERIMENT_KINDS:
    raise ConfigurationError(f"unknown experiment kind: {kind!r}")
  if split not in {"train", "validation", "test_locked"}:
    raise ConfigurationError(f"unknown protocol split: {split!r}")
  methods = tuple(str(x) for x in raw.get("methods", ()))
  if not methods:
    raise ConfigurationError("protocol must list at least one method or variant")
  if kind in {"training", "hyperparameter_sweep", "normalization_calibration"} and split == "test_locked":
    raise ConfigurationError(f"{kind} is prohibited on test_locked")
  if kind == "normalization_calibration":
    if raw.get("fit_split") != "train" or raw.get("evaluation_split") != "validation":
      raise ConfigurationError("normalization must be fitted on train and examined on validation")
    if raw.get("locked_test_access") != "prohibited":
      raise ConfigurationError("normalization calibration must prohibit locked-test access")
  if kind == "hyperparameter_sweep":
    axes = raw.get("search_space", {})
    for axis in (
      "sigma", "radius", "iterations", "alpha",
      "bowsher_neighbours", "bowsher_radius",
      "mlem_iterations", "osem_cycles",
    ):
      values = axes.get(axis, []) if isinstance(axes, dict) else []
      if len(set(values)) < 2:
        raise ConfigurationError(f"sweep axis {axis} must contain at least two values")
  if kind == "ablation":
    if len(methods) != len(set(methods)):
      raise ConfigurationError("ablation variants must have unique names")
    if not raw.get("one_factor_at_a_time", False):
      raise ConfigurationError("ablation protocol must declare one_factor_at_a_time: true")
  if split == "test_locked" and not raw.get("requires_frozen_hyperparameters", False):
    raise ConfigurationError("test_locked protocol must require frozen validation hyperparameters")
  return Protocol(
    name=str(raw.get("name", path.stem)),
    kind=kind,
    split=split,
    methods=methods,
    output=Path(str(raw.get("output", ""))),
    raw=raw,
  )
