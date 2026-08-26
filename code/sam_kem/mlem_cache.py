"""Provenance-bound construction of the canonical MLEM-10 input."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .config import load_contract
from .artifacts import atomic_save_numpy, atomic_write_json
from .noise import array_sha256
from .operator_factory import build_discrete_radon
from .reconstruction import mlem


def write_mlem_input(
  root: Path, config_path: Path, sinogram_path: Path, output_path: Path,
  *, projection_scale: float,
  context=None,
) -> Path:
  if context is None:
    cfg, contract = load_contract(config_path)
    operator = build_discrete_radon(
      repo_root=root, size=int(cfg["forward_model"]["image_size"]),
      angles=contract.angles, operator_id=contract.operator_id,
      expected_implementation_sha256=cfg["forward_model"]["implementation_sha256"],
    )
  else:
    cfg, contract, operator = context.cfg, context.contract, context.operator
  measured = np.load(sinogram_path)
  simulation_sidecar = sinogram_path.with_suffix(".json")
  if not simulation_sidecar.is_file():
    raise FileNotFoundError(f"sinogram provenance sidecar missing: {simulation_sidecar}")
  simulation = json.loads(simulation_sidecar.read_text(encoding="utf-8"))
  if simulation.get("sinogram_sha256") != array_sha256(measured):
    raise ValueError("sinogram differs from its simulation sidecar")
  if not np.isclose(
    float(simulation.get("projection_scale", -1)), projection_scale, rtol=1e-12,
  ):
    raise ValueError("requested projection scale differs from simulation sidecar")
  gain = np.full_like(measured, projection_scale, dtype=np.float64)
  iterations = contract.prompt_mlem_iterations
  image = mlem(
    measured, operator, iterations=iterations, multiplicative=gain,
  )[iterations]
  metadata_path = output_path.with_suffix(".json")
  if output_path.exists() or metadata_path.exists():
    raise FileExistsError(f"refusing to overwrite MLEM input: {output_path}")
  atomic_save_numpy(output_path, image)
  metadata = {
    "schema_version": 2, "artifact_kind": "mlem_prompt_input",
    "identity": {
      name: simulation[name] for name in (
        "dataset", "subject_id", "slice_id", "count_level", "realization_id",
      )
    },
    "method": "MLEM", "iterations": iterations,
    "initialization": "uniform_ones", "additive_background": "none",
    "projection_scale": projection_scale,
    "operator_id": contract.operator_id,
    "operator_implementation_sha256": cfg["forward_model"]["implementation_sha256"],
    "sinogram_sha256": array_sha256(measured),
    "mlem10_sha256": array_sha256(image),
  }
  try:
    atomic_write_json(metadata_path, metadata)
  except BaseException:
    output_path.unlink(missing_ok=True)
    raise
  return output_path
