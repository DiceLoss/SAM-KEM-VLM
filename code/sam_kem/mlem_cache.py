"""Provenance-bound construction of the canonical MLEM-10 input."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .config import load_contract
from .noise import array_sha256
from .operator_factory import build_discrete_radon
from .reconstruction import mlem


def write_mlem_input(
  root: Path, config_path: Path, sinogram_path: Path, output_path: Path,
  *, projection_scale: float,
) -> Path:
  cfg, contract = load_contract(config_path)
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
  operator = build_discrete_radon(
    repo_root=root, size=int(cfg["forward_model"]["image_size"]),
    angles=contract.angles, operator_id=contract.operator_id,
    expected_implementation_sha256=cfg["forward_model"]["implementation_sha256"],
  )
  gain = np.full_like(measured, projection_scale, dtype=np.float64)
  iterations = contract.prompt_mlem_iterations
  image = mlem(
    measured, operator, iterations=iterations, multiplicative=gain,
  )[iterations]
  metadata_path = output_path.with_suffix(".json")
  if output_path.exists() or metadata_path.exists():
    raise FileExistsError(f"refusing to overwrite MLEM input: {output_path}")
  output_path.parent.mkdir(parents=True, exist_ok=True)
  np.save(output_path, image)
  metadata_path.write_text(json.dumps({
    "schema_version": 1,
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
  }, indent=2), encoding="utf-8")
  return output_path
