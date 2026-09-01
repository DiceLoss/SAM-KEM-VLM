from pathlib import Path

import numpy as np

from sam_kem.experiments import Case
from sam_kem.operators import CallableOperator
from sam_kem.protocols import Protocol
from sam_kem.sweep_evaluation import evaluate_validation_case, parse_sweep_grid


def test_shared_sweep_evaluator_emits_each_declared_grid_point(tmp_path):
  shape = (16, 16)
  target = np.linspace(0.1, 1.0, num=256, dtype=np.float32).reshape(shape)
  measured = target.copy()
  anatomy = np.flipud(target).copy()
  prior = np.clip(target * 1.1 - 0.05, -0.1, 1.2).astype(np.float32)
  paths = {}
  for name, value in (
    ("target", target), ("sinogram", measured), ("anatomy", anatomy), ("prior", prior),
  ):
    path = tmp_path / f"{name}.npy"
    np.save(path, value, allow_pickle=False)
    paths[name] = path
  case = Case(
    run_id="run", split_id="split", split_sha256="a" * 64,
    dataset="brainweb", subject_id="subject04", slice_id="z038",
    realization_id=0, count_level=200000, split="validation", seed=1,
    scale_constant_C=1.0, normalization_id="fixed", normalization_offset=0.0,
    normalization_clip=False, operator_id="identity", projection_scale=1.0,
    target_path=paths["target"], sinogram_path=paths["sinogram"],
    mlem10_input_path=paths["target"], anatomy_path=paths["anatomy"],
    sam_prior_path=paths["prior"], mask_path=None,
  )
  operator = CallableOperator(
    image_shape=shape, projection_shape=shape, operator_id="identity",
    forward_fn=lambda value: value, adjoint_fn=lambda value: value,
  )
  protocol = Protocol(
    name="small", kind="hyperparameter_sweep", split="validation",
    methods=(
      "MLEM", "OSEM", "Bowsher-KEM", "SelfKEM", "MR-KEM", "CT-KEM",
      "SAM-KEM-MedSAM2-Qwen3-full",
    ), output=Path("unused"), raw={"search_space": {
      "sigma": [0.1], "radius": [1], "iterations": [1], "alpha": [0.0, 0.5],
      "bowsher_neighbours": [1], "bowsher_radius": [1],
      "mlem_iterations": [1], "osem_cycles": [1],
    }},
  )
  cfg = {
    "normalization": {"expected_target_range": [0.0, 1.0]},
    "reconstruction": {"prompt_mlem_iterations": 1, "osem_subsets": 1},
    "evaluation": {"nonnegative_tolerance": 1e-7},
  }
  records = evaluate_validation_case(
    case, operator, np.ones(shape, dtype=bool), cfg, "run", protocol.methods,
    parse_sweep_grid(protocol),
  )
  assert len(records) == 7
  assert [record["method"] for record in records] == [
    "MLEM", "OSEM", "Bowsher-KEM", "SelfKEM", "MR-KEM",
    "SAM-KEM-MedSAM2-Qwen3-full", "SAM-KEM-MedSAM2-Qwen3-full",
  ]
  assert all(np.isfinite(float(record["nrmse"])) for record in records)
  assert all(np.isfinite(float(record["ssim"])) for record in records)
