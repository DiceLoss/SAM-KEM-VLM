"""Construction of the one canonical discrete Radon operator."""

from __future__ import annotations

import sys
import hashlib
from pathlib import Path

from .operators import CallableOperator, assert_adjoint


def build_discrete_radon(
  *, repo_root: Path, size: int, angles: int,
  operator_id: str | None = None,
  expected_implementation_sha256: str | None = None,
) -> CallableOperator:
  original = repo_root / "code" / "src" / "original"
  implementation = original / "discrete_radon.py"
  if expected_implementation_sha256 is not None:
    actual = hashlib.sha256(implementation.read_bytes()).hexdigest()
    if actual != expected_implementation_sha256:
      raise ValueError(f"forward-operator implementation SHA mismatch: {actual}")
  sys.path.insert(0, str(original))
  from discrete_radon import build_pair

  forward, adjoint, _ = build_pair(size=size, n_angles=angles)
  operator = CallableOperator(
    image_shape=(size, size),
    projection_shape=(angles, size),
    operator_id=operator_id or f"pixel_driven_parallel_beam_circular_fov_size{size}_n{angles}",
    forward_fn=forward,
    adjoint_fn=adjoint,
  )
  assert_adjoint(operator, tolerance=1e-5)
  return operator
