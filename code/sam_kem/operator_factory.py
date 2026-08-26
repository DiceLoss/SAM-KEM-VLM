"""Construction of the one canonical discrete Radon operator."""

from __future__ import annotations

import hashlib
from pathlib import Path

from .operators import CallableOperator, assert_adjoint
from .pet_operator import build_pair, minimum_detector_bins


def build_discrete_radon(
  *, repo_root: Path, size: int, angles: int,
  detector_bins: int | None = None,
  operator_id: str | None = None,
  expected_implementation_sha256: str | None = None,
) -> CallableOperator:
  implementation = repo_root / "code" / "sam_kem" / "pet_operator.py"
  if expected_implementation_sha256 is not None:
    actual = hashlib.sha256(implementation.read_bytes()).hexdigest()
    if actual != expected_implementation_sha256:
      raise ValueError(f"forward-operator implementation SHA mismatch: {actual}")
  bins = minimum_detector_bins(size) if detector_bins is None else int(detector_bins)
  forward, adjoint = build_pair(size=size, angles=angles, detector_bins=bins)
  operator = CallableOperator(
    image_shape=(size, size),
    projection_shape=(angles, bins),
    operator_id=operator_id or (
      f"pixel_driven_parallel_beam_full_square_size{size}_n{angles}_d{bins}"
    ),
    forward_fn=forward,
    adjoint_fn=adjoint,
  )
  assert_adjoint(operator, tolerance=1e-5)
  return operator
