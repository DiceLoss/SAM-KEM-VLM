"""Forward-operator interfaces and numerical consistency checks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Protocol

import numpy as np


class LinearOperator(Protocol):
  image_shape: tuple[int, int]
  projection_shape: tuple[int, int]
  operator_id: str

  def forward(self, image: np.ndarray) -> np.ndarray: ...
  def adjoint(self, projection: np.ndarray) -> np.ndarray: ...


@dataclass(frozen=True)
class CallableOperator:
  image_shape: tuple[int, int]
  projection_shape: tuple[int, int]
  operator_id: str
  forward_fn: Callable[[np.ndarray], np.ndarray]
  adjoint_fn: Callable[[np.ndarray], np.ndarray]

  def forward(self, image: np.ndarray) -> np.ndarray:
    arr = np.asarray(image, dtype=np.float64)
    if arr.shape != self.image_shape:
      raise ValueError(f"image shape {arr.shape} != {self.image_shape}")
    result = np.asarray(self.forward_fn(arr), dtype=np.float64)
    if result.shape != self.projection_shape:
      raise ValueError(f"projection shape {result.shape} != {self.projection_shape}")
    return result

  def adjoint(self, projection: np.ndarray) -> np.ndarray:
    arr = np.asarray(projection, dtype=np.float64)
    if arr.shape != self.projection_shape:
      raise ValueError(f"projection shape {arr.shape} != {self.projection_shape}")
    result = np.asarray(self.adjoint_fn(arr), dtype=np.float64)
    if result.shape != self.image_shape:
      raise ValueError(f"adjoint image shape {result.shape} != {self.image_shape}")
    return result


def relative_adjoint_error(operator: LinearOperator, *, seed: int = 0) -> float:
  """Return |<Ax,y>-<x,A^Ty>| normalized by both inner products."""
  rng = np.random.default_rng(seed)
  x = rng.normal(size=operator.image_shape)
  y = rng.normal(size=operator.projection_shape)
  lhs = float(np.vdot(operator.forward(x), y).real)
  rhs = float(np.vdot(x, operator.adjoint(y)).real)
  return abs(lhs - rhs) / max(abs(lhs), abs(rhs), 1.0)


def assert_adjoint(operator: LinearOperator, *, tolerance: float = 1e-6) -> None:
  error = relative_adjoint_error(operator)
  if not np.isfinite(error) or error > tolerance:
    raise ValueError(
      f"operator {operator.operator_id!r} failed adjoint test: "
      f"relative error={error:.3e}, tolerance={tolerance:.3e}"
    )

