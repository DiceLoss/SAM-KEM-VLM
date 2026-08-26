"""Deterministic identifiers and independent per-case random streams."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass


@dataclass(frozen=True)
class CaseKey:
  dataset: str
  subject_id: str
  slice_id: str
  count_level: int
  realization_id: int

  def canonical(self) -> str:
    return "|".join(
      (
        self.dataset.strip().lower(),
        self.subject_id.strip(),
        self.slice_id.strip(),
        str(int(self.count_level)),
        str(int(self.realization_id)),
      )
    )


def case_seed(key: CaseKey, *, experiment_seed: int) -> int:
  """Map the complete case identity to a stable NumPy-compatible seed.

  Including dataset, subject, slice, count and realization prevents the
  correlated random streams produced by ``base_seed + realization``.
  """
  payload = f"sam-kem-v1|{int(experiment_seed)}|{key.canonical()}".encode("utf-8")
  digest = hashlib.sha256(payload).digest()
  return int.from_bytes(digest[:8], "big", signed=False)

