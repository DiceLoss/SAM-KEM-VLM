"""Subject-level split loading and leakage checks."""

from __future__ import annotations

import csv
from collections import Counter, defaultdict
from pathlib import Path


ALLOWED_SPLITS = {"train", "validation", "test_locked"}


def load_subject_splits(path: Path) -> list[dict[str, str]]:
  with path.open("r", encoding="utf-8", newline="") as handle:
    rows = list(csv.DictReader(handle))
  required = {"dataset", "subject_id", "split"}
  if not rows or not required.issubset(rows[0]):
    raise ValueError(f"split file must contain {sorted(required)}")
  assignments: dict[tuple[str, str], str] = {}
  for row in rows:
    split = row["split"]
    if split not in ALLOWED_SPLITS:
      raise ValueError(f"unknown split {split!r}")
    key = (row["dataset"], row["subject_id"])
    previous = assignments.setdefault(key, split)
    if previous != split:
      raise ValueError(f"subject leakage across splits: {key} in {previous} and {split}")
  return rows


def split_counts(rows: list[dict[str, str]]) -> dict[str, dict[str, int]]:
  grouped: dict[str, Counter] = defaultdict(Counter)
  seen: set[tuple[str, str]] = set()
  for row in rows:
    key = (row["dataset"], row["subject_id"])
    if key not in seen:
      grouped[row["dataset"]][row["split"]] += 1
      seen.add(key)
  return {dataset: dict(counts) for dataset, counts in grouped.items()}

