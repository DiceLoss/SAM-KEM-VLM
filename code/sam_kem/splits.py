"""Subject-level split loading and leakage checks."""

from __future__ import annotations

import csv
from collections import Counter, defaultdict
from pathlib import Path

from .artifacts import file_sha256


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


def load_configured_subject_splits(root: Path, cfg: dict) -> list[dict[str, str]]:
  """Load a SHA-pinned split and reject previously exposed locked-test rows."""
  split_path = root / str(cfg["data"]["split_file"])
  expected_sha = cfg["data"].get("split_sha256")
  if not isinstance(expected_sha, str) or len(expected_sha) != 64:
    raise ValueError("data.split_sha256 must pin the clean subject split")
  if file_sha256(split_path) != expected_sha:
    raise ValueError(f"configured subject-split SHA mismatch: {split_path}")
  rows = load_subject_splits(split_path)
  required = {
    "split_id", "split_locked_utc", "source_inventory_sha256",
    "split_script_sha256", "previous_exposure_status",
  }
  missing = required - set(rows[0])
  if missing:
    raise ValueError(f"clean subject split lacks provenance columns: {sorted(missing)}")
  expected_id = str(cfg["data"].get("split_id", ""))
  if not expected_id or any(row["split_id"] != expected_id for row in rows):
    raise ValueError("subject split_id differs from the configured clean split identity")
  identities = [(row["dataset"], row["subject_id"]) for row in rows]
  if len(identities) != len(set(identities)):
    raise ValueError("clean split must contain exactly one row per dataset/subject")
  required_exposure = str(cfg["data"]["locked_test_required_exposure_status"])
  exposed = [
    (row["dataset"], row["subject_id"], row["previous_exposure_status"])
    for row in rows
    if row["split"] == "test_locked"
    and row["previous_exposure_status"] != required_exposure
  ]
  if exposed:
    raise ValueError(f"locked-test subjects are not declared untouched: {exposed[:3]}")
  incomplete = [
    (row["dataset"], row["subject_id"])
    for row in rows
    if not row["split_locked_utc"]
    or any(len(row[field]) != 64 for field in ("source_inventory_sha256", "split_script_sha256"))
  ]
  if incomplete:
    raise ValueError(f"subject split lacks lock/SHA provenance: {incomplete[:3]}")
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
