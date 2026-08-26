"""Deterministic subject-level split creation with an exposure-sealed test set."""

from __future__ import annotations

import csv
import hashlib
import random
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from .artifacts import atomic_csv_writer, file_sha256, finalize_csv
from .config import load_yaml_strict


INVENTORY_COLUMNS = (
  "dataset", "subject_id", "stratum", "previous_exposure_status",
)
UNTOUCHED = "never_used_for_training_validation_tuning_or_reporting"


def _stable_seed(seed: int, dataset: str, stage: str) -> int:
  digest = hashlib.sha256(f"{seed}|{dataset}|{stage}".encode("utf-8")).digest()
  return int.from_bytes(digest[:8], "big")


def _stratified_select(rows: list[dict[str, str]], count: int, *, seed: int) -> list[dict[str, str]]:
  if count < 0 or count > len(rows):
    raise ValueError(f"cannot select {count} subjects from {len(rows)} candidates")
  if count == 0:
    return []
  groups: dict[str, list[dict[str, str]]] = defaultdict(list)
  for row in rows:
    groups[row["stratum"] or "__all__"].append(row)
  total = len(rows)
  quotas = {name: count * len(group) / total for name, group in groups.items()}
  allocations = {name: min(len(groups[name]), int(quota)) for name, quota in quotas.items()}
  remainder = count - sum(allocations.values())
  priority = sorted(
    groups,
    key=lambda name: (-(quotas[name] - int(quotas[name])), name),
  )
  while remainder:
    progressed = False
    for name in priority:
      if allocations[name] < len(groups[name]):
        allocations[name] += 1
        remainder -= 1
        progressed = True
        if remainder == 0:
          break
    if not progressed:
      raise RuntimeError("stratified allocation could not satisfy the requested count")
  selected = []
  rng = random.Random(seed)
  for name in sorted(groups):
    candidates = sorted(groups[name], key=lambda row: row["subject_id"])
    rng.shuffle(candidates)
    selected.extend(candidates[:allocations[name]])
  return selected


def create_subject_split(
  root: Path, config_path: Path, inventory_path: Path, output_path: Path,
) -> Path:
  cfg = load_yaml_strict(config_path)
  with inventory_path.open("r", encoding="utf-8", newline="") as handle:
    reader = csv.DictReader(handle)
    missing = set(INVENTORY_COLUMNS) - set(reader.fieldnames or ())
    if missing:
      raise ValueError(f"subject inventory missing columns: {sorted(missing)}")
    inventory = list(reader)
  identities = [(row["dataset"], row["subject_id"]) for row in inventory]
  if not inventory or len(identities) != len(set(identities)):
    raise ValueError("subject inventory must contain one unique row per dataset/subject")
  if any(not row["previous_exposure_status"] for row in inventory):
    raise ValueError("every subject requires a previous_exposure_status declaration")
  expected = cfg["data"]["expected_subject_counts"]
  seed = int(cfg["reproducibility"]["experiment_seed"])
  assignments: dict[tuple[str, str], str] = {}
  for dataset, counts in expected.items():
    rows = [row for row in inventory if row["dataset"] == dataset]
    required_total = sum(int(counts[name]) for name in ("train", "validation", "test_locked"))
    if len(rows) != required_total:
      raise ValueError(f"{dataset} inventory has {len(rows)} subjects; expected {required_total}")
    untouched = [row for row in rows if row["previous_exposure_status"] == UNTOUCHED]
    test_rows = _stratified_select(
      untouched, int(counts["test_locked"]),
      seed=_stable_seed(seed, dataset, "test_locked"),
    )
    test_ids = {(row["dataset"], row["subject_id"]) for row in test_rows}
    remaining = [row for row in rows if (row["dataset"], row["subject_id"]) not in test_ids]
    validation_rows = _stratified_select(
      remaining, int(counts["validation"]),
      seed=_stable_seed(seed, dataset, "validation"),
    )
    validation_ids = {(row["dataset"], row["subject_id"]) for row in validation_rows}
    for row in rows:
      identity = (row["dataset"], row["subject_id"])
      assignments[identity] = (
        "test_locked" if identity in test_ids
        else "validation" if identity in validation_ids else "train"
      )
  unknown_datasets = sorted(set(row["dataset"] for row in inventory) - set(expected))
  if unknown_datasets:
    raise ValueError(f"inventory contains unconfigured datasets: {unknown_datasets}")
  locked_utc = datetime.now(timezone.utc).isoformat()
  inventory_sha = file_sha256(inventory_path)
  script_sha = file_sha256(Path(__file__))
  fields = [
    "dataset", "subject_id", "split", "split_id", "stratum", "split_locked_utc",
    "source_inventory_sha256", "split_script_sha256", "previous_exposure_status",
  ]
  with atomic_csv_writer(output_path, fields) as writer:
    for row in sorted(inventory, key=lambda item: (item["dataset"], item["subject_id"])):
      writer.writerow({
        "dataset": row["dataset"], "subject_id": row["subject_id"],
        "split": assignments[(row["dataset"], row["subject_id"])],
        "split_id": cfg["data"]["split_id"], "stratum": row["stratum"],
        "split_locked_utc": locked_utc,
        "source_inventory_sha256": inventory_sha,
        "split_script_sha256": script_sha,
        "previous_exposure_status": row["previous_exposure_status"],
      })
  finalize_csv(
    output_path, schema_name="subject_level_split_v1",
    primary_key=["dataset", "subject_id"],
  )
  return output_path
