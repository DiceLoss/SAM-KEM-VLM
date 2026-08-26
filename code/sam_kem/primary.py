"""Derive the headline table from the unique locked comparison artifact."""

from __future__ import annotations

import csv
from pathlib import Path

from .artifacts import (
  atomic_csv_writer, file_sha256, finalize_csv, verify_csv_artifact,
  verify_success_marker, write_success_marker,
)
from .config import load_contract, require_locked_execution
from .protocols import load_protocol
from .runmeta import write_run_metadata


def derive_primary(root: Path, config_path: Path, protocol_path: Path) -> Path:
  cfg, _ = load_contract(config_path)
  require_locked_execution(cfg)
  protocol = load_protocol(protocol_path)
  if protocol.kind != "primary" or protocol.split != "test_locked":
    raise ValueError("primary derivation requires the locked primary protocol")
  comparison = root / str(cfg["outputs"]["per_case_metrics"])
  if comparison != root / str(protocol.raw.get("source", "")):
    raise ValueError("primary protocol source differs from configured comparison artifact")
  comparison_record = verify_csv_artifact(comparison)
  if comparison_record.get("schema_name") != "per_case_reconstruction_metrics_v2":
    raise ValueError("primary source must be the canonical comparison artifact")
  verify_success_marker(comparison.parent)
  with comparison.open("r", encoding="utf-8", newline="") as handle:
    reader = csv.DictReader(handle)
    fields = list(reader.fieldnames or ()) + ["source_comparison_sha256"]
    rows = [
      {
        **row, "protocol": protocol.name,
        "source_comparison_sha256": file_sha256(comparison),
      }
      for row in reader
      if row["split"] == protocol.split
      and row["dataset"] in set(protocol.raw["datasets"])
      and int(row["count_level"]) in {int(value) for value in protocol.raw["count_levels"]}
      and int(row["realization_id"]) in {int(value) for value in protocol.raw["realizations"]}
      and row["method"] in set(protocol.methods)
    ]
  if not rows:
    raise ValueError("comparison artifact contains no rows for the primary protocol")
  expected_datasets = set(protocol.raw["datasets"])
  if {row["dataset"] for row in rows} != expected_datasets:
    raise ValueError("primary derivation does not cover every declared dataset")
  output_dir = root / protocol.output
  output_dir.mkdir(parents=True, exist_ok=True)
  run_metadata = write_run_metadata(
    output_dir, root=root, config_path=config_path, protocol_path=protocol_path,
  )
  output = output_dir / "per_case_primary.csv"
  with atomic_csv_writer(output, fields) as writer:
    writer.writerows(rows)
  manifest = finalize_csv(
    output, schema_name="derived_primary_metrics_v2",
    primary_key=[
      "run_id", "protocol", "dataset", "subject_id", "slice_id", "realization_id",
      "count_level", "method",
    ], split="test_locked",
  )
  write_success_marker(output_dir, [
    run_metadata, output_dir / "environment_freeze.txt", output, manifest,
  ])
  return output
