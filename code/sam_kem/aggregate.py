"""Duplicate-safe aggregation for cross-count reporting."""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import numpy as np

from .config import load_contract, require_locked_execution
from .protocols import load_protocol
from .runmeta import write_run_metadata
from .artifacts import atomic_csv_writer, finalize_csv, verify_csv_artifact, verify_success_marker, write_success_marker


def aggregate_cross_count(
  per_case_path: Path, output_path: Path, *, datasets: set[str] | None = None,
  count_levels: set[int] | None = None, methods: set[str] | None = None,
) -> Path:
  rows = list(csv.DictReader(per_case_path.open("r", encoding="utf-8", newline="")))
  run_ids = {row.get("run_id") for row in rows}
  if len(run_ids) != 1 or None in run_ids:
    raise ValueError(f"aggregation input mixes run IDs: {run_ids}")
  run_id = next(iter(run_ids))
  case_keys = [
    tuple(row[name] for name in ("run_id", "dataset", "subject_id", "slice_id", "realization_id", "count_level", "method"))
    for row in rows
  ]
  if len(case_keys) != len(set(case_keys)):
    raise ValueError("refusing to aggregate duplicate per-case keys")
  record = verify_csv_artifact(per_case_path)
  if record.get("schema_name") != "per_case_reconstruction_metrics_v2":
    raise ValueError("cross-count input must be canonical per-case reconstruction metrics")
  rows = [
    row for row in rows
    if (datasets is None or row["dataset"] in datasets)
    and (count_levels is None or int(row["count_level"]) in count_levels)
    and (methods is None or row["method"] in methods)
  ]
  if not rows:
    raise ValueError("cross-count filters selected no canonical per-case rows")
  groups: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
  for row in rows:
    groups[(row["dataset"], row["count_level"], row["method"])].append(row)
  output_path.parent.mkdir(parents=True, exist_ok=True)
  fields = [
      "run_id", "dataset", "count_level", "method", "n_subjects", "n_cases",
      "subject_mean_psnr_db", "subject_mean_ssim", "subject_mean_nrmse",
      "subject_mean_mae",
  ]
  with atomic_csv_writer(output_path, fields) as writer:
    for key, group in sorted(groups.items()):
      by_subject: dict[str, list[dict]] = defaultdict(list)
      for row in group:
        by_subject[row["subject_id"]].append(row)
      def subject_mean(field: str) -> float:
        return float(np.mean([
          np.mean([float(row[field]) for row in subject_rows])
          for subject_rows in by_subject.values()
        ]))
      writer.writerow({
        "run_id": run_id, "dataset": key[0], "count_level": key[1], "method": key[2],
        "n_subjects": len(by_subject), "n_cases": len(group),
        "subject_mean_psnr_db": subject_mean("psnr_db"),
        "subject_mean_ssim": subject_mean("ssim"),
        "subject_mean_nrmse": subject_mean("nrmse"),
        "subject_mean_mae": subject_mean("mae"),
      })
  finalize_csv(
    output_path, schema_name="subject_weighted_cross_count_summary_v2",
    primary_key=["run_id", "dataset", "count_level", "method"], split="test_locked",
  )
  return output_path


def run_cross_count(root: Path, config_path: Path, protocol_path: Path) -> Path:
  cfg, _ = load_contract(config_path)
  require_locked_execution(cfg)
  protocol = load_protocol(protocol_path)
  if protocol.kind != "cross_count" or protocol.split != "test_locked":
    raise ValueError("cross-count runner requires the locked cross_count protocol")
  source = root / str(cfg["outputs"]["per_case_metrics"])
  verify_success_marker(source.parent)
  output_dir = root / protocol.output
  output_dir.mkdir(parents=True, exist_ok=True)
  run_metadata = write_run_metadata(
    output_dir, root=root, config_path=config_path, protocol_path=protocol_path,
  )
  output = aggregate_cross_count(
    source, output_dir / "cross_count.csv",
    datasets=set(protocol.raw["datasets"]),
    count_levels={int(value) for value in protocol.raw["count_levels"]},
    methods=set(protocol.methods),
  )
  manifest = output.with_suffix(output.suffix + ".manifest.json")
  write_success_marker(output_dir, [
    run_metadata, output_dir / "environment_freeze.txt", output, manifest,
  ])
  return output
