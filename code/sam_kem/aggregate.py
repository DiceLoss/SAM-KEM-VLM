"""Duplicate-safe aggregation for cross-count reporting."""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import numpy as np

from .config import load_contract, require_locked_execution
from .protocols import load_protocol
from .runmeta import write_run_metadata
from .artifacts import atomic_csv_writer, finalize_csv, write_success_marker


def aggregate_cross_count(per_case_path: Path, output_path: Path) -> Path:
  rows = list(csv.DictReader(per_case_path.open("r", encoding="utf-8", newline="")))
  case_keys = [
    tuple(row[name] for name in ("dataset", "subject_id", "slice_id", "realization_id", "count_level", "method"))
    for row in rows
  ]
  if len(case_keys) != len(set(case_keys)):
    raise ValueError("refusing to aggregate duplicate per-case keys")
  groups: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
  for row in rows:
    groups[(row["dataset"], row["count_level"], row["method"])].append(row)
  output_path.parent.mkdir(parents=True, exist_ok=True)
  fields = [
      "dataset", "count_level", "method", "n_subjects", "n_cases",
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
        "dataset": key[0], "count_level": key[1], "method": key[2],
        "n_subjects": len(by_subject), "n_cases": len(group),
        "subject_mean_psnr_db": subject_mean("psnr_db"),
        "subject_mean_ssim": subject_mean("ssim"),
        "subject_mean_nrmse": subject_mean("nrmse"),
        "subject_mean_mae": subject_mean("mae"),
      })
  finalize_csv(
    output_path, schema_name="subject_weighted_cross_count_summary_v1",
    primary_key=["dataset", "count_level", "method"], split="test_locked",
  )
  return output_path


def run_cross_count(root: Path, config_path: Path, protocol_path: Path) -> Path:
  cfg, _ = load_contract(config_path)
  require_locked_execution(cfg)
  protocol = load_protocol(protocol_path)
  if protocol.kind != "cross_count" or protocol.split != "test_locked":
    raise ValueError("cross-count runner requires the locked cross_count protocol")
  source = root / str(cfg["outputs"]["per_case_metrics"])
  output_dir = root / protocol.output
  output_dir.mkdir(parents=True, exist_ok=True)
  run_metadata = write_run_metadata(
    output_dir, root=root, config_path=config_path, protocol_path=protocol_path,
  )
  output = aggregate_cross_count(source, output_dir / "cross_count.csv")
  manifest = output.with_suffix(output.suffix + ".manifest.json")
  write_success_marker(output_dir, [
    run_metadata, output_dir / "environment_freeze.txt", output, manifest,
  ])
  return output
