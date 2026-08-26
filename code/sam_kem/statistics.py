"""Subject-level paired statistics without slice-level pseudoreplication."""

from __future__ import annotations

import csv
import hashlib
from collections import defaultdict
from pathlib import Path

import numpy as np
from .artifacts import atomic_csv_writer, finalize_csv, verify_csv_artifact


def _holm(p_values: list[float]) -> list[float]:
  order = np.argsort(p_values)
  adjusted = np.empty(len(p_values), dtype=float)
  running = 0.0
  for rank, index in enumerate(order):
    value = min(1.0, (len(p_values) - rank) * p_values[index])
    running = max(running, value)
    adjusted[index] = running
  return adjusted.tolist()


def paired_subject_statistics(
  per_case_path: Path,
  output_path: Path,
  *,
  proposed_method: str,
  experiment_seed: int,
  bootstrap_repetitions: int = 10000,
) -> Path:
  from scipy.stats import wilcoxon

  record = verify_csv_artifact(per_case_path)
  if record.get("schema_name") != "per_case_reconstruction_metrics_v2":
    raise ValueError("paired statistics require canonical per-case reconstruction metrics")
  rows = list(csv.DictReader(per_case_path.open("r", encoding="utf-8", newline="")))
  run_ids = {row.get("run_id") for row in rows}
  if len(run_ids) != 1 or None in run_ids:
    raise ValueError(f"paired statistics input mixes run IDs: {run_ids}")
  run_id = next(iter(run_ids))
  primary_keys = [
    (
      row["run_id"], row["dataset"], row["count_level"], row["subject_id"], row["slice_id"],
      row["realization_id"], row["method"],
    ) for row in rows
  ]
  if len(primary_keys) != len(set(primary_keys)):
    raise ValueError("paired statistics input contains duplicate per-case method rows")
  for dataset, count in sorted({(row["dataset"], row["count_level"]) for row in rows}):
    method_cases = {}
    for row in rows:
      if (row["dataset"], row["count_level"]) != (dataset, count):
        continue
      method_cases.setdefault(row["method"], set()).add(
        (row["subject_id"], row["slice_id"], row["realization_id"])
      )
    if len({frozenset(cases) for cases in method_cases.values()}) != 1:
      raise ValueError(f"methods do not share identical paired cases for {dataset}/{count}")
  subject_values: dict[tuple[str, str, str, str], list[float]] = defaultdict(list)
  for row in rows:
    subject_values[(row["dataset"], row["count_level"], row["subject_id"], row["method"])].append(float(row["nrmse"]))
  means = {key: float(np.mean(values)) for key, values in subject_values.items()}
  comparisons = []
  for dataset, count in sorted({(row["dataset"], row["count_level"]) for row in rows}):
    methods = sorted({row["method"] for row in rows if row["dataset"] == dataset and row["count_level"] == count})
    subjects = sorted({row["subject_id"] for row in rows if row["dataset"] == dataset and row["count_level"] == count})
    for baseline in methods:
      if baseline == proposed_method:
        continue
      paired = [
        (means[(dataset, count, subject, proposed_method)], means[(dataset, count, subject, baseline)])
        for subject in subjects
        if (dataset, count, subject, proposed_method) in means and (dataset, count, subject, baseline) in means
      ]
      if len(paired) < 2:
        continue
      differences = np.asarray([proposed - base for proposed, base in paired])
      p_value = 1.0 if np.all(differences == 0) else float(
        wilcoxon(differences, zero_method="pratt", alternative="two-sided").pvalue
      )
      comparison_seed = int.from_bytes(hashlib.sha256(
        f"{experiment_seed}|{dataset}|{count}|{baseline}".encode("utf-8")
      ).digest()[:8], "big")
      rng = np.random.default_rng(comparison_seed)
      bootstrap = np.asarray([
        np.mean(rng.choice(differences, size=len(differences), replace=True))
        for _ in range(bootstrap_repetitions)
      ])
      comparisons.append({
        "run_id": run_id, "dataset": dataset, "count_level": count,
        "proposed_method": proposed_method,
        "baseline_method": baseline, "n_subjects": len(paired),
        "mean_difference_nrmse_proposed_minus_baseline": float(differences.mean()),
        "median_difference_nrmse_proposed_minus_baseline": float(np.median(differences)),
        "bootstrap95_low": float(np.quantile(bootstrap, 0.025)),
        "bootstrap95_high": float(np.quantile(bootstrap, 0.975)),
        "wilcoxon_p_raw": p_value,
      })
  adjusted = _holm([row["wilcoxon_p_raw"] for row in comparisons])
  for row, value in zip(comparisons, adjusted):
    row["wilcoxon_p_holm"] = value
  if not comparisons:
    raise ValueError("no complete subject-level method pairs were available")
  output_path.parent.mkdir(parents=True, exist_ok=True)
  with atomic_csv_writer(output_path, list(comparisons[0])) as writer:
    writer.writerows(comparisons)
  finalize_csv(
    output_path, schema_name="paired_subject_statistics_v2",
    primary_key=["run_id", "dataset", "count_level", "proposed_method", "baseline_method"],
    split="test_locked",
  )
  return output_path
