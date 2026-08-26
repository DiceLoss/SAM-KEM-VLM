"""Duplicate-safe, subject-weighted cross-count aggregation entry point."""
from pathlib import Path
from sam_kem.aggregate import run_cross_count

if __name__ == "__main__":
  root = Path(__file__).resolve().parents[2]
  print(run_cross_count(
    root, root / "code/configs/experiment.yaml",
    root / "code/configs/protocols/cross_count.yaml",
  ))
