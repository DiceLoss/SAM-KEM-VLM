"""Validation-only hyperparameter sweep entry point."""

import argparse
from pathlib import Path

from sam_kem.sweep import run_validation_sweep


def main() -> int:
  parser = argparse.ArgumentParser()
  parser.add_argument("--config", type=Path, required=True)
  parser.add_argument("--protocol", type=Path, required=True)
  args = parser.parse_args()
  root = Path(__file__).resolve().parents[2]
  print(run_validation_sweep(root, args.config.resolve(), args.protocol.resolve()))
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
