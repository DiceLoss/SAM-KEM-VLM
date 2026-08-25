"""Validation-only hyperparameter sweep entry point."""
from pathlib import Path
from sam_kem.sweep import run_validation_sweep

if __name__ == "__main__":
  root = Path(__file__).resolve().parents[2]
  print(run_validation_sweep(root, root / "code/configs/experiment.yaml", root / "code/configs/protocols/hyperparameter_sweep.yaml"))

