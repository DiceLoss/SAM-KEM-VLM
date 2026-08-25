"""Descriptive convergence entry point; never used for model selection."""
from pathlib import Path
from sam_kem.convergence import run_convergence

if __name__ == "__main__":
  root = Path(__file__).resolve().parents[2]
  print(run_convergence(root, root / "code/configs/experiment.yaml", root / "code/configs/protocols/convergence.yaml"))

