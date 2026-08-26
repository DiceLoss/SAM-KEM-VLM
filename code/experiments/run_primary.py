"""Locked primary experiment entry point."""
from pathlib import Path
from sam_kem.experiments import run_protocol

if __name__ == "__main__":
  root = Path(__file__).resolve().parents[2]
  print(run_protocol(root, root / "code/configs/experiment.yaml", root / "code/configs/protocols/primary.yaml"))

