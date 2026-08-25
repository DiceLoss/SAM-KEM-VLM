"""One-factor-at-a-time ablation entry point."""
from pathlib import Path
from sam_kem.ablation import run_ablation

if __name__ == "__main__":
  root = Path(__file__).resolve().parents[2]
  print(run_ablation(root, root / "code/configs/experiment.yaml", root / "code/configs/protocols/ablation.yaml"))

