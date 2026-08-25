"""Run cached-prior reconstruction runtime benchmark."""
from pathlib import Path
from sam_kem.runtime import run_runtime

if __name__ == "__main__":
  root = Path(__file__).resolve().parents[2]
  print(run_runtime(root, root / "code/configs/experiment.yaml", root / "code/configs/protocols/runtime.yaml"))
