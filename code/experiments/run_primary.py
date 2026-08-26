"""Derive the headline rows from the unique locked comparison artifact."""
from pathlib import Path
from sam_kem.primary import derive_primary

if __name__ == "__main__":
  root = Path(__file__).resolve().parents[2]
  print(derive_primary(
    root, root / "code/configs/experiment.yaml",
    root / "code/configs/protocols/primary.yaml",
  ))
