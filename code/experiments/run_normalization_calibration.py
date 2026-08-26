"""Fit normalization candidates on train data and diagnose them on validation."""

from pathlib import Path

from sam_kem.normalization import run_normalization_calibration


if __name__ == "__main__":
  root = Path(__file__).resolve().parents[2]
  candidates, diagnostics = run_normalization_calibration(
    root, root / "code/configs/experiment.yaml",
    root / "code/configs/protocols/normalization_calibration.yaml",
  )
  print(f"wrote {candidates}\nwrote {diagnostics}")
