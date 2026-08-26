"""Paired baseline-comparison entry point."""
from pathlib import Path
from sam_kem.experiments import run_protocol
from sam_kem.statistics import paired_subject_statistics
from sam_kem.artifacts import write_success_marker

if __name__ == "__main__":
  root = Path(__file__).resolve().parents[2]
  per_case = run_protocol(
    root, root / "code/configs/experiment.yaml",
    root / "code/configs/protocols/comparison.yaml", finalize_run=False,
  )
  statistics = paired_subject_statistics(
    per_case, root / "runs/comparison/paired_subject_statistics.csv",
    proposed_method="SAM-KEM-MedSAM2-Qwen3-full", experiment_seed=20260817,
  )
  output_dir = root / "runs/comparison"
  write_success_marker(output_dir, [
    output_dir / "run_metadata.json", output_dir / "environment_freeze.txt",
    per_case, per_case.with_suffix(per_case.suffix + ".manifest.json"),
    statistics, statistics.with_suffix(statistics.suffix + ".manifest.json"),
  ])
  print(per_case, statistics)
