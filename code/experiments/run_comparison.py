"""Paired baseline-comparison entry point."""
from pathlib import Path
from sam_kem.experiments import run_protocol
from sam_kem.statistics import paired_subject_statistics
from sam_kem.artifacts import write_success_marker
from sam_kem.config import load_contract
from sam_kem.protocols import load_protocol

if __name__ == "__main__":
  root = Path(__file__).resolve().parents[2]
  protocol_path = root / "code/configs/protocols/comparison.yaml"
  protocol = load_protocol(protocol_path)
  output_dir = root / protocol.output
  per_case = run_protocol(
    root, root / "code/configs/experiment.yaml",
    protocol_path, finalize_run=False,
  )
  _, contract = load_contract(root / "code/configs/experiment.yaml")
  statistics = paired_subject_statistics(
    per_case, output_dir / "paired_subject_statistics.csv",
    proposed_method="SAM-KEM-MedSAM2-Qwen3-full",
    experiment_seed=contract.experiment_seed,
  )
  write_success_marker(output_dir, [
    output_dir / "run_metadata.json", output_dir / "environment_freeze.txt",
    per_case, per_case.with_suffix(per_case.suffix + ".manifest.json"),
    statistics, statistics.with_suffix(statistics.suffix + ".manifest.json"),
  ])
  print(per_case, statistics)
