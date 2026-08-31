from pathlib import Path

from sam_kem.protocols import Protocol
from sam_kem.sweep_preflight import sweep_workload


def test_declared_validation_sweep_workload_is_exact():
  protocol = Protocol(
    name="sweep", kind="hyperparameter_sweep", split="validation",
    methods=(
      "MLEM", "OSEM", "Bowsher-KEM", "SelfKEM", "MR-KEM", "CT-KEM",
      "SAM-KEM-MedSAM2-Qwen3-full",
    ),
    output=Path("runs/sweep"),
    raw={"search_space": {
      "sigma": [0.025, 0.05, 0.1], "radius": [1, 2, 3],
      "iterations": [10, 20, 40, 60, 80, 100, 120],
      "alpha": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
      "bowsher_neighbours": [4, 8, 12], "bowsher_radius": [1, 2, 3],
      "mlem_iterations": [1, 5, 10, 20, 40, 60, 80, 120],
      "osem_cycles": [1, 2, 5, 10, 20],
    }},
  )
  workload = sweep_workload(
    protocol, {"brainweb": 8352, "hecktor": 540},
    prompt_mlem_iterations=10, osem_subsets=10,
  )
  assert workload["per_dataset"]["brainweb"]["result_rows_per_case"] == 580
  assert workload["per_dataset"]["hecktor"]["result_rows_per_case"] == 580
  assert workload["per_dataset"]["brainweb"]["operator_updates_per_case"] == 4650
  assert workload["total_result_rows"] == 5_157_360
  assert workload["total_operator_updates"] == 41_347_800
