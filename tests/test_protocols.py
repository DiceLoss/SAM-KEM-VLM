from pathlib import Path

from sam_kem.protocols import load_protocol


# Allow-list of protocol kinds. Stage 7A-v2 planning protocol is
# added as `stage7a_v2_prompt_planning`.
KNOWN_PROTOCOL_KINDS = {
    "training", "hyperparameter_sweep", "primary", "comparison",
    "ablation", "convergence", "cross_count",
    "runtime", "normalization_calibration",
    "stage7a_prompt_planning",
    "stage7a_v2_prompt_planning",
}


def test_all_repository_protocols_are_valid():
  paths = sorted(Path("code/configs/protocols").glob("*.yaml"))
  # Must contain at least the 9 original protocols; new protocols
  # (e.g. stage7a_v1.yaml) are allowed in addition.
  assert len(paths) >= 9
  protocols = [load_protocol(path) for path in paths]
  assert {p.kind for p in protocols} <= KNOWN_PROTOCOL_KINDS
  # Original 9 must all be present.
  required = {
    "training", "hyperparameter_sweep", "primary", "comparison",
    "ablation", "convergence", "cross_count",
    "runtime", "normalization_calibration",
  }
  assert required <= {p.kind for p in protocols}
