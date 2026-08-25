from pathlib import Path

from sam_kem.protocols import load_protocol


def test_all_repository_protocols_are_valid():
  paths = sorted(Path("code/configs/protocols").glob("*.yaml"))
  assert len(paths) == 8
  protocols = [load_protocol(path) for path in paths]
  assert {p.kind for p in protocols} == {
    "training", "hyperparameter_sweep", "primary", "comparison",
    "ablation", "convergence", "cross_count",
    "runtime",
  }
