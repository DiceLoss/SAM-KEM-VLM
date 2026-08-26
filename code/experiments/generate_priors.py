"""Generate continuous trained-decoder priors for validation/test cases."""
import argparse
from pathlib import Path
from sam_kem.prior_generation import generate_priors
from sam_kem.config import load_yaml_strict

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--splits", nargs="+", choices=("train", "validation", "test_locked"), required=True)
  parser.add_argument("--manifest", type=Path)
  args = parser.parse_args()
  root = Path(__file__).resolve().parents[2]
  config_path = root / "code/configs/experiment.yaml"
  cfg = load_yaml_strict(config_path)
  manifest = root / (
    args.manifest if args.manifest is not None
    else Path(str(cfg["outputs"]["pretraining_manifest"]))
  )
  count = generate_priors(
    root, config_path, manifest, splits=set(args.splits),
  )
  print(f"wrote {count} priors")
