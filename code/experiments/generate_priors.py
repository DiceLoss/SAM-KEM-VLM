"""Generate continuous trained-decoder priors for validation/test cases."""
import argparse
import json
from pathlib import Path
from sam_kem.prior_generation import generate_priors
from sam_kem.config import load_yaml_strict

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--config", type=Path, required=True)
  parser.add_argument("--splits", nargs="+", choices=("train", "validation", "test_locked"), required=True)
  parser.add_argument("--manifest", type=Path)
  parser.add_argument("--output-dir", type=Path, required=True)
  parser.add_argument("--resume-verified", action="store_true")
  args = parser.parse_args()
  root = Path(__file__).resolve().parents[2]
  config_path = args.config.resolve()
  cfg = load_yaml_strict(config_path)
  manifest = root / (
    args.manifest if args.manifest is not None
    else Path(str(cfg["outputs"]["pretraining_manifest"]))
  )
  result = generate_priors(
    root, config_path, manifest, splits=set(args.splits),
    resume_verified=args.resume_verified, output_dir=args.output_dir,
  )
  print(json.dumps(result, indent=2, sort_keys=True))
