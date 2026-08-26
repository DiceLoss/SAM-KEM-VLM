"""Generate continuous trained-decoder priors for validation/test cases."""
import argparse
from pathlib import Path
from sam_kem.prior_generation import generate_priors

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--splits", nargs="+", choices=("train", "validation", "test_locked"), required=True)
  parser.add_argument("--manifest", type=Path, default=Path("results/metrics/case_manifest_inputs.csv"))
  args = parser.parse_args()
  root = Path(__file__).resolve().parents[2]
  count = generate_priors(
    root, root / "code/configs/experiment.yaml",
    root / args.manifest, splits=set(args.splits),
  )
  print(f"wrote {count} priors")
