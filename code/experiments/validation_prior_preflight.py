"""Run a read-only two-dataset validation-prior GPU smoke test."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from sam_kem.prior_preflight import run_validation_prior_preflight


def main() -> int:
  parser = argparse.ArgumentParser()
  parser.add_argument("--root", type=Path, default=Path("."))
  parser.add_argument("--config", type=Path, required=True)
  parser.add_argument("--manifest", type=Path, required=True)
  parser.add_argument("--count-level", type=int, default=200000)
  args = parser.parse_args()
  result = run_validation_prior_preflight(
    args.root, args.config, args.manifest, count_level=args.count_level,
  )
  print(json.dumps(result, indent=2, sort_keys=True))
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
