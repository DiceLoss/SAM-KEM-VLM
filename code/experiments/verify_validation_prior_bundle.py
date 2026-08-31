"""Independently verify every member of a validation-prior bundle."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from sam_kem.artifacts import atomic_write_json
from sam_kem.prior_delivery import verify_validation_prior_bundle


def main() -> int:
  parser = argparse.ArgumentParser()
  parser.add_argument("--archive", type=Path, required=True)
  parser.add_argument("--output", type=Path, required=True)
  args = parser.parse_args()
  result = verify_validation_prior_bundle(args.archive)
  atomic_write_json(args.output, result)
  print(json.dumps(result, indent=2, sort_keys=True))
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
