"""Build a verified local validation-prior bundle pending LFS publication."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from sam_kem.prior_delivery import build_validation_prior_bundle


def main() -> int:
  parser = argparse.ArgumentParser()
  parser.add_argument("--root", type=Path, default=Path("."))
  parser.add_argument("--config", type=Path, required=True)
  parser.add_argument("--prior-dir", type=Path, required=True)
  parser.add_argument("--audit", type=Path, required=True)
  parser.add_argument("--diagnostics", type=Path, required=True)
  parser.add_argument("--output", type=Path, required=True)
  parser.add_argument("--record", type=Path, required=True)
  args = parser.parse_args()
  result = build_validation_prior_bundle(
    args.root, args.config, args.prior_dir, args.audit, args.diagnostics,
    args.output, args.record,
  )
  print(json.dumps(result, indent=2, sort_keys=True))
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
