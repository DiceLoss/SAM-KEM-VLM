"""Freeze the complete slice-group plan before a formal validation sweep."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from sam_kem.sweep_sharding import build_validation_sweep_shard_plan


def main() -> int:
  parser = argparse.ArgumentParser()
  parser.add_argument("--root", type=Path, default=Path("."))
  parser.add_argument("--config", type=Path, required=True)
  parser.add_argument("--protocol", type=Path, required=True)
  parser.add_argument("--runtime-preflight", type=Path, required=True)
  parser.add_argument("--parallel-preflight", type=Path, required=True)
  parser.add_argument("--smoke-audit", type=Path, required=True)
  parser.add_argument("--output", type=Path, required=True)
  args = parser.parse_args()
  result = build_validation_sweep_shard_plan(
    args.root, args.config, args.protocol, args.runtime_preflight,
    args.parallel_preflight, args.smoke_audit, args.output,
  )
  print(json.dumps(result, indent=2, sort_keys=True))
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
