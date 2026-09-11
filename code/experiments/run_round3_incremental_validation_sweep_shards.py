"""Release-check the reviewed round-3 incremental shard execution."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from sam_kem.sweep_incremental_round3 import (
  release_check_round3_incremental_validation_sweep,
)


def main() -> int:
  parser = argparse.ArgumentParser()
  parser.add_argument("--root", type=Path, default=Path("."))
  parser.add_argument("--config", type=Path, required=True)
  parser.add_argument("--protocol", type=Path, required=True)
  parser.add_argument("--round2-plan", type=Path, required=True)
  parser.add_argument("--union-audit", type=Path, required=True)
  parser.add_argument("--round2-merge-audit", type=Path, required=True)
  parser.add_argument("--round3-plan", type=Path, required=True)
  parser.add_argument("--output-dir", type=Path, required=True)
  parser.add_argument("--workers", type=int, required=True)
  parser.add_argument("--release-check-only", action="store_true")
  args = parser.parse_args()
  result = release_check_round3_incremental_validation_sweep(
    args.root, args.config, args.protocol, args.round2_plan,
    args.union_audit, args.round2_merge_audit, args.round3_plan,
    args.output_dir, workers=args.workers,
    release_check_only=args.release_check_only,
  )
  print(json.dumps(result, indent=2, sort_keys=True))
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
