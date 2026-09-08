"""Validate the reviewed round-2 incremental shard release identity."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from sam_kem.sweep_incremental_round2 import (
  run_round2_incremental_validation_sweep_shards,
)


def main() -> int:
  parser = argparse.ArgumentParser()
  parser.add_argument("--root", type=Path, default=Path("."))
  parser.add_argument("--config", type=Path, required=True)
  parser.add_argument("--protocol", type=Path, required=True)
  parser.add_argument("--previous-plan", type=Path, required=True)
  parser.add_argument("--union-audit", type=Path, required=True)
  parser.add_argument("--incremental-merge-audit", type=Path, required=True)
  parser.add_argument("--round2-plan", type=Path, required=True)
  parser.add_argument("--output-dir", type=Path, required=True)
  parser.add_argument("--workers", type=int, required=True)
  parser.add_argument("--resume-verified", action="store_true")
  parser.add_argument("--release-check-only", action="store_true")
  args = parser.parse_args()
  result = run_round2_incremental_validation_sweep_shards(
    args.root, args.config, args.protocol, args.previous_plan,
    args.union_audit, args.incremental_merge_audit, args.round2_plan,
    args.output_dir, workers=args.workers,
    resume_verified=args.resume_verified,
    release_check_only=args.release_check_only,
  )
  print(json.dumps(result, indent=2, sort_keys=True))
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
