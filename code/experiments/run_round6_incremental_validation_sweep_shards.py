"""Release-check the reviewed Round-6 incremental shard execution."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from sam_kem.sweep_incremental_round6 import (
  release_check_round6_incremental_validation_sweep,
  run_round6_incremental_validation_sweep_shards,
)


def main() -> int:
  parser = argparse.ArgumentParser()
  parser.add_argument("--root", type=Path, default=Path("."))
  parser.add_argument("--config", type=Path, required=True)
  parser.add_argument("--protocol", type=Path, required=True)
  parser.add_argument("--union-audit", type=Path, required=True)
  parser.add_argument("--round5-plan", type=Path, required=True)
  parser.add_argument("--round5-merge-audit", type=Path, required=True)
  parser.add_argument("--round6-plan", type=Path, required=True)
  parser.add_argument("--smoke-audit", type=Path)
  parser.add_argument("--output-dir", type=Path, required=True)
  parser.add_argument("--workers", type=int, required=True)
  parser.add_argument("--resume-verified", action="store_true")
  parser.add_argument("--release-check-only", action="store_true")
  args = parser.parse_args()
  if args.smoke_audit is None:
    if args.resume_verified:
      parser.error("--resume-verified requires --smoke-audit")
    result = release_check_round6_incremental_validation_sweep(
      args.root, args.config, args.protocol, args.union_audit,
      args.round5_plan, args.round5_merge_audit, args.round6_plan,
      args.output_dir, workers=args.workers,
      release_check_only=args.release_check_only,
    )
  else:
    result = run_round6_incremental_validation_sweep_shards(
      args.root, args.config, args.protocol, args.union_audit,
      args.round5_plan, args.round5_merge_audit, args.round6_plan,
      args.smoke_audit, args.output_dir, workers=args.workers,
      resume_verified=args.resume_verified,
      release_check_only=args.release_check_only,
    )
  print(json.dumps(result, indent=2, sort_keys=True))
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
