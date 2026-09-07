"""Run the reviewed incremental validation-grid shell as resumable shards."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from sam_kem.sweep_incremental import run_incremental_validation_sweep_shards


def main() -> int:
  parser = argparse.ArgumentParser()
  parser.add_argument("--root", type=Path, default=Path("."))
  parser.add_argument("--config", type=Path, required=True)
  parser.add_argument("--protocol", type=Path, required=True)
  parser.add_argument("--selection-audit", type=Path, required=True)
  parser.add_argument("--plan", type=Path, required=True)
  parser.add_argument("--smoke-audit", type=Path, required=True)
  parser.add_argument("--output-dir", type=Path, required=True)
  parser.add_argument("--workers", type=int, required=True)
  parser.add_argument("--resume-verified", action="store_true")
  parser.add_argument("--release-check-only", action="store_true")
  args = parser.parse_args()
  result = run_incremental_validation_sweep_shards(
    args.root, args.config, args.protocol, args.selection_audit, args.plan,
    args.smoke_audit, args.output_dir, workers=args.workers,
    resume_verified=args.resume_verified,
    release_check_only=args.release_check_only,
  )
  print(json.dumps(result, indent=2, sort_keys=True))
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
