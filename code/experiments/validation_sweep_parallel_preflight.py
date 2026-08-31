"""Measure exact-grid validation throughput using eight CPU processes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from sam_kem.sweep_preflight import run_parallel_sweep_scaling_preflight


def main() -> int:
  parser = argparse.ArgumentParser()
  parser.add_argument("--root", type=Path, default=Path("."))
  parser.add_argument("--config", type=Path, required=True)
  parser.add_argument("--protocol", type=Path, required=True)
  parser.add_argument("--runtime-preflight", type=Path, required=True)
  parser.add_argument("--output", type=Path, required=True)
  parser.add_argument("--workers", type=int, required=True)
  args = parser.parse_args()
  result = run_parallel_sweep_scaling_preflight(
    args.root, args.config, args.protocol, args.runtime_preflight, args.output,
    workers=args.workers,
  )
  print(json.dumps(result, indent=2, sort_keys=True))
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
