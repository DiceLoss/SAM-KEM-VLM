"""Write and verify two exact case fragments before the formal sharded sweep."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from sam_kem.sweep_shard_smoke import run_sweep_shard_smoke


def main() -> int:
  parser = argparse.ArgumentParser()
  parser.add_argument("--root", type=Path, default=Path("."))
  parser.add_argument("--config", type=Path, required=True)
  parser.add_argument("--protocol", type=Path, required=True)
  parser.add_argument("--runtime-preflight", type=Path, required=True)
  parser.add_argument("--parallel-preflight", type=Path, required=True)
  parser.add_argument("--output-dir", type=Path, required=True)
  parser.add_argument("--audit", type=Path, required=True)
  args = parser.parse_args()
  result = run_sweep_shard_smoke(
    args.root, args.config, args.protocol, args.runtime_preflight,
    args.parallel_preflight, args.output_dir, args.audit,
  )
  print(json.dumps(result, indent=2, sort_keys=True))
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
