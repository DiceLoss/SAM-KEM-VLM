"""Transactionally merge an audited validation shard set without freezing it."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from sam_kem.sweep_merge import merge_audited_validation_sweep


def main() -> int:
  parser = argparse.ArgumentParser()
  parser.add_argument("--root", type=Path, default=Path("."))
  parser.add_argument("--config", type=Path, required=True)
  parser.add_argument("--protocol", type=Path, required=True)
  parser.add_argument("--plan", type=Path, required=True)
  parser.add_argument("--shard-dir", type=Path, required=True)
  parser.add_argument("--shard-audit", type=Path, required=True)
  parser.add_argument("--output-dir", type=Path, required=True)
  parser.add_argument("--audit", type=Path, required=True)
  args = parser.parse_args()
  result = merge_audited_validation_sweep(
    args.root, args.config, args.protocol, args.plan, args.shard_dir,
    args.shard_audit, args.output_dir, args.audit,
  )
  print(json.dumps(result, indent=2, sort_keys=True))
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
