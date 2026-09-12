"""Run the fixed two-slice smoke for the fifth incremental shell."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from sam_kem.sweep_incremental_round5_smoke import (
  run_round5_incremental_validation_sweep_smoke,
)


def main() -> int:
  parser = argparse.ArgumentParser()
  parser.add_argument("--root", type=Path, default=Path("."))
  parser.add_argument("--config", type=Path, required=True)
  parser.add_argument("--protocol", type=Path, required=True)
  parser.add_argument("--union-audit", type=Path, required=True)
  parser.add_argument("--round4-plan", type=Path, required=True)
  parser.add_argument("--round4-merge-audit", type=Path, required=True)
  parser.add_argument("--round5-plan", type=Path, required=True)
  parser.add_argument("--output-dir", type=Path, required=True)
  parser.add_argument("--audit", type=Path, required=True)
  args = parser.parse_args()
  result = run_round5_incremental_validation_sweep_smoke(
    args.root, args.config, args.protocol, args.union_audit,
    args.round4_plan, args.round4_merge_audit, args.round5_plan,
    args.output_dir, args.audit,
  )
  print(json.dumps(result, indent=2, sort_keys=True))
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
