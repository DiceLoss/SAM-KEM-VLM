"""Run the fixed two-slice numerical smoke for the second incremental shell."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from sam_kem.sweep_incremental_round2_smoke import (
  run_round2_incremental_validation_sweep_smoke,
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
  parser.add_argument("--audit", type=Path, required=True)
  args = parser.parse_args()
  result = run_round2_incremental_validation_sweep_smoke(
    args.root, args.config, args.protocol, args.previous_plan,
    args.union_audit, args.incremental_merge_audit, args.round2_plan,
    args.output_dir, args.audit,
  )
  print(json.dumps(result, indent=2, sort_keys=True))
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
