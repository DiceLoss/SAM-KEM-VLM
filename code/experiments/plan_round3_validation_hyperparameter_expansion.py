"""Plan the third local shell for still-boundary validation optima."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from sam_kem.sweep_expansion_round3 import (
  plan_round3_validation_hyperparameter_expansion,
)


def main() -> int:
  parser = argparse.ArgumentParser()
  parser.add_argument("--root", type=Path, default=Path("."))
  parser.add_argument("--union-audit", type=Path, required=True)
  parser.add_argument("--round2-plan", type=Path, required=True)
  parser.add_argument("--round2-merge-audit", type=Path, required=True)
  parser.add_argument("--output", type=Path, required=True)
  args = parser.parse_args()
  result = plan_round3_validation_hyperparameter_expansion(
    args.root, args.union_audit, args.round2_plan,
    args.round2_merge_audit, args.output,
  )
  print(json.dumps(result, indent=2, sort_keys=True))
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
