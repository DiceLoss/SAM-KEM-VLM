"""Plan the fifth local shell for still-boundary validation optima."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from sam_kem.sweep_expansion_round5 import (
  plan_round5_validation_hyperparameter_expansion,
)


def main() -> int:
  parser = argparse.ArgumentParser()
  parser.add_argument("--root", type=Path, default=Path("."))
  parser.add_argument("--union-audit", type=Path, required=True)
  parser.add_argument("--round4-plan", type=Path, required=True)
  parser.add_argument("--round4-merge-audit", type=Path, required=True)
  parser.add_argument("--output", type=Path, required=True)
  args = parser.parse_args()
  result = plan_round5_validation_hyperparameter_expansion(
    args.root, args.union_audit, args.round4_plan,
    args.round4_merge_audit, args.output,
  )
  print(json.dumps(result, indent=2, sort_keys=True))
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
