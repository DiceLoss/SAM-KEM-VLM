"""Plan, but do not execute, the reviewed validation-grid expansion shell."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from sam_kem.sweep_expansion import plan_validation_hyperparameter_expansion


def main() -> int:
  parser = argparse.ArgumentParser()
  parser.add_argument("--root", type=Path, default=Path("."))
  parser.add_argument("--selection-audit", type=Path, required=True)
  parser.add_argument("--output", type=Path, required=True)
  args = parser.parse_args()
  result = plan_validation_hyperparameter_expansion(
    args.root, args.selection_audit, args.output,
  )
  print(json.dumps(result, indent=2, sort_keys=True))
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
