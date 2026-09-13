"""Publish the SHA-bound terminal validation hyperparameter selection."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from sam_kem.hyperparameter_freeze import freeze_reviewed_hyperparameters


def main() -> int:
  parser = argparse.ArgumentParser()
  parser.add_argument("--root", type=Path, default=Path("."))
  parser.add_argument("--config", type=Path, required=True)
  parser.add_argument("--protocol", type=Path, required=True)
  parser.add_argument("--selection-audit", type=Path, required=True)
  parser.add_argument("--audit", type=Path, required=True)
  parser.add_argument("--release-check-only", action="store_true")
  args = parser.parse_args()
  result = freeze_reviewed_hyperparameters(
    args.root, args.config, args.protocol, args.selection_audit, args.audit,
    release_check_only=args.release_check_only,
  )
  print(json.dumps(result, indent=2, sort_keys=True))
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
