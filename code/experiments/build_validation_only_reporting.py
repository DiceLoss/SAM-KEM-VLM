"""Build the post-freeze validation-only reporting package."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from sam_kem.validation_only_reporting import build_validation_only_reporting_package


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--root", type=Path, default=Path("."))
  parser.add_argument("--config", type=Path, required=True)
  parser.add_argument("--protocol", type=Path, required=True)
  parser.add_argument("--selection-audit", type=Path, required=True)
  parser.add_argument("--freeze-audit", type=Path, required=True)
  parser.add_argument("--output", type=Path, required=True)
  parser.add_argument("--audit", type=Path, required=True)
  parser.add_argument("--release-check-only", action="store_true")
  args = parser.parse_args()
  result = build_validation_only_reporting_package(
    args.root, args.config, args.protocol, args.selection_audit, args.freeze_audit,
    args.output, args.audit, release_check_only=args.release_check_only,
  )
  print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
  main()
