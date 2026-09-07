"""Audit subject-balanced selection from the merged validation sweep."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from sam_kem.sweep_selection import audit_streaming_hyperparameter_selection


def main() -> int:
  parser = argparse.ArgumentParser()
  parser.add_argument("--root", type=Path, default=Path("."))
  parser.add_argument("--config", type=Path, required=True)
  parser.add_argument("--protocol", type=Path, required=True)
  parser.add_argument("--merge-audit", type=Path, required=True)
  parser.add_argument("--output", type=Path, required=True)
  args = parser.parse_args()
  result = audit_streaming_hyperparameter_selection(
    args.root, args.config, args.protocol, args.merge_audit, args.output,
  )
  print(json.dumps(result, indent=2, sort_keys=True))
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
