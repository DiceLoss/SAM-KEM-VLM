"""Audit subject-balanced selection over original and incremental validation results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from sam_kem.sweep_union_selection import audit_union_hyperparameter_selection


def main() -> int:
  parser = argparse.ArgumentParser()
  parser.add_argument("--root", type=Path, default=Path("."))
  parser.add_argument("--config", type=Path, required=True)
  parser.add_argument("--protocol", type=Path, required=True)
  parser.add_argument("--original-merge-audit", type=Path, required=True)
  parser.add_argument("--original-selection-audit", type=Path, required=True)
  parser.add_argument("--plan", type=Path, required=True)
  parser.add_argument("--incremental-merge-audit", type=Path, required=True)
  parser.add_argument("--output", type=Path, required=True)
  args = parser.parse_args()
  result = audit_union_hyperparameter_selection(
    args.root, args.config, args.protocol, args.original_merge_audit,
    args.original_selection_audit, args.plan, args.incremental_merge_audit,
    args.output,
  )
  print(json.dumps(result, indent=2, sort_keys=True))
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
