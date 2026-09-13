"""Audit subject-balanced selection over six reviewed validation sources."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from sam_kem.sweep_union_selection_round5 import (
  audit_round5_union_hyperparameter_selection,
)


def main() -> int:
  parser = argparse.ArgumentParser()
  parser.add_argument("--root", type=Path, default=Path("."))
  parser.add_argument("--config", type=Path, required=True)
  parser.add_argument("--protocol", type=Path, required=True)
  parser.add_argument("--original-merge-audit", type=Path, required=True)
  parser.add_argument("--original-selection-audit", type=Path, required=True)
  parser.add_argument("--round1-plan", type=Path, required=True)
  parser.add_argument("--round1-merge-audit", type=Path, required=True)
  parser.add_argument("--round1-union-audit", type=Path, required=True)
  parser.add_argument("--round2-plan", type=Path, required=True)
  parser.add_argument("--round2-merge-audit", type=Path, required=True)
  parser.add_argument("--round2-union-audit", type=Path, required=True)
  parser.add_argument("--round3-plan", type=Path, required=True)
  parser.add_argument("--round3-merge-audit", type=Path, required=True)
  parser.add_argument("--round3-union-audit", type=Path, required=True)
  parser.add_argument("--round4-plan", type=Path, required=True)
  parser.add_argument("--round4-merge-audit", type=Path, required=True)
  parser.add_argument("--round4-union-audit", type=Path, required=True)
  parser.add_argument("--round5-plan", type=Path, required=True)
  parser.add_argument("--round5-merge-audit", type=Path, required=True)
  parser.add_argument("--output", type=Path, required=True)
  args = parser.parse_args()
  result = audit_round5_union_hyperparameter_selection(
    args.root, args.config, args.protocol,
    args.original_merge_audit, args.original_selection_audit,
    args.round1_plan, args.round1_merge_audit, args.round1_union_audit,
    args.round2_plan, args.round2_merge_audit, args.round2_union_audit,
    args.round3_plan, args.round3_merge_audit, args.round3_union_audit,
    args.round4_plan, args.round4_merge_audit, args.round4_union_audit,
    args.round5_plan, args.round5_merge_audit, args.output,
  )
  print(json.dumps(result, indent=2, sort_keys=True))
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
