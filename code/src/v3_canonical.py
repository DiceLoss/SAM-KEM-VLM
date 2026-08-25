"""Retired v3 entry point.

The former phase orchestrator ignored subprocess failures and depended on
machine-specific absolute paths. It is intentionally fail-closed after the
scientific audit. Use ``sam-kem`` and ``code/experiments`` instead.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from sam_kem.audit import audit_repository


def main() -> int:
  parser = argparse.ArgumentParser()
  parser.add_argument("--phase", type=int, required=True)
  args = parser.parse_args()
  if args.phase != 9:
    parser.error(
      "v3 phases are retired because they produced non-admissible results; "
      "follow docs/EXECUTION.md"
    )
  root = Path(__file__).resolve().parents[2]
  issues = audit_repository(root, root / "code/configs/experiment.yaml")
  for issue in issues:
    print(f"{issue.severity.upper()} {issue.code}: {issue.message}")
  return 1 if issues else 0


if __name__ == "__main__":
  raise SystemExit(main())
