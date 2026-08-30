"""Audit every generated validation prior before any parameter sweep."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from sam_kem.prior_audit import audit_validation_priors


def main() -> int:
  parser = argparse.ArgumentParser()
  parser.add_argument("--root", type=Path, default=Path("."))
  parser.add_argument("--config", type=Path, required=True)
  parser.add_argument("--case-manifest", type=Path, required=True)
  parser.add_argument("--prior-dir", type=Path, required=True)
  parser.add_argument("--diagnostics", type=Path, required=True)
  parser.add_argument("--summary", type=Path, required=True)
  args = parser.parse_args()
  result = audit_validation_priors(
    args.root, args.config, args.case_manifest, args.prior_dir,
    args.diagnostics, args.summary,
  )
  print(json.dumps(result, indent=2, sort_keys=True))
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
