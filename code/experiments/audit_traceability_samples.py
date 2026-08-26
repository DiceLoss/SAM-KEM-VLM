"""Audit the committed raw-source traceability samples."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from sam_kem.traceability import audit_traceability_archives


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--root", type=Path, default=Path("."))
  args = parser.parse_args()
  print(json.dumps(
    audit_traceability_archives(args.root.resolve()), indent=2, sort_keys=True,
  ))


if __name__ == "__main__":
  main()
