"""Inventory BrainWeb MR candidates and the declared HECKTOR CT channel."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from sam_kem.anatomy_inventory import audit_anatomy_sources


def main() -> int:
  parser = argparse.ArgumentParser()
  parser.add_argument("--root", type=Path, default=Path("."))
  parser.add_argument("--source-manifest", type=Path, required=True)
  parser.add_argument("--output", type=Path, required=True)
  args = parser.parse_args()
  result = audit_anatomy_sources(args.root, args.source_manifest, args.output)
  print(json.dumps(result, indent=2, sort_keys=True))
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
