"""Freeze or ingest the researcher-approved anatomy comparison sources."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from sam_kem.anatomy_preparation import (
  audit_ingested_anatomy,
  freeze_anatomy_source_manifest,
  ingest_anatomy_manifest,
)


def main() -> int:
  parser = argparse.ArgumentParser()
  sub = parser.add_subparsers(dest="command", required=True)
  freeze = sub.add_parser("freeze")
  freeze.add_argument("--root", type=Path, default=Path("."))
  freeze.add_argument("--inventory", type=Path, required=True)
  freeze.add_argument("--inventory-sha256", required=True)
  freeze.add_argument("--source-manifest", type=Path, required=True)
  freeze.add_argument("--output", type=Path, required=True)
  ingest = sub.add_parser("ingest")
  ingest.add_argument("--root", type=Path, default=Path("."))
  ingest.add_argument("--manifest", type=Path, required=True)
  ingest.add_argument("--arrays", type=Path, required=True)
  ingest.add_argument("--splits", nargs="+", choices=("train", "validation"), required=True)
  ingest.add_argument("--resume-verified", action="store_true")
  audit = sub.add_parser("audit")
  audit.add_argument("--root", type=Path, default=Path("."))
  audit.add_argument("--manifest", type=Path, required=True)
  audit.add_argument("--arrays", type=Path, required=True)
  audit.add_argument("--output", type=Path, required=True)
  audit.add_argument("--splits", nargs="+", choices=("train", "validation"), required=True)
  args = parser.parse_args()
  if args.command == "freeze":
    output = freeze_anatomy_source_manifest(
      args.root, args.inventory, args.source_manifest, args.output,
      expected_inventory_sha256=args.inventory_sha256,
    )
    print(json.dumps({"status": "complete", "output": str(output)}, indent=2))
  elif args.command == "ingest":
    print(json.dumps(ingest_anatomy_manifest(
      args.root, args.manifest, args.arrays, splits=set(args.splits),
      resume_verified=args.resume_verified,
    ), indent=2, sort_keys=True))
  else:
    print(json.dumps(audit_ingested_anatomy(
      args.root, args.manifest, args.arrays, args.output, splits=set(args.splits),
    ), indent=2, sort_keys=True))
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
