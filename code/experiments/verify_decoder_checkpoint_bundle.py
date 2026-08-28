"""Independently verify the embedded SHA manifest in a decoder bundle."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from sam_kem.artifacts import atomic_write_json
from sam_kem.delivery import verify_decoder_checkpoint_bundle


def main() -> int:
  parser = argparse.ArgumentParser()
  parser.add_argument("--archive", type=Path, required=True)
  parser.add_argument("--output", type=Path, required=True)
  args = parser.parse_args()
  result = verify_decoder_checkpoint_bundle(args.archive)
  atomic_write_json(args.output, result)
  print(json.dumps(result, indent=2, sort_keys=True))
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
