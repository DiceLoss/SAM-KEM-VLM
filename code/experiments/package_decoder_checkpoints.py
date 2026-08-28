"""Build the verified local decoder bundle pending immutable upload."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from sam_kem.delivery import build_decoder_checkpoint_bundle


def main() -> int:
  parser = argparse.ArgumentParser()
  parser.add_argument("--root", type=Path, default=Path("."))
  parser.add_argument("--config", type=Path, required=True)
  parser.add_argument("--output", type=Path, required=True)
  parser.add_argument("--record", type=Path, required=True)
  args = parser.parse_args()
  result = build_decoder_checkpoint_bundle(
    args.root, args.config, args.output, args.record,
  )
  print(json.dumps(result, indent=2, sort_keys=True))
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
