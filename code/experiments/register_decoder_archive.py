"""Create a downstream config after immutable decoder archive publication."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from sam_kem.checkpoint_registration import register_decoder_archive


def main() -> int:
  parser = argparse.ArgumentParser()
  parser.add_argument("--root", type=Path, default=Path("."))
  parser.add_argument("--training-config", type=Path, required=True)
  parser.add_argument("--build-record", type=Path, required=True)
  parser.add_argument("--verify-record", type=Path, required=True)
  parser.add_argument("--verified-download", type=Path, required=True)
  parser.add_argument("--archive-uri", required=True)
  parser.add_argument("--archive-version", required=True)
  parser.add_argument("--output-config", type=Path, required=True)
  parser.add_argument("--registration-record", type=Path, required=True)
  args = parser.parse_args()
  result = register_decoder_archive(
    args.root, args.training_config, args.build_record, args.verify_record,
    args.verified_download, archive_uri=args.archive_uri,
    archive_version=args.archive_version, output_config_path=args.output_config,
    registration_record_path=args.registration_record,
  )
  print(json.dumps(result, indent=2, sort_keys=True))
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
