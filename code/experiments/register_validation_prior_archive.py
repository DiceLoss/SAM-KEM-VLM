"""Create a downstream sweep config after LFS publication of validation priors."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from sam_kem.prior_registration import register_validation_prior_archive


def main() -> int:
  parser = argparse.ArgumentParser()
  parser.add_argument("--root", type=Path, default=Path("."))
  parser.add_argument("--source-config", type=Path, required=True)
  parser.add_argument("--build-record", type=Path, required=True)
  parser.add_argument("--verify-record", type=Path, required=True)
  parser.add_argument("--verified-download", type=Path, required=True)
  parser.add_argument("--archive-uri", required=True)
  parser.add_argument("--archive-version", required=True)
  parser.add_argument("--output-config", type=Path, required=True)
  parser.add_argument("--registration-record", type=Path, required=True)
  args = parser.parse_args()
  result = register_validation_prior_archive(
    args.root, args.source_config, args.build_record, args.verify_record,
    args.verified_download, archive_uri=args.archive_uri,
    archive_version=args.archive_version, output_config_path=args.output_config,
    registration_record_path=args.registration_record,
  )
  print(json.dumps(result, indent=2, sort_keys=True))
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
