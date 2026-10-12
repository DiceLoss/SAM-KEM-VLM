"""Read-only asset and secret preflight for the four-case Stage 7 smoke."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import stat
from collections import Counter
from pathlib import Path

import numpy as np

ROLES = ("mlem10", "target", "sinogram")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _secret_available(token_file: Path | None) -> tuple[bool, str]:
    if os.environ.get("API_TOKEN"):
        return True, "environment"
    if token_file is None or not token_file.is_file():
        return False, "missing"
    mode = stat.S_IMODE(token_file.stat().st_mode)
    if mode & 0o077:
        return False, "token_file_permissions_not_private"
    if not token_file.read_text(encoding="utf-8").strip():
        return False, "token_file_empty"
    return True, "private_token_file"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--asset-project", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--api-token-file", type=Path)
    args = parser.parse_args()

    rows = list(csv.DictReader(args.manifest.open(newline="", encoding="utf-8")))
    failures: list[str] = []
    distribution = Counter(row["dataset"] for row in rows)
    identity_ok = (
        len(rows) == 4
        and distribution == Counter({"brainweb": 2, "hecktor": 2})
        and all(row["count_level"] == "100000" for row in rows)
        and all(row["realization_id"] == "0" for row in rows)
    )
    if not identity_ok:
        failures.append("smoke manifest identity/distribution mismatch")

    assets = []
    for row in rows:
        for role in ROLES:
            path = Path(row[f"{role}_path"])
            actual = sha256_file(path) if path.is_file() else ""
            sha_match = actual == row[f"{role}_sha256"]
            shape = None
            finite = False
            if sha_match:
                array = np.load(path, mmap_mode="r")
                shape = list(array.shape)
                finite = bool(np.isfinite(array).all())
            record = {
                "case": "/".join((row["dataset"], row["subject_id"],
                                   row["slice_id"], "100000", "r0")),
                "role": role, "path": str(path), "present": path.is_file(),
                "actual_sha256": actual,
                "expected_sha256": row[f"{role}_sha256"],
                "sha_match": sha_match, "shape": shape, "finite": finite,
            }
            assets.append(record)
            if not sha_match or not finite:
                failures.append(f"invalid asset: {record['case']}:{role}")

    project = args.asset_project.resolve()
    required = (
        project / "checkpoints/dev_trainmax_v1/medsam2_decoder_brainweb.pt",
        project / "checkpoints/dev_trainmax_v1/medsam2_decoder_hecktor.pt",
        project / "checkpoints/MedSAM2_latest.pt",
        project / "external/MedSAM2",
    )
    model_assets = {str(path): path.exists() for path in required}
    if not all(model_assets.values()):
        failures.append("one or more MedSAM2/checkpoint assets are missing")

    secret_ok, secret_source = _secret_available(args.api_token_file)
    if not secret_ok:
        failures.append(f"API secret unavailable: {secret_source}")

    audit = {
        "artifact_kind": "stage7_formal_fixed_case_smoke_preflight",
        "status": "PASS_ready_for_four_case_smoke" if not failures
                  else "BLOCKED_smoke_preflight_failed",
        "read_only": True,
        "smoke_case_count": len(rows),
        "dataset_case_counts": dict(sorted(distribution.items())),
        "case_identity_match": identity_ok,
        "assets": assets,
        "model_assets": model_assets,
        "api_secret_available": secret_ok,
        "api_secret_source": secret_source,
        "api_secret_value_recorded": False,
        "vlm_calls": 0,
        "medsam_calls": 0,
        "kem_endpoints": 0,
        "failures": failures,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n",
                           encoding="utf-8")
    print(json.dumps(audit, indent=2, sort_keys=True))
    if failures:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
