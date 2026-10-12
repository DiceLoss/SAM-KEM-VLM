"""Generate the preregistered Stage 7 subject-level analysis package."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

from sam_kem.stage7_formal_analysis import (
    analyse_subject_cells,
    paired_case_deltas,
    subject_deltas,
)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoints", required=True, type=Path)
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite analysis namespace: {args.output}")
    rows = list(csv.DictReader(args.endpoints.open(newline="", encoding="utf-8")))
    cases = paired_case_deltas(rows)
    subjects = subject_deltas(cases)
    cells = analyse_subject_cells(subjects)
    args.output.mkdir(parents=True)
    write_csv(args.output / "paired_case_deltas.csv", cases)
    write_csv(args.output / "paired_subject_deltas.csv", subjects)
    write_csv(args.output / "cell_inference.csv", cells)
    audit = {
        "artifact_kind": "stage7_formal_preregistered_analysis",
        "status": "PASS_complete",
        "input_endpoints_sha256": sha256_file(args.endpoints),
        "protocol_sha256": sha256_file(args.protocol),
        "paired_case_count": len(cases), "subject_row_count": len(subjects),
        "cell_metric_count": len(cells), "bootstrap_samples": 10000,
        "bootstrap_seed": 20261012, "sign_flip_seed": 20261012,
        "monte_carlo_sign_flips": 100000,
        "holm_family_size": 10,
        "outputs": {
            name: sha256_file(args.output / name)
            for name in (
                "paired_case_deltas.csv", "paired_subject_deltas.csv",
                "cell_inference.csv",
            )
        },
    }
    (args.output / "analysis_audit.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8",
    )
    print(json.dumps(audit, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
