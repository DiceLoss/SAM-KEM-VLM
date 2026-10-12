"""Select the frozen subject-balanced Stage 7 secondary-ablation cases."""
from __future__ import annotations

import argparse
import csv
import hashlib
from collections import defaultdict
from pathlib import Path

CELLS = (
    ("brainweb", 100000),
    ("hecktor", 100000), ("hecktor", 200000),
    ("hecktor", 500000), ("hecktor", 1000000),
)
SEED = 20261012


def rank(value: str) -> str:
    return hashlib.sha256(f"{SEED}:{value}".encode()).hexdigest()


def select(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    grouped: dict[tuple[str, int, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        cell = row["dataset"], int(row["count_level"])
        if row["split"] == "validation" and cell in CELLS:
            grouped[cell + (row["subject_id"],)].append(row)
    selected = []
    for dataset, count in CELLS:
        subjects = sorted(
            {key[2] for key in grouped if key[:2] == (dataset, count)},
            key=lambda subject: rank(f"subject:{dataset}:{count}:{subject}"),
        )[:4]
        if len(subjects) != 4:
            raise ValueError(f"cell {(dataset, count)} has fewer than four subjects")
        for subject in subjects:
            candidates = grouped[(dataset, count, subject)]
            chosen = min(candidates, key=lambda row: rank(
                f"case:{dataset}:{count}:{subject}:{row['slice_id']}:{row['realization_id']}"
            ))
            selected.append({
                "dataset": dataset, "subject_id": subject,
                "slice_id": chosen["slice_id"], "count_level": count,
                "realization_id": int(chosen["realization_id"]),
                "mlem10_input_path": chosen["mlem10_input_path"],
                "target_path": chosen["target_path"],
                "sinogram_path": chosen["sinogram_path"],
                "projection_scale": chosen["projection_scale"],
            })
    if len(selected) != 20:
        raise AssertionError("secondary subset must contain exactly 20 cases")
    return selected


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    rows = list(csv.DictReader(args.manifest.open(newline="", encoding="utf-8")))
    chosen = select(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(chosen[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(chosen)


if __name__ == "__main__":
    main()
