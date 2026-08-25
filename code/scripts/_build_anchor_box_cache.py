"""Phase 4: build the anchor-box cache from r=0 prompt records.

The cache is keyed by `(dataset, subject_id, slice_id, count_level)`
and stores the box that downstream methods must reuse for r=1, r=2.

Inputs:  metrics/prompt_records.csv (444 rows; AUDIT_ONLY marked) +
         the per-record JSON files in prompts/brainweb/ and prompts/hecktor/.
Outputs:
  - prompts/anchor_box_lookup.csv
  - prompts/anchor_box_audit.json

Notes on coverage:
  - The original prompt records were generated for the OLD test_locked
    subjects (subject41/42/43 for BW; 34 HE subjects).
  - After Phase 2's split swap, those subjects are now in NEW validation
    rather than NEW test_locked. The v3 canonical rerun on NEW test_locked
    must therefore use the fallback centroid box for subjects without an
    anchor. This is documented in the audit JSON so the validator and
    v3 run script can apply the fallback rule consistently.
"""
from __future__ import annotations

import csv
import json
import re
from pathlib import Path

ROOT = Path(r"D:/MIAProject/SAM_KEM_QWEN3_MEDSAM2_DELIVERY")
PROMPT_RECORDS = ROOT / "metrics" / "prompt_records.csv"
LOOKUP = ROOT / "prompts" / "anchor_box_lookup.csv"
AUDIT = ROOT / "prompts" / "anchor_box_audit.json"
SPLITS = ROOT / "splits" / "subject_split.csv"
PROMPTS_BW = ROOT / "prompts" / "brainweb"
PROMPTS_HE = ROOT / "prompts" / "hecktor"


COUNT_RE = re.compile(r"_(1000000|100000|200000|500000)_")


def count_of(case_id: str) -> int | None:
    m = COUNT_RE.search(case_id)
    return int(m.group(1)) if m else None


def load_box_from_json(json_path: Path) -> tuple[int, int, int, int] | None:
    """Read a per-record JSON prompt file and extract the box."""
    try:
        d = json.loads(json_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if d.get("parse_status") != "success":
        return None
    box = d.get("box")
    if not isinstance(box, dict):
        return None
    try:
        return (
            int(box["xmin"]), int(box["ymin"]),
            int(box["xmax"]), int(box["ymax"]),
        )
    except Exception:
        return None


def json_path_for(ds: str, sid: str, c: int, r_idx: int) -> Path:
    folder = PROMPTS_BW if ds == "brainweb" else PROMPTS_HE
    # Filenames are like:
    #   brainweb_subject41_axial_center_100000_anchor_0_0.json
    #   hecktor_107_axial_center_100000_anchor_0_0.json
    if ds == "brainweb":
        name = f"{ds}_{sid}_axial_center_{c}_anchor_{r_idx}_0.json"
    else:
        name = f"{ds}_{sid}_axial_center_{c}_anchor_{r_idx}_0.json"
    return folder / name


def main() -> int:
    rows = list(csv.DictReader(open(PROMPT_RECORDS, encoding="utf-8")))
    r0 = [r for r in rows if r["realization_id"] == "0"]
    print(f"prompt records: total={len(rows)}, r=0={len(r0)}")

    # Build the lookup table by reading each r=0 JSON file.
    fields = [
        "dataset", "subject_id", "slice_id", "count_level",
        "xmin", "ymin", "xmax", "ymax",
        "source_request_id", "source_image_sha256", "source_latency_s",
        "source_json_path",
    ]
    out_rows: list[dict] = []
    fallback_count = 0
    for r in r0:
        ds = r["dataset"]
        sid = r["subject_id"]
        sl = r["slice_id"]
        c = count_of(r["case_id"])
        if c is None:
            fallback_count += 1
            continue
        jp = json_path_for(ds, sid, c, 0)
        if not jp.is_file():
            fallback_count += 1
            continue
        box = load_box_from_json(jp)
        if box is None:
            fallback_count += 1
            continue
        out_rows.append({
            "dataset": ds, "subject_id": sid, "slice_id": sl,
            "count_level": str(c),
            "xmin": box[0], "ymin": box[1], "xmax": box[2], "ymax": box[3],
            "source_request_id": r["request_id"],
            "source_image_sha256": r["image_sha256"],
            "source_latency_s": r.get("latency_s", ""),
            "source_json_path": str(jp.relative_to(ROOT)).replace("\\", "/"),
        })

    parse_counts: dict[str, int] = {}
    for r in r0:
        s = r.get("parse_status", "unknown")
        parse_counts[s] = parse_counts.get(s, 0) + 1

    with open(LOOKUP, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(out_rows)
    print(f"wrote: {LOOKUP.relative_to(ROOT)} ({len(out_rows)} rows)")

    # Coverage report.
    split_rows = list(csv.DictReader(open(SPLITS, encoding="utf-8")))
    new_test_subjects = {
        (r["dataset"], r["subject_id"]) for r in split_rows
        if r["split"] == "test_locked"
    }
    new_val_subjects = {
        (r["dataset"], r["subject_id"]) for r in split_rows
        if r["split"] == "validation"
    }
    covered = {(r["dataset"], r["subject_id"]) for r in out_rows}
    missing_in_test = sorted(
        (ds, sid) for ds, sid in new_test_subjects
        if (ds, sid) not in covered
    )
    missing_in_val = sorted(
        (ds, sid) for ds, sid in new_val_subjects
        if (ds, sid) not in covered
    )

    bw_anchors = [r for r in out_rows if r["dataset"] == "brainweb"]
    he_anchors = [r for r in out_rows if r["dataset"] == "hecktor"]
    audit = {
        "anchor_count_total": len(out_rows),
        "anchor_count_brainweb": len(bw_anchors),
        "anchor_count_hecktor": len(he_anchors),
        "parse_status_counts": parse_counts,
        "fallback_count": fallback_count,
        "new_test_locked_subjects_total": len(new_test_subjects),
        "new_test_locked_subjects_covered": len(new_test_subjects) - len(missing_in_test),
        "new_test_locked_subjects_missing": missing_in_test,
        "new_validation_subjects_total": len(new_val_subjects),
        "new_validation_subjects_covered": len(new_val_subjects) - len(missing_in_val),
        "new_validation_subjects_missing": missing_in_val,
        "note": (
            "Prompts were generated only for the OLD test_locked subjects. "
            "After Phase 2 split swap, those subjects are in NEW validation, "
            "not NEW test_locked. The v3 canonical rerun on NEW test_locked "
            "subjects must therefore use the fallback centroid box for "
            "subjects not in this lookup. This is the same fallback rule "
            "documented in configs/headline_qwen3_medsam2.yaml::vlm.fallback."
        ),
        "fallback_rule": (
            "intensity_centroid_box_recorded_as_failure; "
            "fallback_centroid_source=low_count_mlem_initialization; "
            "fallback_box_width_height=32x32"
        ),
        "zero_new_qwen3_calls": True,
    }
    AUDIT.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(f"wrote: {AUDIT.relative_to(ROOT)}")
    print(
        f"new test_locked coverage: "
        f"{audit['new_test_locked_subjects_covered']}/"
        f"{audit['new_test_locked_subjects_total']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())