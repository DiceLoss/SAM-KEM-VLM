"""Build a per-case manifest from per_case_metrics_v3.csv.

Outputs:
- results/metrics/case_manifest.csv with one row per
  (dataset, subject_id, slice_id, realization_id, count_level).
  Columns include: case_id, dataset, subject_id, slice_id,
  realization_id, count_level, split, checkpoint_sha256, seed,
  scale_constant_C, operator_id, anchor_box, alpha, sigma, J,
  n_kem_iters, mlem_iters.

This is the file reviewers can use to reproduce any single case
end-to-end.
"""
import csv
import hashlib
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(r"D:/MIAProject/SAM_KEM_QWEN3_V3_REPO")
PER_CASE = ROOT / "results" / "metrics" / "per_case_metrics_v3.csv"
FROZEN = ROOT / "results" / "metrics" / "frozen_hyperparameters.csv"
AUDIT = ROOT / "code" / "configs" / "audit_report.csv"
ANCHORS = ROOT / "prompts" / "anchor_box_lookup.csv"
HEAD = ROOT / "code" / "configs" / "headline_qwen3_medsam2.yaml"
OUT = ROOT / "results" / "metrics" / "case_manifest.csv"

SEED_BASE = 20260817  # per_case_metrics_v3.csv seed


def seed_for(dataset, subject_id, slice_id, realization, count):
    h = hashlib.sha256(
        f"{dataset}|{subject_id}|{slice_id}|{realization}|{count}|v3_180".encode()
    ).hexdigest()
    return SEED_BASE + (int(h, 16) % (2**31))


def main():
    cfg = yaml.safe_load(open(HEAD, encoding="utf-8"))
    scale_c = cfg.get("inputs", {}).get("scale_constant_C", None)
    operator_id = cfg.get("method", {}).get("forward_operator", None)
    sigma_global = cfg.get("kem", {}).get("sigma", None)
    j_global = cfg.get("kem", {}).get("J", None)
    n_kem_global = cfg.get("kem", {}).get("selected_iteration", None)

    # Frozen per-(dataset, count) hyperparameters.
    frozen = {}
    if FROZEN.is_file():
        for r in csv.DictReader(open(FROZEN, encoding="utf-8")):
            frozen[(r["dataset"], int(r["count_level"]))] = r

    # Decoder SHA per dataset.
    sha = {}
    if AUDIT.is_file():
        for r in csv.DictReader(open(AUDIT, encoding="utf-8")):
            sha[r["checkpoint_id"]] = r["hash_actual"]

    # Anchor boxes.
    anchors = {}
    if ANCHORS.is_file():
        for r in csv.DictReader(open(ANCHORS, encoding="utf-8")):
            anchors[(r["dataset"], r["subject_id"], r["slice_id"], int(r["count_level"]))] = r

    pc_rows = list(csv.DictReader(open(PER_CASE, encoding="utf-8")))
    by_case: dict[tuple, dict] = {}
    for r in pc_rows:
        key = (r["dataset"], r["subject_id"], r["slice_id"],
               r["realization_id"], int(r["count_level"]))
        if key not in by_case:
            by_case[key] = {
                "checkpoint_sha256": r["checkpoint_sha256"],
                "split": r["split"],
            }

    fields = ["case_id", "dataset", "subject_id", "slice_id",
              "realization_id", "count_level", "split",
              "checkpoint_sha256", "seed",
              "scale_constant_C", "operator_id",
              "alpha_star", "sigma_star", "J_star", "n_kem_star",
              "mlem_iters", "anchor_box_xmin", "anchor_box_ymin",
              "anchor_box_xmax", "anchor_box_ymax",
              "anchor_source"]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(OUT, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for (ds, sid, slice_id, rid, cnt), info in sorted(by_case.items()):
            fr = frozen.get((ds, cnt), {})
            box = anchors.get((ds, sid, slice_id, cnt))
            row = {
                "case_id": f"{ds}_{sid}_{slice_id}_r{rid}_c{cnt}",
                "dataset": ds,
                "subject_id": sid,
                "slice_id": slice_id,
                "realization_id": rid,
                "count_level": cnt,
                "split": info["split"],
                "checkpoint_sha256": info["checkpoint_sha256"],
                "seed": seed_for(ds, sid, slice_id, rid, cnt),
                "scale_constant_C": scale_c,
                "operator_id": operator_id,
                "alpha_star": fr.get("alpha_star", ""),
                "sigma_star": fr.get("sigma_star", sigma_global),
                "J_star": fr.get("J_star", j_global),
                "n_kem_star": fr.get("N_KEM_star", n_kem_global),
                "mlem_iters": cfg.get("method", {}).get("mlem_iterations", ""),
                "anchor_box_xmin": box["xmin"] if box else "",
                "anchor_box_ymin": box["ymin"] if box else "",
                "anchor_box_xmax": box["xmax"] if box else "",
                "anchor_box_ymax": box["ymax"] if box else "",
                "anchor_source": box["source_request_id"] if box else "fallback_centroid",
            }
            w.writerow(row)
            n += 1
    print(f"wrote case_manifest.csv: {n} cases")


if __name__ == "__main__":
    main()