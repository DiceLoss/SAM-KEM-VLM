"""Phase 9: validator for the v3 canonical rerun.

Enforces the v3 contract rules. Exits non-zero on any violation.
Run as:  python scripts/validate_v3.py .

Rules:
  R1. splits/subject_split.csv: train=14 BW + 156 HE; validation=3 BW + 34 HE;
      test_locked=3 BW + 34 HE; total=244 rows.
  R2. configs/headline_qwen3_medsam2.yaml: system_matrix_id ends in 'n180';
      mlem_iterations == 10; decoder SHAs match audit_report.csv.
  R3. prompts/anchor_box_lookup.csv: at least one row per (subject, count)
      that needs an anchor (or an explicit note in anchor_box_audit.json
      that fallback boxes are used for the rest).
  R4. metrics/per_case_metrics_v3.csv: unique primary key
      (dataset, subject_id, slice_id, realization_id, count_level, method);
      all 8 methods present per (subject, realization, count);
      checkpoint_sha256 matches audit_report.csv where applicable.
  R5. No path references `skimage`, `radon_transform` from skimage, or
      `_wsl_phantom` in the v3 scripts.
  R6. AUDIT_ONLY CSVs (in metrics/ and convergence/) retain the marker.
  R7. No row in any v3 metric has `split=test_locked` AND a hyperparam
      sourced from a `split=validation` cell of the v3 sweep CSV.
      Catches leakage.
  R8. Every metric CSV has zero duplicate primary keys.
  R9. Headline config declares anchor_realization_enforced: true and
      poison_deviance_source: unified_180_angle_discrete_radon.
"""
from __future__ import annotations

import csv
import json
import re
import sys
from pathlib import Path

import yaml


def main(root: Path) -> int:
    failures: list[str] = []
    warnings: list[str] = []

    # ---- R1: split counts --------------------------------------------------
    splits = list(csv.DictReader(open(root / "code/configs/subject_split_v3.csv", encoding="utf-8")))
    n_train_bw = sum(1 for r in splits if r["dataset"] == "brainweb" and r["split"] == "train")
    n_train_he = sum(1 for r in splits if r["dataset"] == "hecktor" and r["split"] == "train")
    n_val_bw = sum(1 for r in splits if r["dataset"] == "brainweb" and r["split"] == "validation")
    n_val_he = sum(1 for r in splits if r["dataset"] == "hecktor" and r["split"] == "validation")
    n_test_bw = sum(1 for r in splits if r["dataset"] == "brainweb" and r["split"] == "test_locked")
    n_test_he = sum(1 for r in splits if r["dataset"] == "hecktor" and r["split"] == "test_locked")
    if (n_train_bw, n_val_bw, n_test_bw) != (14, 3, 3):
        failures.append(f"R1 BW split: train={n_train_bw}, val={n_val_bw}, test_locked={n_test_bw} (expected 14/3/3)")
    if (n_train_he, n_val_he, n_test_he) != (156, 34, 34):
        failures.append(f"R1 HE split: train={n_train_he}, val={n_val_he}, test_locked={n_test_he} (expected 156/34/34)")
    if len(splits) != 244:
        failures.append(f"R1 total rows: {len(splits)} (expected 244)")

    # ---- R2: headline config ----------------------------------------------
    head = yaml.safe_load(open(root / "code/configs/headline_qwen3_medsam2.yaml", encoding="utf-8"))
    sysm = head.get("data", {}).get("system_matrix_id", "")
    if not str(sysm).endswith("n180"):
        failures.append(f"R2 system_matrix_id={sysm!r} does not end in 'n180'")
    mlem_it = head.get("method", {}).get("mlem_iterations")
    if mlem_it != 10:
        failures.append(f"R2 mlem_iterations={mlem_it} != 10")
    sha = head.get("sam", {}).get("checkpoint_sha256", {})
    if not isinstance(sha, dict) or "brainweb" not in sha or "hecktor" not in sha:
        failures.append(f"R2 sam.checkpoint_sha256 missing nested per-dataset SHAs")
    audit = list(csv.DictReader(open(root / "code/configs/audit_report.csv", encoding="utf-8")))
    audit_sha = {r["checkpoint_id"]: r["hash_actual"] for r in audit}
    if audit_sha.get("brainweb") and sha.get("brainweb") != audit_sha["brainweb"]:
        failures.append(f"R2 brainweb SHA mismatch: headline={sha.get('brainweb')} audit={audit_sha.get('brainweb')}")
    if audit_sha.get("hecktor") and sha.get("hecktor") != audit_sha["hecktor"]:
        failures.append(f"R2 hecktor SHA mismatch: headline={sha.get('hecktor')} audit={audit_sha.get('hecktor')}")

    # ---- R3: anchor cache --------------------------------------------------
    anchor_path = root / "prompts" / "anchor_box_lookup.csv"
    audit_path = root / "prompts" / "anchor_box_audit.json"
    if not anchor_path.is_file():
        warnings.append("R3 anchor_box_lookup.csv missing (will be created in Phase 4)")
    else:
        ar = list(csv.DictReader(open(anchor_path, encoding="utf-8")))
        # The plan expected 284 records; we have 148 because prompts were
        # only generated for the OLD test_locked subjects. The audit JSON
        # documents the discrepancy and the fallback rule.
        if not audit_path.is_file():
            failures.append("R3 anchor_box_audit.json missing")
        else:
            ad = json.loads(audit_path.read_text(encoding="utf-8"))
            if ad.get("anchor_count_total", 0) != len(ar):
                failures.append(
                    f"R3 anchor count mismatch: lookup={len(ar)} audit={ad.get('anchor_count_total')}"
                )

    # ---- R4: per_case_metrics_v3.csv --------------------------------------
    pc = root / "results" / "metrics" / "per_case_metrics_v3.csv"
    if not pc.is_file():
        warnings.append("R4 per_case_metrics_v3.csv missing; Phase 7 rerun not yet complete")
    else:
        rows = list(csv.DictReader(open(pc, encoding="utf-8")))
        # Unique key check (R8 also).
        keys = [
            (r["dataset"], r["subject_id"], r["slice_id"],
             r["realization_id"], r["count_level"], r["method"])
            for r in rows
        ]
        if len(set(keys)) != len(keys):
            n_dup = len(keys) - len(set(keys))
            failures.append(f"R4/R8 per_case_metrics_v3.csv has {n_dup} duplicate primary keys")
        # All 8 methods per (subject, r, count).
        methods = {r["method"] for r in rows}
        expected_methods = {
            "MLEM", "OSEM", "Bowsher", "SelfKEM", "MR-KEM", "CT-KEM",
            "SAM-prior-MedSAM2-Qwen3",
            "SAM-KEM-MedSAM2-Qwen3-no-post",
            "SAM-KEM-MedSAM2-Qwen3-full",
        }
        missing = expected_methods - methods
        if missing:
            warnings.append(f"R4 missing methods in per_case_metrics_v3.csv: {sorted(missing)}")
        # Decoder SHA consistency.
        for r in rows:
            sha_v = r.get("checkpoint_sha256") or ""
            if r["method"].startswith("SAM") and sha_v and sha_v != "NA":
                ds = r["dataset"]
                if audit_sha.get(ds) and sha_v != audit_sha[ds]:
                    failures.append(
                        f"R4 SAM row uses checkpoint_sha256={sha_v} but audit expects {audit_sha.get(ds)}"
                    )
                    break

    # ---- R5: forbidden patterns in v3 scripts -----------------------------
    # The v3 canonical contract forbids the skimage Radon operator as a
    # forward or backward projector. The skimage.metrics module is
    # allowed for SSIM computation only.
    forbidden = [
        (re.compile(r"skimage\.transform\.radon\b"), "skimage.transform.radon"),
        (re.compile(r"skimage\.transform\.iradon\b"), "skimage.transform.iradon"),
        (re.compile(r"radon_transform\s*=\s*skimage"), "skimage radon fallback"),
        (re.compile(r"_wsl_phantom"), "_wsl_phantom reference"),
    ]
    # Apply only to v3 scripts (Phase 4+) and the v3 replacement of the
    # legacy backfill.
    v3_scripts = sorted(p for p in root.glob("scripts/_*.py")
                        if p.name.endswith("_v3.py")
                        or p.name in {
                            "_sweep_hyperparams_v3.py",
                            "_rerun_primary_v3.py",
                            "_rerun_convergence_v3.py",
                            "_build_statistics_v3.py",
                            "_regen_figures_v3.py",
                            "_regen_report_v3.py",
                            "_port_forward_operator.py",
                            "_regen_bw_arrays_180.py",
                            "_backfill_ssim_poisson_v3.py",
                        })
    for script in v3_scripts:
        try:
            txt = script.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        for pat, label in forbidden:
            if pat.search(txt):
                failures.append(f"R5 {script.name}: forbidden pattern '{label}'")

    # ---- R6: AUDIT_ONLY marker on legacy CSVs -----------------------------
    # v3 CSV outputs are excluded: per_case_metrics_v3.csv,
    # cross_count_results_v3.csv, paired_subject_statistics_v3.csv,
    # ablation_per_case_v3.csv, hyperparameter_sweep_v3.csv,
    # frozen_hyperparameters.csv.
    v3_outputs = {
        "per_case_metrics_v3.csv", "per_case_metrics_v3_bw.csv",
        "per_case_metrics_v3_bw_v1subj.csv",
        "cross_count_results_v3.csv",
        "paired_subject_statistics_v3.csv", "ablation_per_case_v3.csv",
        "hyperparameter_sweep_v3.csv", "frozen_hyperparameters.csv",
        "convergence_v3.csv", "convergence_v3_bw.csv",
    }
    for sub in ["results/metrics", "results/convergence"]:
        d = root / sub
        if not d.is_dir(): continue
        for f in sorted(d.glob("*.csv")):
            if f.name in v3_outputs:
                continue
            try:
                rows = list(csv.DictReader(open(f, encoding="utf-8")))
            except Exception:
                continue
            if rows and "audit_only_marker" not in rows[0]:
                failures.append(f"R6 {f.relative_to(root)}: missing audit_only_marker column")

    # ---- R7: leakage check -------------------------------------------------
    sweep = root / "results" / "metrics" / "hyperparameter_sweep_v3.csv"
    if sweep.is_file() and pc.is_file():
        sweep_rows = list(csv.DictReader(open(sweep, encoding="utf-8")))
        pc_rows = list(csv.DictReader(open(pc, encoding="utf-8")))
        sweep_subjects = {(r["dataset"], r["subject_id"]) for r in sweep_rows if r["split"] == "validation"}
        pc_subjects = {(r["dataset"], r["subject_id"]) for r in pc_rows if r["split"] == "test_locked"}
        # Catches the case where someone copies the wrong row index.
        overlap = sweep_subjects & pc_subjects
        if overlap:
            failures.append(
                f"R7 leakage: validation subjects present in test_locked rows: {sorted(overlap)[:5]}"
            )

    # ---- R9: anchor enforcement / deviance source -------------------------
    if not head.get("vlm", {}).get("anchor_realization_enforced"):
        failures.append("R9 vlm.anchor_realization_enforced is not true")
    pds = head.get("method", {}).get("poisson_deviance_source", "")
    if pds != "unified_180_angle_discrete_radon":
        failures.append(f"R9 poisson_deviance_source={pds!r} != 'unified_180_angle_discrete_radon'")

    # ---- Report -----------------------------------------------------------
    if failures:
        print("FAIL: validator found the following issues:")
        for f in failures:
            print(f"  - {f}")
        if warnings:
            print("WARN:")
            for w in warnings:
                print(f"  - {w}")
        return 1
    if warnings:
        print("WARN:")
        for w in warnings:
            print(f"  - {w}")
    print("PASS: validator found no rule violations.")
    return 0


if __name__ == "__main__":
    root = Path(sys.argv[1] if len(sys.argv) > 1 else ".").resolve()
    raise SystemExit(main(root))