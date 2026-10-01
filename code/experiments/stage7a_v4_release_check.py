"""Stage 7A-v4 closure-patch release check CLI.

Per directive 5928360149. Validates:
- Stage 7E = 8,892 cases from source keys (24-cell breakdown).
- Cross-source exact key-set equality across 8 Stage C variant sources.
- All required hashes bound (full HEAD commit, protocol, 8 source CSVs,
  summary, integrity_audit, frozen hyperparam, validation manifest).
- Stage 7D count remains pending (NEVER fabricated).
- FAIL-closed: any invariant violation emits diagnostic audit AND
  exits nonzero.
- No formal Stage 7 namespace created (this CLI is itself zero-write).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_yaml_simple(p):
    import yaml
    with open(p, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def discover_case_keys(csv_path):
    import csv
    keys = set()
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            keys.add((row["dataset"], row["subject_id"], row["slice_id"],
                       int(row["count_level"]), int(row["realization_id"])))
    return keys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audit-tag", type=str, required=True)
    parser.add_argument("--n-cases-7e", type=int, default=8892)
    parser.add_argument("--hash-bindings-from", type=Path, default=None,
                        help="Head commit hash to bind (default: git rev-parse HEAD)")
    args = parser.parse_args()

    repo_root = args.root.resolve()
    protocol_path = args.protocol.resolve()
    output_path = args.output.resolve()
    proto = load_yaml_simple(protocol_path)
    proto_sha = sha256_file(protocol_path)

    run_dir = repo_root / "results/dev_trainmax_v1/causal_ablations_corrected_v3/full_run"
    variants = ["Q_ADAPT", "FULL_ADAPT", "CENTER_ADAPT", "CENTROID_ADAPT",
                "Q_RELEASED", "MLEM_FEATURE_MATCHED",
                "Q_SHUFFLED_MATCHED", "UNIFORM_FEATURE_MATCHED"]
    key_sets = {}
    source_sha = {}
    for v in variants:
        csv = run_dir / v / "per_case.csv"
        if csv.is_file():
            key_sets[v] = discover_case_keys(csv)
            source_sha[v] = sha256_file(csv)
    all_present = len(key_sets) == len(variants)
    key_set_violations = []
    if key_sets:
        ref = next(iter(key_sets.values()))
        for v, ks in key_sets.items():
            if ks != ref:
                key_set_violations.append({"variant": v, "diff_size": len(ks ^ ref)})

    e_counts = proto["stage_7e_full_historical_validation"]["counts"]
    e_realizations = proto["stage_7e_full_historical_validation"]["realizations"]
    e_datasets = proto.get("datasets", [])
    expected_e_n_cases = args.n_cases_7e
    if key_sets:
        ref_keys = next(iter(key_sets.values()))
        stage_7e_keys = [
            k for k in ref_keys
            if k[0] in e_datasets
            and k[3] in e_counts
            and k[4] in e_realizations
        ]
        actual_e_n_cases = len(stage_7e_keys)
    else:
        stage_7e_keys = []
        actual_e_n_cases = 0

    # Counter-based 24-cell breakdown (per directive §6).
    breakdown = Counter()
    if key_sets:
        for k in stage_7e_keys:
            breakdown[(k[0], k[3], k[4])] += 1
    breakdown_dict = {
        f"{d}/c{c}/r{r}": v for (d, c, r), v in sorted(breakdown.items())
    }

    # All 24 cells (2 datasets * 4 counts * 3 realizations) must equal
    # BW=696 and HE=45.
    expected_cells = {(d, c, r): v for d in e_datasets
                                       for c in e_counts
                                       for r in e_realizations
                                       for v in [696 if d == "brainweb" else 45]}
    breakdown_match = breakdown_dict == expected_cells

    # Hash bindings
    hash_bindings = {}
    for k in ["frozen_hyperparameters.csv",
              "validation_case_manifest.csv"]:
        p = repo_root / "results/dev_trainmax_v1" / k
        if p.is_file():
            hash_bindings[k] = sha256_file(p)
    for v in variants:
        hash_bindings[f"causal_ablations_corrected_v3/full_run/{v}/per_case.csv"] = \
            source_sha.get(v, "")
    for kind in ("summary.json", "integrity_audit.json"):
        p = run_dir / kind
        if p.is_file():
            hash_bindings[f"causal_ablations_corrected_v3/full_run/{kind}"] = sha256_file(p)
    # HEAD commit
    head_hash = ""
    if args.hash_bindings_from and args.hash_bindings_from.is_file():
        head_hash = args.hash_bindings_from.read_text().strip()
    else:
        import subprocess
        out = subprocess.run(["git", "-C", str(repo_root), "rev-parse", "HEAD"],
                              capture_output=True, text=True)
        if out.returncode == 0:
            head_hash = out.stdout.strip()

    # Bind head commit, protocol
    hash_bindings["protocol"] = proto_sha
    hash_bindings["head_commit"] = head_hash

    invariants_pass = (
        all_present
        and not key_set_violations
        and actual_e_n_cases == expected_e_n_cases
        and breakdown_match
        and bool(head_hash)
        and all(hash_bindings.get(k) for k in (
            "frozen_hyperparameters.csv", "validation_case_manifest.csv",
            "head_commit", "protocol",
            "causal_ablations_corrected_v3/full_run/summary.json",
            "causal_ablations_corrected_v3/full_run/integrity_audit.json"))
    )

    arm_summaries = []
    for arm in proto["arms"]:
        arm_summaries.append({
            "name": arm["name"],
            "arm_kind": arm["arm_kind"],
            "max_vlm_calls_per_case": arm["max_calls_per_case"],
            "max_medsam_calls_per_case": (
                arm["arm_kind"] in ("deterministic_full", "deterministic_components")
                and 0 or arm["max_calls_per_case"]),
            "n_cases_7e": actual_e_n_cases,
            "total_calls_7e": actual_e_n_cases * arm["max_calls_per_case"],
            "total_vlm_calls_7e": (
                actual_e_n_cases * arm["max_calls_per_case"]
                if arm["arm_kind"].startswith("vlm") else 0),
        })

    audit = {
        "artifact_kind": f"validation_only_causal_ablations_stage7a_{args.audit_tag}",
        "schema_version": 1,
        "status": "PASS_zero_write_plan" if invariants_pass
                                       else "FAIL_invariant_violation",
        "protocol": str(protocol_path.relative_to(repo_root)),
        "protocol_sha256": proto_sha,
        "head_commit_sha256": head_hash,
        "phase_counts": {
            "stage_7d_internal_development_screen": {
                "counts": proto["stage_7d_internal_development_screen"]["counts"],
                "realizations": proto["stage_7d_internal_development_screen"]["realizations"],
                "expected_n_cases": "pending_stage7b_subject_split",
                "actual_n_cases_computed": 0,
            },
            "stage_7e_full_historical_validation": {
                "counts": e_counts,
                "realizations": e_realizations,
                "expected_n_cases": expected_e_n_cases,
                "actual_n_cases_7e": actual_e_n_cases,
                "matches_expected": actual_e_n_cases == expected_e_n_cases,
            },
            "stage_7e_breakdown": breakdown_dict,
            "stage_7e_breakdown_matches_expected": breakdown_match,
        },
        "arm_summaries": arm_summaries,
        "invariants_actually_checked": {
            "all_8_variants_present": all_present,
            "key_set_violations": key_set_violations,
            "stage_7e_count_matches_expected":
                actual_e_n_cases == expected_e_n_cases,
            "stage_7e_breakdown_matches_expected": breakdown_match,
            "head_commit_bound": bool(head_hash),
            "all_required_hash_bindings_present": all(
                hash_bindings.get(k) for k in (
                    "frozen_hyperparameters.csv", "validation_case_manifest.csv",
                    "head_commit", "protocol",
                    "causal_ablations_corrected_v3/full_run/summary.json",
                    "causal_ablations_corrected_v3/full_run/integrity_audit.json")),
            "hash_bindings_count": len(hash_bindings),
        },
        "hash_bindings": hash_bindings,
        "counter_separation": {
            "vlm_calls_per_arm": {a["name"]: a["total_vlm_calls_7e"]
                                    for a in arm_summaries},
            "medsam_calls_per_arm_max": {
                a["name"]: a["max_medsam_calls_per_case"]
                for a in arm_summaries},
            "executed_calls_in_stage7a": {
                "vlm_calls": 0, "decoder_training_runs": 0,
                "prior_generation_runs": 0, "kem_endpoints": 0,
            },
        },
        "scope_notes": (
            "Stage 7A-v4 closure-patch release check. No VLM call, no "
            "decoder training, no prior generation, no KEM execution. "
            "Stage 7E = 8,892 validated directly from source keys; Stage 7D "
            "count is pending (never fabricated). Cross-source exact "
            "key-set equality assertion is enforced. All required hash "
            "bindings (head commit, protocol, 8 source CSVs, summary, "
            "integrity_audit, frozen hyperparams, validation manifest) "
            "are required. Full 24-cell breakdown (BW=696, HE=45 per "
            "dataset × count × realization cell) must match."
        ),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(audit, f, indent=2)
    # FAIL-closed: any invariant violation -> nonzero exit
    print(json.dumps(audit, indent=2))
    sys.exit(0 if invariants_pass else 1)


if __name__ == "__main__":
    main()