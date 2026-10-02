"""Stage 7A-v4.2 closure-patch release check CLI.

Per directive 5945647996. Validates:
- Stage 7E = 8,892 cases from source keys.
- Cross-source exact key-set equality across all 8 Stage C variant sources.
- 24-cell Counter breakdown (BW=696, HE=45) via tuple-key comparison.
- All required hash bindings compared against pinned expected SHA-256
  values (not merely echoed).
- Stage 7D count remains pending (NEVER fabricated).
- Formal-namespace absence check before and after (snapshot equality).
- Output containment rejected using Path.resolve() ancestry even if
  the output path does not exist yet.
- FAIL-closed: any invariant violation emits diagnostic audit AND
  exits nonzero.
- Required expected code commit verified; clean-tree execution required.
- 8,892 = 696*4 (BW counts) + 45*4 (HE counts); 4 counts × 3 realizations
  = 12 cells × 2 datasets = 24 cells.
- No formal Stage 7 namespace created by this CLI (zero-write).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
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


# Formal Stage 7 execution-output namespaces (prospective only).
# Immutable Stage C inputs (causal_ablations_corrected_v3, etc.) are NOT
# forbidden — they are read-only sources.
FORMAL_NAMESPACES = (
    "results/dev_trainmax_v1/stage7a_run",
    "results/dev_trainmax_v1/stage7a_outputs",
)


def is_under_formal(p: Path) -> str:
    """Resolve p and check if any parent is a formal Stage 7 execution
    namespace. Returns the matching namespace or empty string."""
    try:
        rp = p.resolve()
    except (OSError, RuntimeError):
        rp = p.absolute()
    for ns in FORMAL_NAMESPACES:
        ns_path = (Path.cwd() / ns).resolve()
        try:
            rp.relative_to(ns_path)
            return ns
        except ValueError:
            continue
    return ""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audit-tag", type=str, required=True)
    parser.add_argument("--n-cases-7e", type=int, default=8892)
    parser.add_argument("--expected-head", type=str, default="")
    args = parser.parse_args()

    repo_root = args.root.resolve()
    protocol_path = args.protocol.resolve()
    output_path = args.output.resolve()
    proto = load_yaml_simple(protocol_path)
    proto_sha = sha256_file(protocol_path)

    # Output containment: reject if path resolves under any formal ns
    # EVEN IF THE PATH DOES NOT YET EXIST.
    output_under_formal = is_under_formal(output_path)
    output_violation = bool(output_under_formal)

    # Snapshot formal namespaces before audit (dirs that exist OR don't)
    pre_existing = set()
    for ns in FORMAL_NAMESPACES:
        ns_path = repo_root / ns
        if ns_path.exists():
            pre_existing.add(ns)

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

    # TUPLE-KEY Counter breakdown
    breakdown = Counter()
    for k in stage_7e_keys:
        breakdown[(k[0], k[3], k[4])] += 1
    expected_cells = {(d, c, r): v for d in e_datasets
                                       for c in e_counts
                                       for r in e_realizations
                                       for v in [696 if d == "brainweb" else 45]}
    breakdown_match = dict(breakdown) == expected_cells

    # Hash bindings — pinned expected values
    pinned_hashes = {}
    for k, expected, rel in (
        ("frozen_hyperparameters.csv", proto.get("hash_pins", {})
            .get("frozen_hyperparameters.csv", ""),
         "results/dev_trainmax_v1/frozen_hyperparameters.csv"),
        ("validation_case_manifest.csv", proto.get("hash_pins", {})
            .get("validation_case_manifest.csv", ""),
         "results/dev_trainmax_v1/validation_case_manifest.csv"),
    ):
        p = repo_root / rel
        actual = sha256_file(p) if p.is_file() else ""
        pinned_hashes[k] = {"actual": actual, "expected": expected,
                               "path_present": p.exists(),
                               "match": bool(actual and expected) and actual == expected}
    for v in variants:
        # Allow pins to be stored either with or without the
        # results/dev_trainmax_v1/ prefix.
        key1 = f"causal_ablations_corrected_v3/full_run/{v}/per_case.csv"
        key2 = f"results/dev_trainmax_v1/{key1}"
        expected = proto.get("hash_pins", {}).get(key1, "") or \
                   proto.get("hash_pins", {}).get(key2, "")
        pinned_hashes[key1] = {
            "actual": source_sha.get(v, ""),
            "expected": expected,
            "path_present": bool(source_sha.get(v)),
            "match": bool(expected and source_sha.get(v)) and source_sha.get(v, "") == expected,
        }
    for kind in ("summary.json", "integrity_audit.json"):
        rel = f"causal_ablations_corrected_v3/full_run/{kind}"
        rel2 = f"results/dev_trainmax_v1/{rel}"
        expected = proto.get("hash_pins", {}).get(rel, "") or \
                   proto.get("hash_pins", {}).get(rel2, "")
        p = run_dir / kind
        actual = sha256_file(p) if p.is_file() else ""
        pinned_hashes[rel] = {
            "actual": actual, "expected": expected,
            "path_present": p.is_file(),
            "match": bool(expected and actual) and actual == expected,
        }
    # The protocol pins itself implicitly via the git head commit (the
    # next assertion in invariants_pass). Do NOT pin it as a separate
    # expected hash (chicken-and-egg).
    pinned_hashes["protocol"] = {
        "actual": proto_sha, "expected": "",
        "path_present": True,
        "match": True,  # implicit-pin; covered by head_commit
    }

    # Required commit verification
    head_hash = ""
    head_msg = ""
    try:
        out = subprocess.run(["git", "-C", str(repo_root), "rev-parse", "HEAD"],
                              capture_output=True, text=True)
        if out.returncode == 0:
            head_hash = out.stdout.strip()
    except Exception:
        pass
    try:
        out = subprocess.run(["git", "-C", str(repo_root), "status", "--porcelain"],
                              capture_output=True, text=True)
        if out.returncode == 0:
            head_msg = "clean" if len(out.stdout.strip()) == 0 else "dirty"
    except Exception:
        head_msg = "unknown"
    required_head = args.expected_head or proto.get("required_head_commit", "")
    head_match = bool(required_head) and head_hash == required_head

    # Per-arm summary with explicit counters
    arms = proto["arms"]
    arm_summaries = []
    for arm in arms:
        vlm_max = arm["vlm_calls"]
        medsam_max = arm["medsam_calls"]
        n_cases_7e = actual_e_n_cases
        arm_summaries.append({
            "name": arm["name"],
            "arm_kind": arm["arm_kind"],
            "max_vlm_calls_per_case": vlm_max,
            "max_medsam_calls_per_case": medsam_max,
            "n_cases_7e": n_cases_7e,
            "total_vlm_calls_7e": vlm_max * n_cases_7e,
            "total_medsam_calls_7e": medsam_max * n_cases_7e,
        })

    # ALL hash bindings must match (not just be present)
    all_hash_bindings_match = all(v["match"] for v in pinned_hashes.values())

    invariants_pass = (
        all_present
        and not key_set_violations
        and actual_e_n_cases == expected_e_n_cases
        and breakdown_match
        and bool(head_hash)
        and head_msg == "clean"
        and head_match
        and not output_violation
        and all_hash_bindings_match
        and all(v["path_present"] for v in pinned_hashes.values() if v["expected"])
    )

    # Snapshot after (must equal pre, since we never create directories)
    post_existing = set()
    for ns in FORMAL_NAMESPACES:
        ns_path = repo_root / ns
        if ns_path.exists():
            post_existing.add(ns)
    snapshot_equal = pre_existing == post_existing

    invariants_pass = invariants_pass and snapshot_equal

    audit = {
        "artifact_kind": f"validation_only_causal_ablations_stage7a_{args.audit_tag}",
        "schema_version": 1,
        "status": "PASS_zero_write_plan" if invariants_pass
                                       else "FAIL_invariant_violation",
        "protocol": str(protocol_path.relative_to(repo_root)),
        "protocol_sha256": proto_sha,
        "head_commit_sha256": head_hash,
        "head_tree_status": head_msg,
        "required_head_commit": required_head,
        "head_commit_matches_required": head_match,
        "formal_namespace_snapshot": {
            "checked_namespaces": list(FORMAL_NAMESPACES),
            "pre_existing": sorted(pre_existing),
            "post_existing": sorted(post_existing),
            "snapshot_equal": snapshot_equal,
        },
        "output_containment_check": {
            "output_path": str(output_path),
            "output_under_formal_namespace": output_under_formal,
            "violated": output_violation,
        },
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
            "stage_7e_breakdown": {
                f"{d}/c{c}/r{r}": v for (d, c, r), v in
                sorted(breakdown.items())
            },
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
            "head_tree_clean": head_msg == "clean",
            "head_commit_matches_required": head_match,
            "all_hash_bindings_match_pins": all_hash_bindings_match,
            "output_not_under_formal_namespace": not output_violation,
            "formal_namespace_snapshot_equal": snapshot_equal,
        },
        "hash_bindings": pinned_hashes,
        "counter_separation": {
            "vlm_calls_per_arm_max": {
                a["name"]: a["total_vlm_calls_7e"]
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
            "Stage 7A-v4.2 closure-patch release check. No VLM call, no "
            "decoder training, no prior generation, no KEM execution. "
            "Stage 7E = 8,892 validated directly from source keys; Stage 7D "
            "count is pending (never fabricated). 24-cell Counter breakdown "
            "compared via tuple keys. Cross-source exact key-set equality "
            "assertion. Formal-namespace absence check before/after "
            "(snapshot equality). All required hash bindings compared "
            "against pinned expected SHA-256 values. Output containment "
            "rejected via resolved Path ancestry even if output does not "
            "exist. Required head commit + clean-tree enforced."
        ),
    }

    # Only write the audit if invariants pass (FAIL-closed: do NOT write
    # forbidden or invalid audits). Always write if no formal-ns issue.
    if output_violation:
        sys.stderr.write(
            f"REFUSED: output {output_path} resolves under formal "
            f"namespace {output_under_formal}\n")
        sys.exit(2)
    if not snapshot_equal:
        sys.stderr.write(
            "REFUSED: formal-namespace snapshot changed during audit "
            "(creation forbidden)\n")
        sys.exit(2)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(audit, f, indent=2)
    print(json.dumps(audit, indent=2))
    sys.exit(0 if invariants_pass else 1)


if __name__ == "__main__":
    main()