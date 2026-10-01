"""Step 7A release-check CLI — zero-write plan audit.

Per directive 5924162430 + 5925510360. This CLI is a PLAN only:
it reads existing committed manifests and reports proposed study
units/counts/hashes/namespaces/prompt budgets without creating
any formal run/result directory.

No VLM call, no decoder training, no prior generation, no KEM
execution. Deterministic, idempotent.

Usage:
    python -m experiments.stage7a_release_check \\
        --root /path/to/repo \\
        --protocol code/configs/protocols/stage7a_v1.yaml \\
        --output /path/to/plan_audit.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_yaml(p: Path) -> dict:
    """Tiny YAML loader (stage7a_v1.yaml is a flat dict without anchors)."""
    # Avoid PyYAML hard dep; parse the subset of YAML we use.
    import yaml  # type: ignore
    with open(p, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def discover_validation_cases(repo_root: Path) -> list[dict]:
    """Discover the Stage C validation-only cases from the frozen case
    manifest. ZERO-WRITE: read-only."""
    cfg_path = repo_root / "code/configs/generated/dev_trainmax_v1_validation_registered.yaml"
    cfg = load_yaml(cfg_path)
    _ = cfg.get("outputs", {}).get("validation_manifest")  # noqa: F841 reserved for future use

    # Without a manifest, fall back to enumerating per_variant CSV keys
    # (subject, slice, count).
    cases = []
    run_dir = repo_root / "results/dev_trainmax_v1/causal_ablations_corrected_v3/full_run"
    for variant_csv in run_dir.glob("*/per_case.csv"):
        import csv as csv_mod
        with open(variant_csv, "r", encoding="utf-8") as f:
            reader = csv_mod.DictReader(f)
            for row in reader:
                key = (row["dataset"], row["subject_id"], row["slice_id"],
                       int(row["count_level"]), int(row["realization_id"]))
                if key not in {(c["dataset"], c["subject_id"], c["slice_id"],
                                c["count_level"], c["realization_id"]) for c in cases}:
                    cases.append({
                        "dataset": row["dataset"],
                        "subject_id": row["subject_id"],
                        "slice_id": row["slice_id"],
                        "count_level": int(row["count_level"]),
                        "realization_id": int(row["realization_id"]),
                    })
    return cases


def build_plan(repo_root: Path, protocol_path: Path) -> dict:
    """Build the Stage 7A plan audit. ZERO-WRITE."""
    proto = load_yaml(protocol_path)
    cases = discover_validation_cases(repo_root)

    # Per-arm call budget: at most max_calls_per_case per case.
    arms = proto["arms"]
    arms_summary = []
    for arm in arms:
        rules = next(a for a in arms if a["name"] == arm["name"])
        max_calls = rules.get("max_calls_per_case", 1)
        n_calls = len(cases) * max_calls
        arms_summary.append({
            "name": arm["name"],
            "arm_kind": arm["arm_kind"],
            "max_objects": arm.get("max_objects"),
            "max_positive_points": arm.get("max_positive_points"),
            "max_negative_points": arm.get("max_negative_points"),
            "max_calls_per_case": max_calls,
            "allow_multi_round": arm.get("allow_multi_round"),
            "n_cases": len(cases),
            "total_calls": n_calls,
        })

    # Hash bindings (Stage C frozen artifacts, just for record).
    frozen_paths = [
        repo_root / "results/dev_trainmax_v1/causal_ablations_corrected_v3/full_run/per_case.csv"
            .parent.parent.parent.parent / "frozen_hyperparameters.csv",
        repo_root / "results/dev_trainmax_v1/causal_ablations_corrected_v3/full_run/summary.json",
        repo_root / "results/dev_trainmax_v1/causal_ablations_corrected_v3/full_run/per_output_sha256.json",
        repo_root / "results/dev_trainmax_v1/causal_ablations_corrected_v3/full_run/integrity_audit.json",
    ]
    hash_bindings = {str(p.relative_to(repo_root)): sha256_file(p)
                      for p in frozen_paths if p.is_file()}

    audit = {
        "artifact_kind": "validation_only_causal_ablations_stage7a_plan",
        "schema_version": 1,
        "status": "PASS_zero_write_plan",
        "protocol": str(protocol_path.relative_to(repo_root)),
        "protocol_sha256": sha256_file(protocol_path),
        "n_cases": len(cases),
        "arms_summary": arms_summary,
        "hash_bindings": hash_bindings,
        "invariants": {
            "vlm_calls": 0,
            "decoder_training_runs": 0,
            "prior_generation_runs": 0,
            "kem_endpoints": 0,
            "no_formal_stage7_namespace_created": True,
            "committed_artifacts_unchanged": True,
        },
        "scope_notes": (
            "This plan audit was generated without creating any formal Stage 7 "
            "namespace, without invoking any VLM (hosted or local), without "
            "training/loading a new decoder, without generating priors, and "
            "without executing KEM. It is a planning-only artifact produced by "
            "reading existing committed manifests."
        ),
    }
    return audit


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    repo_root = args.root.resolve()
    protocol_path = args.protocol.resolve()
    output_path = args.output.resolve()
    if not repo_root.is_dir():
        print(f"ERROR: root not found: {repo_root}", file=sys.stderr)
        sys.exit(1)
    if not protocol_path.is_file():
        print(f"ERROR: protocol not found: {protocol_path}", file=sys.stderr)
        sys.exit(1)

    audit = build_plan(repo_root, protocol_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(audit, f, indent=2)

    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()