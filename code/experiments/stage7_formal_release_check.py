"""Zero-write release check for the formal dataset-aware Stage 7 Agent.

This command validates immutable inputs and budgets.  It never calls a VLM,
MedSAM2, or KEM and refuses to write inside a formal execution namespace.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
from collections import Counter
from pathlib import Path

import yaml


def canonical_lf_sha256(path: Path) -> str:
    raw = path.read_bytes()
    if b"\r" in raw:
        normalized = raw.replace(b"\r\n", b"\n")
        if b"\r" in normalized:
            raise ValueError(f"lone CR byte in {path}")
        raw = normalized
    return hashlib.sha256(raw).hexdigest()


def _git(repo: Path, *args: str) -> tuple[int, str]:
    proc = subprocess.run(
        ["git", "-C", str(repo), *args], capture_output=True, text=True,
        check=False,
    )
    return proc.returncode, proc.stdout.strip()


def _resolved_under(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
    except ValueError:
        return False
    return True


def _manifest_counts(path: Path) -> tuple[int, Counter]:
    cells: Counter = Counter()
    keys: set[tuple[str, str, str, int, int]] = set()
    with path.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            key = (
                row["dataset"], row["subject_id"], row["slice_id"],
                int(row["count_level"]), int(row["realization_id"]),
            )
            if key in keys:
                raise ValueError(f"duplicate manifest key: {key}")
            keys.add(key)
            cells[(key[0], key[3], key[4])] += 1
    return len(keys), cells


def build_audit(repo: Path, protocol_path: Path, output: Path,
                expected_head: str) -> dict:
    protocol = yaml.safe_load(protocol_path.read_text(encoding="utf-8"))
    failures: list[str] = []
    checks: dict[str, object] = {}

    namespaces = [repo / item for item in protocol["formal_namespaces"]]
    output_safe = not any(_resolved_under(output, item) for item in namespaces)
    checks["output_outside_formal_namespaces"] = output_safe
    if not output_safe:
        failures.append("audit output is inside a formal namespace")

    absent = {str(item.relative_to(repo)): not item.exists() for item in namespaces}
    checks["formal_namespaces_absent"] = absent
    if not all(absent.values()):
        failures.append("one or more formal namespaces already exist")

    input_records = {}
    for section, records in (
        ("case_manifest", {"case_manifest": protocol["case_manifest"]}),
        ("frozen_hyperparameters", {
            "frozen_hyperparameters": protocol["frozen_hyperparameters"]}),
        ("prompt_templates", protocol["prompt_templates"]),
        ("fixed_case_smoke", {"manifest": {
            "path": protocol["fixed_case_smoke"]["manifest_path"],
            "canonical_lf_sha256": protocol["fixed_case_smoke"][
                "manifest_canonical_lf_sha256"],
        }, "accepted_audit": {
            "path": protocol["fixed_case_smoke"]["accepted_audit_path"],
            "canonical_lf_sha256": protocol["fixed_case_smoke"][
                "accepted_audit_sha256"],
        }}),
    ):
        for name, record in records.items():
            path = repo / record["path"]
            actual = canonical_lf_sha256(path) if path.is_file() else ""
            match = actual == record["canonical_lf_sha256"]
            input_records[f"{section}.{name}"] = {
                "path": record["path"], "present": path.is_file(),
                "actual_canonical_lf_sha256": actual,
                "expected_canonical_lf_sha256": record["canonical_lf_sha256"],
                "match": match,
            }
            if not match:
                failures.append(f"hash mismatch: {section}.{name}")
    checks["immutable_inputs"] = input_records

    smoke_path = repo / protocol["fixed_case_smoke"]["accepted_audit_path"]
    try:
        smoke = json.loads(smoke_path.read_text(encoding="utf-8"))
        smoke_ok = (
            smoke.get("status") == protocol["fixed_case_smoke"]["accepted_status"]
            and smoke.get("case_count") == 4
            and smoke.get("endpoint_count") == 8
            and smoke.get("api_calls") == 4
            and smoke.get("api_retries") == 0
            and smoke.get("fallback_responses") == 0
            and smoke.get("valid_agent_responses_by_dataset")
            == {"brainweb": 2, "hecktor": 2}
            and smoke.get("all_metrics_finite") is True
        )
    except (OSError, ValueError, TypeError) as exc:
        smoke, smoke_ok = {"read_error": str(exc)}, False
    checks["accepted_smoke"] = {
        "match": smoke_ok,
        "status": smoke.get("status"),
        "valid_agent_responses_by_dataset": smoke.get(
            "valid_agent_responses_by_dataset"),
    }
    if not smoke_ok:
        failures.append("accepted fixed-case smoke evidence is invalid")

    manifest_path = repo / protocol["case_manifest"]["path"]
    try:
        n_cases, cells = _manifest_counts(manifest_path)
    except (KeyError, ValueError, OSError) as exc:
        n_cases, cells = 0, Counter()
        failures.append(f"manifest invalid: {exc}")
    expected_cells = {
        (dataset, count, realization): (
            protocol["case_manifest"]["brainweb_rows_per_cell"]
            if dataset == "brainweb"
            else protocol["case_manifest"]["hecktor_rows_per_cell"]
        )
        for dataset in ("brainweb", "hecktor")
        for count in protocol["case_manifest"]["counts"]
        for realization in protocol["case_manifest"]["realizations"]
    }
    manifest_ok = (
        n_cases == protocol["case_manifest"]["expected_cases"]
        and len(cells) == protocol["case_manifest"]["expected_cells"]
        and dict(cells) == expected_cells
    )
    checks["manifest"] = {
        "unique_cases": n_cases, "cell_count": len(cells),
        "exact_breakdown_match": manifest_ok,
    }
    if not manifest_ok:
        failures.append("manifest count or cell breakdown mismatch")

    expected_agent_cases = (
        3 * protocol["case_manifest"]["brainweb_rows_per_cell"]
        + 12 * protocol["case_manifest"]["hecktor_rows_per_cell"]
    )
    budget = protocol["formal_budgets"]
    budget_ok = (
        expected_agent_cases == budget["vlm_calls_max"] == 2628
        and budget["brainweb_agent_cases"] == 2088
        and budget["hecktor_agent_cases"] == 540
        and budget["routed_endpoint_cases"] == n_cases
        and budget["paired_det_comparator_cases"] == expected_agent_cases
        and budget["total_endpoint_rows"] == n_cases + expected_agent_cases
        and budget["routed_medsam_calls_max_including_det"] == 33948
        and budget["comparator_medsam_calls_max"] == 10512
        and budget["total_medsam_calls_max"] == 44460
        and budget["retries_per_case_max"] == 0
    )
    checks["budgets"] = {"expected_agent_cases": expected_agent_cases,
                          "match": budget_ok, **budget}
    if not budget_ok:
        failures.append("formal call or endpoint budget mismatch")

    required_flags = {
        "locked_test_access": protocol.get("locked_test_access") == "prohibited",
        "post_fusion_forbidden": protocol.get("post_fusion_forbidden") is True,
        "manual_annotations_forbidden":
            protocol.get("manual_annotations_forbidden") is True,
        "anatomical_inputs_forbidden":
            protocol.get("anatomical_inputs_forbidden") is True,
        "transactional":
            protocol["transaction"].get("atomic_per_case_publish") is True,
        "resumable":
            protocol["transaction"].get("explicit_resumable_ledger") is True,
        "intention_to_treat":
            protocol["transaction"].get("intention_to_treat") is True,
        "no_retries": budget.get("retries_per_case_max") == 0,
        "secret_excluded": protocol["api_receipt"].get(
            "authorization_secret_forbidden_in_artifacts") is True,
        "completion_budget_avoids_reasoning_truncation":
            protocol["api_receipt"].get("max_completion_tokens") == 8192,
    }
    checks["safety_contract"] = required_flags
    if not all(required_flags.values()):
        failures.append("one or more safety-contract flags are invalid")

    rc, head = _git(repo, "rev-parse", "HEAD")
    rc_status, porcelain = _git(repo, "status", "--porcelain")
    head_ok = rc == 0 and bool(expected_head) and head == expected_head
    clean = rc_status == 0 and porcelain == ""
    checks["source_identity"] = {
        "head": head, "expected_head": expected_head,
        "head_match": head_ok, "clean_tree": clean,
    }
    if not head_ok:
        failures.append("HEAD does not match required execution commit")
    if not clean:
        failures.append("source tree is not clean")

    return {
        "artifact_kind": "stage7_formal_agent_zero_write_release_check",
        "status": "PASS_ready_for_formal_execution" if not failures
                  else "BLOCKED_release_invariants_failed",
        "read_only": True,
        "formal_output_created": False,
        "vlm_calls": 0,
        "medsam_calls": 0,
        "kem_endpoints": 0,
        "protocol_path": str(protocol_path.relative_to(repo)),
        "protocol_canonical_lf_sha256": canonical_lf_sha256(protocol_path),
        "checks": checks,
        "failures": failures,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--expected-head", required=True)
    args = parser.parse_args()

    repo = args.root.resolve()
    output = args.output.resolve()
    audit = build_audit(repo, args.protocol.resolve(), output, args.expected_head)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n",
                      encoding="utf-8")
    print(json.dumps(audit, indent=2, sort_keys=True))
    if not audit["status"].startswith("PASS_"):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
