"""Resumable no-new-API prompt ablation over all routed Agent cases."""
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import time
import uuid
from pathlib import Path

import numpy as np
from run_stage7_formal_smoke import _reconstruct, _sha, _write_json
from run_stage7_formal_validation import (
    _case_dir,
    _head,
    _predictor,
    _rows,
    _source_path,
)
from sam_kem.medsam import predict_prompt_object_logits, sha256_file
from sam_kem.operator_factory import build_discrete_radon
from sam_kem.reconstruction import operator_support_mask
from sam_kem.stage7_formal_runtime import aggregate_logits
from sam_kem.stage7a_prompt_plan import PromptObject

EXPECTED_AGENT_CASES = 2628
EXPECTED_NEW_ENDPOINTS = 2088 + 2 * 540


def _objects(records: list[dict], *, positive: bool, negative: bool) -> tuple[PromptObject, ...]:
    return tuple(PromptObject(
        object_id=int(record["object_id"]),
        box_xyxy=tuple(record["box_xyxy"]),
        positive_points_xy=tuple(tuple(point) for point in record["positive_points_xy"])
        if positive else (),
        negative_points_xy=tuple(tuple(point) for point in record["negative_points_xy"])
        if negative else (),
        confidence=float(record["confidence"]),
    ) for record in records)


def _destination(output: Path, row: dict[str, str]) -> Path:
    return _case_dir(output, row)


def _completed(path: Path) -> bool:
    result = path / "result.json"
    success = path / "_SUCCESS.json"
    if not result.is_file() or not success.is_file():
        return False
    try:
        return json.loads(success.read_text())["result_sha256"] == _sha(result)
    except (OSError, ValueError, KeyError):
        return False


def initialize(args: argparse.Namespace) -> None:
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite prompt-ablation namespace: {args.output}")
    source_audit = json.loads(
        (args.source_output / "integrity_audit.json").read_text(encoding="utf-8")
    )
    if source_audit["status"] != "PASS_complete_pending_preregistered_analysis":
        raise ValueError("formal source execution has not passed integrity finalization")
    args.output.mkdir(parents=True)
    (args.output / "cases").mkdir()
    record = {
        "artifact_kind": "stage7_formal_prompt_ablation_execution",
        "status": "initialized_pending_workers", "execution_git_commit": _head(args.repo_root),
        "source_integrity_audit_sha256": sha256_file(
            args.source_output / "integrity_audit.json"),
        "ablation_protocol_sha256": sha256_file(args.ablation_protocol),
        "agent_case_count": EXPECTED_AGENT_CASES,
        "expected_new_endpoint_count": EXPECTED_NEW_ENDPOINTS,
        "new_api_calls": 0, "worker_count": args.worker_count,
    }
    _write_json(args.output / "execution_record.json", record)
    print(json.dumps(record, indent=2, sort_keys=True))


def worker(args: argparse.Namespace) -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.worker_index)
    rows = [row for row in _rows(args.manifest) if (
        row["dataset"] == "hecktor" or (
            row["dataset"] == "brainweb" and int(row["count_level"]) == 100000
        )
    )]
    assigned = [row for index, row in enumerate(rows)
                if index % args.worker_count == args.worker_index]
    operator = build_discrete_radon(repo_root=args.asset_project, size=128, angles=180)
    support = operator_support_mask(operator)
    predictors = {}
    written = reused = 0
    started = time.perf_counter()
    for row in assigned:
        destination = _destination(args.output, row)
        if _completed(destination):
            reused += 1
            continue
        if destination.exists():
            raise RuntimeError(f"incomplete published prompt ablation: {destination}")
        source_dir = _case_dir(args.source_output, row)
        source_result_path = source_dir / "result.json"
        source = json.loads(source_result_path.read_text(encoding="utf-8"))
        if source["fallback_reason"]:
            # ITT fallback cases contain no executed Agent prompt to ablate.
            arms = []
        elif row["dataset"] == "brainweb":
            arms = [("multi_box_only", _objects(
                source["policy_objects"], positive=False, negative=False))]
        else:
            arms = [
                ("lesion_box_only", _objects(
                    source["policy_objects"], positive=False, negative=False)),
                ("lesion_box_positive", _objects(
                    source["policy_objects"], positive=True, negative=False)),
            ]
        temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.tmp")
        temporary.mkdir(parents=True)
        try:
            dataset = row["dataset"]
            if dataset not in predictors:
                predictors[dataset], _ = _predictor(args.asset_project, dataset)
            mlem_path = _source_path(
                row, "mlem10_input_path", args.brainweb_array_root,
                args.hecktor_array_root,
            )
            target_path = _source_path(
                row, "target_path", args.brainweb_array_root,
                args.hecktor_array_root,
            )
            sinogram_path = _source_path(
                row, "sinogram_path", args.brainweb_array_root,
                args.hecktor_array_root,
            )
            mlem10 = np.load(mlem_path).astype(np.float32)
            target = np.load(target_path).astype(np.float32)
            measured = np.load(sinogram_path).astype(np.float64)
            endpoints = []
            for arm, objects in arms:
                logits = predict_prompt_object_logits(
                    predictors[dataset], mlem10, objects, output_shape=(128, 128),
                )
                prior = aggregate_logits(logits)
                reconstruction, metrics = _reconstruct(
                    prior=prior, measured=measured, target=target,
                    projection_scale=float(row["projection_scale"]),
                    operator=operator, support=support, dataset=dataset,
                )
                np.save(temporary / f"{arm}_prior.npy", prior, allow_pickle=False)
                np.save(
                    temporary / f"{arm}_reconstruction.npy", reconstruction,
                    allow_pickle=False,
                )
                endpoints.append({
                    "arm": arm, "object_count": len(objects), **metrics,
                    "prior_sha256": _sha(temporary / f"{arm}_prior.npy"),
                    "reconstruction_sha256": _sha(
                        temporary / f"{arm}_reconstruction.npy"),
                })
            result = {
                "case_key": source["case_key"], "dataset": dataset,
                "subject_id": row["subject_id"], "slice_id": row["slice_id"],
                "count_level": int(row["count_level"]),
                "realization_id": int(row["realization_id"]),
                "source_result_sha256": sha256_file(source_result_path),
                "source_fallback_used": bool(source["fallback_reason"]),
                "new_api_calls": 0, "endpoints": endpoints,
            }
            _write_json(temporary / "result.json", result)
            _write_json(temporary / "_SUCCESS.json", {
                "case_key": result["case_key"],
                "result_sha256": _sha(temporary / "result.json"),
            })
            destination.parent.mkdir(parents=True, exist_ok=True)
            os.replace(temporary, destination)
            written += 1
        except BaseException:
            shutil.rmtree(temporary, ignore_errors=True)
            raise
    record = {
        "worker_index": args.worker_index, "assigned_cases": len(assigned),
        "written_cases": written, "reused_verified_cases": reused,
        "elapsed_seconds": time.perf_counter() - started, "status": "complete",
    }
    _write_json(args.output / f"worker_{args.worker_index:02d}_SUCCESS.json", record)
    print(json.dumps(record, indent=2, sort_keys=True))


def finalize(args: argparse.Namespace) -> None:
    rows = [row for row in _rows(args.manifest) if (
        row["dataset"] == "hecktor" or (
            row["dataset"] == "brainweb" and int(row["count_level"]) == 100000
        )
    )]
    failures = []
    output_rows = []
    fallback_cases = 0
    for row in rows:
        path = _destination(args.output, row)
        if not _completed(path):
            failures.append(f"incomplete case: {path}")
            continue
        result = json.loads((path / "result.json").read_text())
        fallback_cases += result["source_fallback_used"]
        for endpoint in result["endpoints"]:
            output_rows.append({
                "dataset": result["dataset"], "subject_id": result["subject_id"],
                "slice_id": result["slice_id"], "count_level": result["count_level"],
                "realization_id": result["realization_id"], **endpoint,
            })
    expected = EXPECTED_NEW_ENDPOINTS - fallback_cases - sum(
        1 for row in rows if row["dataset"] == "hecktor"
        and json.loads((_destination(args.output, row) / "result.json").read_text())[
            "source_fallback_used"]
    ) if not failures else -1
    if not failures and len(output_rows) != expected:
        failures.append(f"new endpoint count {len(output_rows)} != ITT expected {expected}")
    csv_path = args.output / "prompt_ablation_endpoints.csv"
    if output_rows:
        with csv_path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(output_rows[0]), lineterminator="\n")
            writer.writeheader()
            writer.writerows(output_rows)
    audit = {
        "artifact_kind": "stage7_formal_prompt_ablation_integrity",
        "status": "PASS_complete" if not failures else "FAIL_integrity",
        "agent_case_count": len(rows), "source_fallback_cases": fallback_cases,
        "new_endpoint_count": len(output_rows), "new_api_calls": 0,
        "endpoint_csv_sha256": _sha(csv_path) if csv_path.is_file() else "",
        "failures": failures,
    }
    _write_json(args.output / "integrity_audit.json", audit)
    print(json.dumps(audit, indent=2, sort_keys=True))
    if failures:
        raise SystemExit(2)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("initialize", "worker", "finalize"))
    parser.add_argument("--repo-root", required=True, type=Path)
    parser.add_argument("--asset-project", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--ablation-protocol", required=True, type=Path)
    parser.add_argument("--source-output", required=True, type=Path)
    parser.add_argument("--brainweb-array-root", required=True, type=Path)
    parser.add_argument("--hecktor-array-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--worker-count", type=int, default=8)
    parser.add_argument("--worker-index", type=int)
    args = parser.parse_args()
    if args.mode == "initialize":
        initialize(args)
    elif args.mode == "worker":
        if args.worker_index is None:
            parser.error("worker mode requires --worker-index")
        worker(args)
    else:
        finalize(args)


if __name__ == "__main__":
    main()
