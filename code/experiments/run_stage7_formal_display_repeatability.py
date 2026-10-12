"""Run the frozen 20-case display and API-repeatability ablations."""
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
from run_stage7_formal_smoke import (
    MODEL,
    _api_call,
    _reconstruct,
    _response_content,
    _sha,
    _write_json,
)
from run_stage7_formal_validation import _case_dir, _head, _predictor, _source_path
from sam_kem.medsam import predict_prompt_object_logits, sha256_file
from sam_kem.operator_factory import build_discrete_radon
from sam_kem.reconstruction import operator_support_mask
from sam_kem.stage7_formal_runtime import (
    FormalPromptError,
    aggregate_logits,
    parse_formal_prompt,
    prompt_objects_json,
    render_pet_display,
)
from sam_kem.stage7a_prompt_plan import det_component_objects

DISPLAY_ARMS = ("grayscale_only", "jet_only", "grayscale_plus_jet")
EXPECTED_CASES = 20
EXPECTED_CALLS_PER_CASE = 5
EXPECTED_CALLS = EXPECTED_CASES * EXPECTED_CALLS_PER_CASE


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    keys = [(r["dataset"], r["subject_id"], r["slice_id"], r["count_level"],
             r["realization_id"]) for r in rows]
    if len(rows) != EXPECTED_CASES or len(set(keys)) != EXPECTED_CASES:
        raise ValueError("fixed secondary subset must contain 20 unique cases")
    return rows


def _completed(path: Path) -> bool:
    try:
        marker = json.loads((path / "_SUCCESS.json").read_text(encoding="utf-8"))
        return marker["result_sha256"] == _sha(path / "result.json")
    except (OSError, ValueError, KeyError):
        return False


def call_plan() -> tuple[tuple[str, str, int], ...]:
    """Return (family, display mode, replicate) for the five new calls/case."""
    display = tuple(("display", arm, 0) for arm in DISPLAY_ARMS)
    repeats = tuple(("repeatability", "grayscale_plus_jet", index) for index in (1, 2))
    return display + repeats


def initialize(args: argparse.Namespace) -> None:
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite secondary namespace: {args.output}")
    source_audit = json.loads((args.source_output / "integrity_audit.json").read_text())
    if source_audit["status"] != "PASS_complete_pending_preregistered_analysis":
        raise ValueError("formal source execution has not passed integrity finalization")
    args.output.mkdir(parents=True)
    (args.output / "cases").mkdir()
    record = {
        "artifact_kind": "stage7_formal_display_repeatability_execution",
        "status": "initialized_pending_workers",
        "execution_git_commit": _head(args.repo_root),
        "source_integrity_audit_sha256": sha256_file(
            args.source_output / "integrity_audit.json"),
        "ablation_protocol_sha256": sha256_file(args.ablation_protocol),
        "selection_sha256": sha256_file(args.selection),
        "case_count": EXPECTED_CASES,
        "new_api_calls_per_case": EXPECTED_CALLS_PER_CASE,
        "expected_new_api_calls": EXPECTED_CALLS,
        "worker_count": args.worker_count,
        "api_model": MODEL,
        "api_retries_per_call": 0,
    }
    _write_json(args.output / "execution_record.json", record)
    print(json.dumps(record, indent=2, sort_keys=True))


def _prompt_path(args: argparse.Namespace, dataset: str, family: str) -> Path:
    suffix = "display_ablation_v1" if family == "display" else "v1"
    return args.repo_root / f"code/configs/prompts/stage7_formal_{dataset}_{suffix}.txt"


def worker(args: argparse.Namespace) -> None:
    if not 0 <= args.worker_index < args.worker_count:
        raise ValueError("worker index is outside worker count")
    record = json.loads((args.output / "execution_record.json").read_text())
    if record["execution_git_commit"] != _head(args.repo_root):
        raise ValueError("execution HEAD changed after initialization")
    token = args.api_token_file.read_text(encoding="utf-8").strip()
    if not token:
        raise ValueError("API token file is empty")
    gpu_index = args.worker_index if args.gpu_index is None else args.gpu_index
    if gpu_index < 0:
        raise ValueError("GPU index must be non-negative")
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_index)
    rows = [row for index, row in enumerate(_rows(args.selection))
            if index % args.worker_count == args.worker_index]
    operator = build_discrete_radon(repo_root=args.asset_project, size=128, angles=180)
    support = operator_support_mask(operator)
    predictors: dict[str, object] = {}
    written = reused = 0
    started = time.perf_counter()
    for row in rows:
        destination = _case_dir(args.output, row)
        if _completed(destination):
            reused += 1
            continue
        if destination.exists():
            raise RuntimeError(f"incomplete published case requires audit: {destination}")
        temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.tmp")
        temporary.mkdir(parents=True)
        try:
            dataset = row["dataset"]
            if dataset not in predictors:
                predictors[dataset], _ = _predictor(args.asset_project, dataset)
            mlem_path = _source_path(row, "mlem10_input_path", args.brainweb_array_root,
                                     args.hecktor_array_root)
            target_path = _source_path(row, "target_path", args.brainweb_array_root,
                                       args.hecktor_array_root)
            sinogram_path = _source_path(row, "sinogram_path", args.brainweb_array_root,
                                         args.hecktor_array_root)
            mlem10 = np.load(mlem_path).astype(np.float32)
            target = np.load(target_path).astype(np.float32)
            measured = np.load(sinogram_path).astype(np.float64)
            calls = []
            for call_index, (family, view_mode, replicate) in enumerate(call_plan(), start=1):
                call_dir = temporary / f"call_{call_index:02d}"
                call_dir.mkdir()
                png, image_sha = render_pet_display(mlem10, view_mode=view_mode)
                (call_dir / "vlm_input.png").write_bytes(png)
                prompt_path = _prompt_path(args, dataset, family)
                request, response, api_seconds = _api_call(
                    token=token, prompt=prompt_path.read_text(encoding="utf-8"), png=png,
                )
                _write_json(call_dir / "raw_request.json", request)
                _write_json(call_dir / "raw_response.json", response)
                fallback_reason = ""
                try:
                    objects = parse_formal_prompt(_response_content(response), dataset=dataset)
                    if not isinstance(response.get("model"), str) or not isinstance(
                        response.get("usage"), dict
                    ):
                        raise FormalPromptError("API receipt lacks returned model or usage")
                except FormalPromptError as error:
                    fallback_reason = str(error)
                    objects = tuple(det_component_objects(mlem10))
                decode_started = time.perf_counter()
                logits = predict_prompt_object_logits(
                    predictors[dataset], mlem10, objects, output_shape=(128, 128))
                prior = aggregate_logits(logits)
                decode_seconds = time.perf_counter() - decode_started
                reconstruction, metrics = _reconstruct(
                    prior=prior, measured=measured, target=target,
                    projection_scale=float(row["projection_scale"]), operator=operator,
                    support=support, dataset=dataset)
                np.save(call_dir / "prior.npy", prior, allow_pickle=False)
                np.save(call_dir / "reconstruction.npy", reconstruction, allow_pickle=False)
                calls.append({
                    "call_index": call_index, "family": family,
                    "view_mode": view_mode, "replicate": replicate,
                    "prompt_sha256": sha256_file(prompt_path), "image_sha256": image_sha,
                    "requested_model": MODEL, "returned_model": response.get("model"),
                    "usage": response.get("usage"), "api_seconds": api_seconds,
                    "decode_seconds": decode_seconds, "fallback_reason": fallback_reason,
                    "objects": prompt_objects_json(objects), "object_count": len(objects),
                    **metrics, "prior_sha256": _sha(call_dir / "prior.npy"),
                    "reconstruction_sha256": _sha(call_dir / "reconstruction.npy"),
                    "raw_request_sha256": _sha(call_dir / "raw_request.json"),
                    "raw_response_sha256": _sha(call_dir / "raw_response.json"),
                })
            source_path = _case_dir(args.source_output, row) / "result.json"
            source = json.loads(source_path.read_text(encoding="utf-8"))
            result = {
                "case_key": source["case_key"], "dataset": dataset,
                "subject_id": row["subject_id"], "slice_id": row["slice_id"],
                "count_level": int(row["count_level"]),
                "realization_id": int(row["realization_id"]),
                "source_result_sha256": sha256_file(source_path),
                "source_main_objects": source["policy_objects"],
                "source_main_fallback_reason": source["fallback_reason"],
                "new_api_calls": len(calls), "api_retries": 0, "calls": calls,
            }
            _write_json(temporary / "result.json", result)
            _write_json(temporary / "_SUCCESS.json", {
                "case_key": result["case_key"],
                "result_sha256": _sha(temporary / "result.json")})
            destination.parent.mkdir(parents=True, exist_ok=True)
            os.replace(temporary, destination)
            written += 1
        except BaseException:
            shutil.rmtree(temporary, ignore_errors=True)
            raise
    worker_record = {
        "status": "complete", "worker_index": args.worker_index,
        "physical_gpu_index": gpu_index,
        "assigned_cases": len(rows), "written_cases": written,
        "reused_verified_cases": reused,
        "elapsed_seconds": time.perf_counter() - started,
    }
    _write_json(args.output / f"worker_{args.worker_index:02d}_SUCCESS.json", worker_record)
    print(json.dumps(worker_record, indent=2, sort_keys=True))


def finalize(args: argparse.Namespace) -> None:
    failures: list[str] = []
    rows_out = []
    fallbacks = calls = 0
    for row in _rows(args.selection):
        path = _case_dir(args.output, row)
        if not _completed(path):
            failures.append(f"incomplete case: {path}")
            continue
        result = json.loads((path / "result.json").read_text(encoding="utf-8"))
        calls += result["new_api_calls"]
        for call in result["calls"]:
            fallbacks += bool(call["fallback_reason"])
            rows_out.append({
                "dataset": result["dataset"], "subject_id": result["subject_id"],
                "slice_id": result["slice_id"], "count_level": result["count_level"],
                "realization_id": result["realization_id"],
                **{key: call[key] for key in (
                    "call_index", "family", "view_mode", "replicate", "object_count",
                    "fallback_reason", "nrmse", "ssim", "api_seconds", "decode_seconds",
                    "prior_sha256", "reconstruction_sha256")},
            })
    if calls != EXPECTED_CALLS:
        failures.append(f"new API call count {calls} != {EXPECTED_CALLS}")
    if len(rows_out) != EXPECTED_CALLS:
        failures.append(f"endpoint count {len(rows_out)} != {EXPECTED_CALLS}")
    if not all(np.isfinite(row["nrmse"]) and np.isfinite(row["ssim"]) for row in rows_out):
        failures.append("non-finite endpoint metric")
    csv_path = args.output / "display_repeatability_endpoints.csv"
    if rows_out:
        with csv_path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows_out[0]), lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows_out)
    audit = {
        "artifact_kind": "stage7_formal_display_repeatability_integrity",
        "status": "PASS_complete" if not failures else "FAIL_integrity",
        "case_count": EXPECTED_CASES, "new_api_calls": calls,
        "api_retries": 0, "endpoint_count": len(rows_out),
        "fallback_responses": fallbacks,
        "all_metrics_finite": not any("non-finite" in item for item in failures),
        "endpoint_csv_sha256": _sha(csv_path) if csv_path.is_file() else "",
        "worker_success_count": sum(
            (args.output / f"worker_{index:02d}_SUCCESS.json").is_file()
            for index in range(args.worker_count)),
        "locked_test_access": "prohibited", "post_fusion_used": False,
        "manual_or_ground_truth_prompts_used": False,
        "anatomical_inputs_used": False, "authorization_secret_recorded": False,
        "failures": failures,
    }
    _write_json(args.output / "integrity_audit.json", audit)
    if not failures:
        _write_json(args.output / "_SUCCESS.json", {
            "status": audit["status"],
            "integrity_audit_sha256": _sha(args.output / "integrity_audit.json")})
    print(json.dumps(audit, indent=2, sort_keys=True))
    if failures:
        raise SystemExit(2)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("initialize", "worker", "finalize"))
    parser.add_argument("--repo-root", required=True, type=Path)
    parser.add_argument("--asset-project", required=True, type=Path)
    parser.add_argument("--selection", required=True, type=Path)
    parser.add_argument("--ablation-protocol", required=True, type=Path)
    parser.add_argument("--source-output", required=True, type=Path)
    parser.add_argument("--brainweb-array-root", required=True, type=Path)
    parser.add_argument("--hecktor-array-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--api-token-file", type=Path)
    parser.add_argument("--worker-count", type=int, default=8)
    parser.add_argument("--worker-index", type=int)
    parser.add_argument("--gpu-index", type=int)
    args = parser.parse_args()
    if args.mode == "initialize":
        initialize(args)
    elif args.mode == "worker":
        if args.worker_index is None or args.api_token_file is None:
            parser.error("worker requires --worker-index and --api-token-file")
        worker(args)
    else:
        finalize(args)


if __name__ == "__main__":
    main()
