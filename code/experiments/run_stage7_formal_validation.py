"""Resumable multi-GPU execution for the preregistered Stage 7 validation."""
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import time
import uuid
from pathlib import Path

import numpy as np
from run_stage7_formal_smoke import (
    HYPERPARAMETERS,
    MODEL,
    _api_call,
    _reconstruct,
    _response_content,
    _sha,
    _write_json,
)
from sam_kem.medsam import (
    CheckpointSpec,
    predict_prompt_object_logits,
    predictor_factory,
    sha256_file,
)
from sam_kem.operator_factory import build_discrete_radon
from sam_kem.reconstruction import operator_support_mask
from sam_kem.stage7_formal_runtime import (
    FormalPromptError,
    aggregate_logits,
    parse_formal_prompt,
    prompt_objects_json,
    render_pet_views,
)
from sam_kem.stage7a_prompt_plan import det_component_objects

EXPECTED_CASES = 8892
EXPECTED_AGENT_CASES = 2628
EXPECTED_ENDPOINTS = EXPECTED_CASES + EXPECTED_AGENT_CASES


def _head(repo: Path) -> str:
    return subprocess.check_output(
        ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True,
    ).strip()


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    keys = [
        (row["dataset"], row["subject_id"], row["slice_id"],
         int(row["count_level"]), int(row["realization_id"]))
        for row in rows if row["split"] == "validation"
    ]
    selected = [row for row in rows if row["split"] == "validation"]
    if len(selected) != EXPECTED_CASES or len(keys) != len(set(keys)):
        raise ValueError("validation manifest does not contain 8,892 unique cases")
    return sorted(selected, key=lambda row: (
        row["dataset"], row["subject_id"], row["slice_id"],
        int(row["count_level"]), int(row["realization_id"]),
    ))


def _is_agent(row: dict[str, str]) -> bool:
    return row["dataset"] == "hecktor" or (
        row["dataset"] == "brainweb" and int(row["count_level"]) == 100000
    )


def _case_key(row: dict[str, str]) -> str:
    return "__".join((row["dataset"], row["subject_id"], row["slice_id"],
                       f"c{row['count_level']}", f"r{row['realization_id']}"))


def _case_dir(output: Path, row: dict[str, str]) -> Path:
    return (output / "cases" / row["dataset"] / row["subject_id"]
            / row["slice_id"] / f"c{row['count_level']}"
            / f"r{row['realization_id']}")


def _source_path(row: dict[str, str], field: str,
                 brainweb_root: Path, hecktor_root: Path) -> Path:
    relative = Path(row[field])
    dataset = row["dataset"]
    parts = relative.parts
    try:
        position = parts.index(dataset)
    except ValueError as error:
        raise ValueError(f"{field} has no dataset component: {relative}") from error
    root = brainweb_root if dataset == "brainweb" else hecktor_root
    return root.joinpath(*parts[position + 1:])


def _predictor(asset_project: Path, dataset: str):
    base = asset_project / "checkpoints/MedSAM2_latest.pt"
    checkpoint = asset_project / f"checkpoints/dev_trainmax_v1/medsam2_decoder_{dataset}.pt"
    digest = sha256_file(checkpoint)
    value = predictor_factory(
        medsam_source=asset_project / "external/MedSAM2",
        pretrained_checkpoint=base,
    )(CheckpointSpec(
        dataset=dataset, path=checkpoint, sha256=digest,
        normalization_id="train_max_fixed_C", run_id="dev_trainmax_v1",
    ))
    return value, digest


def initialize(args: argparse.Namespace) -> None:
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite formal namespace: {args.output}")
    rows = _rows(args.manifest)
    agent_cases = sum(_is_agent(row) for row in rows)
    if agent_cases != EXPECTED_AGENT_CASES:
        raise ValueError(f"agent routing count mismatch: {agent_cases}")
    args.output.mkdir(parents=True)
    (args.output / "cases").mkdir()
    record = {
        "artifact_kind": "stage7_formal_execution_record",
        "status": "initialized_pending_workers",
        "execution_git_commit": _head(args.repo_root),
        "manifest_sha256": sha256_file(args.manifest),
        "protocol_sha256": sha256_file(args.protocol),
        "smoke_audit_sha256": sha256_file(args.smoke_audit),
        "case_count": len(rows), "agent_case_count": agent_cases,
        "expected_endpoint_rows": EXPECTED_ENDPOINTS,
        "worker_count": args.worker_count,
        "gpu_mapping": {str(index): index for index in range(args.worker_count)},
        "api_model": MODEL, "api_retries_per_case": 0,
        "brainweb_array_root": str(args.brainweb_array_root),
        "hecktor_array_root": str(args.hecktor_array_root),
        "historical_namespaces_read_only": True,
    }
    _write_json(args.output / "execution_record.json", record)
    print(json.dumps(record, indent=2, sort_keys=True))


def _completed(path: Path) -> bool:
    success = path / "_SUCCESS.json"
    result = path / "result.json"
    if not success.is_file() or not result.is_file():
        return False
    try:
        marker = json.loads(success.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    return marker.get("result_sha256") == _sha(result)


def worker(args: argparse.Namespace) -> None:
    if not (0 <= args.worker_index < args.worker_count):
        raise ValueError("worker index is outside the worker count")
    record = json.loads((args.output / "execution_record.json").read_text())
    if record["execution_git_commit"] != _head(args.repo_root):
        raise ValueError("execution HEAD changed after initialization")
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.worker_index)
    token = args.api_token_file.read_text(encoding="utf-8").strip()
    if not token:
        raise ValueError("API token file is empty")
    rows = _rows(args.manifest)
    assigned = [row for index, row in enumerate(rows) if index % args.worker_count == args.worker_index]
    operator = build_discrete_radon(repo_root=args.asset_project, size=128, angles=180)
    support = operator_support_mask(operator)
    predictors: dict[str, object] = {}
    checkpoint_shas: dict[str, str] = {}
    written = reused = 0
    started = time.perf_counter()
    for row in assigned:
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
                predictors[dataset], checkpoint_shas[dataset] = _predictor(
                    args.asset_project, dataset,
                )
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
            agent = _is_agent(row)
            api_calls = 0
            fallback_reason = ""
            returned_model = None
            usage = None
            latency = 0.0
            image_sha = ""
            if agent:
                png, image_sha = render_pet_views(mlem10)
                (temporary / "vlm_input.png").write_bytes(png)
                prompt_path = args.repo_root / f"code/configs/prompts/stage7_formal_{dataset}_v1.txt"
                request_payload, response_payload, latency = _api_call(
                    token=token, prompt=prompt_path.read_text(encoding="utf-8"), png=png,
                )
                api_calls = 1
                _write_json(temporary / "raw_request.json", request_payload)
                _write_json(temporary / "raw_response.json", response_payload)
                returned_model = response_payload.get("model")
                usage = response_payload.get("usage")
                try:
                    objects = parse_formal_prompt(
                        _response_content(response_payload), dataset=dataset,
                    )
                    if not isinstance(returned_model, str) or not isinstance(usage, dict):
                        raise FormalPromptError("API receipt lacks returned model or usage")
                    executed = (
                        "BW_AGENT_LOCAL_POS" if dataset == "brainweb"
                        else "HE_AGENT_LESION_BPN"
                    )
                except FormalPromptError as error:
                    fallback_reason = str(error)
                    objects = tuple(det_component_objects(mlem10))
                    executed = "DET_COMPONENT_FALLBACK"
            else:
                objects = tuple(det_component_objects(mlem10))
                executed = "DET_COMPONENT"

            object_logits = predict_prompt_object_logits(
                predictors[dataset], mlem10, objects, output_shape=(128, 128),
            )
            prior = aggregate_logits(object_logits)
            reconstruction, metrics = _reconstruct(
                prior=prior, measured=measured, target=target,
                projection_scale=float(row["projection_scale"]), operator=operator,
                support=support, dataset=dataset,
            )
            np.save(temporary / "policy_prior.npy", prior, allow_pickle=False)
            np.save(temporary / "policy_reconstruction.npy", reconstruction, allow_pickle=False)
            endpoints = [{
                "role": "policy", "requested_method": (
                    "BW_AGENT_LOCAL_POS" if dataset == "brainweb" and agent
                    else "HE_AGENT_LESION_BPN" if dataset == "hecktor" and agent
                    else "DET_COMPONENT"
                ),
                "executed_method": executed, "fallback_used": bool(fallback_reason),
                "object_count": len(objects), **metrics,
                "prior_sha256": _sha(temporary / "policy_prior.npy"),
                "reconstruction_sha256": _sha(temporary / "policy_reconstruction.npy"),
            }]
            if agent:
                comparator_objects = tuple(det_component_objects(mlem10))
                if fallback_reason:
                    comparator_prior = prior
                    comparator_reconstruction = reconstruction
                    comparator_metrics = metrics
                    comparator_reused = True
                else:
                    comparator_logits = predict_prompt_object_logits(
                        predictors[dataset], mlem10, comparator_objects,
                        output_shape=(128, 128),
                    )
                    comparator_prior = aggregate_logits(comparator_logits)
                    comparator_reconstruction, comparator_metrics = _reconstruct(
                        prior=comparator_prior, measured=measured, target=target,
                        projection_scale=float(row["projection_scale"]), operator=operator,
                        support=support, dataset=dataset,
                    )
                    comparator_reused = False
                np.save(temporary / "det_prior.npy", comparator_prior, allow_pickle=False)
                np.save(
                    temporary / "det_reconstruction.npy", comparator_reconstruction,
                    allow_pickle=False,
                )
                endpoints.append({
                    "role": "paired_comparator", "requested_method": "DET_COMPONENT",
                    "executed_method": "DET_COMPONENT", "fallback_used": False,
                    "object_count": len(comparator_objects), **comparator_metrics,
                    "prior_sha256": _sha(temporary / "det_prior.npy"),
                    "reconstruction_sha256": _sha(temporary / "det_reconstruction.npy"),
                    "computation_reused_from_fallback": comparator_reused,
                })
            result = {
                "case_key": _case_key(row), "dataset": dataset,
                "subject_id": row["subject_id"], "slice_id": row["slice_id"],
                "count_level": int(row["count_level"]),
                "realization_id": int(row["realization_id"]),
                "projection_scale": float(row["projection_scale"]),
                "input_sha256": {
                    "mlem10": sha256_file(mlem_path), "target": sha256_file(target_path),
                    "sinogram": sha256_file(sinogram_path),
                },
                "agent_routed": agent, "api_calls": api_calls, "api_retries": 0,
                "requested_model": MODEL if agent else None,
                "returned_model": returned_model, "usage": usage,
                "api_latency_seconds": latency, "rendered_image_sha256": image_sha,
                "fallback_reason": fallback_reason,
                "policy_objects": prompt_objects_json(objects),
                "checkpoint_sha256": checkpoint_shas[dataset],
                "hyperparameters": HYPERPARAMETERS[dataset],
                "endpoints": endpoints,
                "locked_test_access": "prohibited", "post_fusion_used": False,
                "manual_or_ground_truth_prompts_used": False,
                "anatomical_inputs_used": False,
                "authorization_secret_recorded": False,
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
    worker_record = {
        "worker_index": args.worker_index, "worker_count": args.worker_count,
        "assigned_cases": len(assigned), "written_cases": written,
        "reused_verified_cases": reused,
        "elapsed_seconds": time.perf_counter() - started,
        "status": "complete",
    }
    _write_json(args.output / f"worker_{args.worker_index:02d}_SUCCESS.json", worker_record)
    print(json.dumps(worker_record, indent=2, sort_keys=True))


def finalize(args: argparse.Namespace) -> None:
    rows = _rows(args.manifest)
    failures = []
    endpoint_rows = []
    api_calls = fallbacks = valid_agents = 0
    for row in rows:
        path = _case_dir(args.output, row)
        if not _completed(path):
            failures.append(f"incomplete case: {_case_key(row)}")
            continue
        result = json.loads((path / "result.json").read_text(encoding="utf-8"))
        api_calls += result["api_calls"]
        if result["agent_routed"]:
            fallback = bool(result["fallback_reason"])
            fallbacks += fallback
            valid_agents += not fallback
        for endpoint in result["endpoints"]:
            endpoint_rows.append({
                "dataset": result["dataset"], "subject_id": result["subject_id"],
                "slice_id": result["slice_id"], "count_level": result["count_level"],
                "realization_id": result["realization_id"], **endpoint,
            })
    if len(endpoint_rows) != EXPECTED_ENDPOINTS:
        failures.append(f"endpoint count {len(endpoint_rows)} != {EXPECTED_ENDPOINTS}")
    if api_calls != EXPECTED_AGENT_CASES:
        failures.append(f"API call count {api_calls} != {EXPECTED_AGENT_CASES}")
    if not all(np.isfinite(item["nrmse"]) and np.isfinite(item["ssim"])
               for item in endpoint_rows):
        failures.append("non-finite endpoint metric")
    fieldnames = [
        "dataset", "subject_id", "slice_id", "count_level", "realization_id",
        "role", "requested_method", "executed_method", "fallback_used",
        "object_count", "nrmse", "ssim", "prior_sha256",
        "reconstruction_sha256", "computation_reused_from_fallback",
    ]
    csv_path = args.output / "per_case_endpoints.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, lineterminator="\n",
                                extrasaction="ignore")
        writer.writeheader()
        writer.writerows(endpoint_rows)
    audit = {
        "artifact_kind": "stage7_formal_validation_integrity_audit",
        "status": "PASS_complete_pending_preregistered_analysis" if not failures
                  else "FAIL_integrity",
        "case_count": len(rows), "endpoint_count": len(endpoint_rows),
        "api_calls": api_calls, "api_retries": 0,
        "valid_agent_responses": valid_agents, "fallback_responses": fallbacks,
        "all_metrics_finite": not any("non-finite" in item for item in failures),
        "per_case_endpoints_sha256": _sha(csv_path),
        "execution_record_sha256": _sha(args.output / "execution_record.json"),
        "worker_success_count": sum(
            (args.output / f"worker_{index:02d}_SUCCESS.json").is_file()
            for index in range(args.worker_count)
        ),
        "locked_test_access": "prohibited", "post_fusion_used": False,
        "manual_or_ground_truth_prompts_used": False,
        "anatomical_inputs_used": False, "authorization_secret_recorded": False,
        "failures": failures,
    }
    _write_json(args.output / "integrity_audit.json", audit)
    if not failures:
        _write_json(args.output / "_SUCCESS.json", {
            "status": audit["status"],
            "integrity_audit_sha256": _sha(args.output / "integrity_audit.json"),
        })
    print(json.dumps(audit, indent=2, sort_keys=True))
    if failures:
        raise SystemExit(2)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("initialize", "worker", "finalize"))
    parser.add_argument("--repo-root", required=True, type=Path)
    parser.add_argument("--asset-project", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--smoke-audit", required=True, type=Path)
    parser.add_argument("--brainweb-array-root", required=True, type=Path)
    parser.add_argument("--hecktor-array-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--api-token-file", type=Path)
    parser.add_argument("--worker-count", type=int, default=8)
    parser.add_argument("--worker-index", type=int)
    args = parser.parse_args()
    if args.mode == "initialize":
        initialize(args)
    elif args.mode == "worker":
        if args.worker_index is None or args.api_token_file is None:
            parser.error("worker mode requires --worker-index and --api-token-file")
        worker(args)
    else:
        finalize(args)


if __name__ == "__main__":
    main()
