"""Measure frozen-case Stage 7 component efficiency without new API calls."""
from __future__ import annotations

import argparse
import csv
import json
import os
import time
from pathlib import Path

import numpy as np
from run_stage7_formal_smoke import HYPERPARAMETERS, _sha, _write_json
from run_stage7_formal_validation import _case_dir, _predictor, _source_path
from sam_kem.medsam import predict_prompt_object_logits
from sam_kem.metrics import nrmse, ssim
from sam_kem.operator_factory import build_discrete_radon
from sam_kem.reconstruction import (
    kem,
    normalize_kernel_feature,
    operator_support_mask,
    rbf_kernel,
)
from sam_kem.stage7_formal_runtime import aggregate_logits, render_pet_views
from sam_kem.stage7a_prompt_plan import PromptObject, det_component_objects


def _objects(records: list[dict]) -> tuple[PromptObject, ...]:
    return tuple(PromptObject(
        object_id=int(value["object_id"]), box_xyxy=tuple(value["box_xyxy"]),
        positive_points_xy=tuple(tuple(point) for point in value["positive_points_xy"]),
        negative_points_xy=tuple(tuple(point) for point in value["negative_points_xy"]),
        confidence=float(value["confidence"]),
    ) for value in records)


def _measure_arm(*, predictor, objects, mlem: np.ndarray, measured: np.ndarray,
                 target: np.ndarray, projection_scale: float, operator,
                 support: np.ndarray, dataset: str) -> tuple[dict, np.ndarray]:
    import torch

    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    logits = predict_prompt_object_logits(predictor, mlem, objects, output_shape=(128, 128))
    prior = aggregate_logits(logits)
    decoder_seconds = time.perf_counter() - started
    decoder_peak = int(torch.cuda.max_memory_allocated())
    hp = HYPERPARAMETERS[dataset]
    started = time.perf_counter()
    kernel = rbf_kernel(normalize_kernel_feature(prior), sigma=hp["sigma"],
                        radius=hp["radius"], support=support)
    kernel_seconds = time.perf_counter() - started
    started = time.perf_counter()
    reconstruction = kem(
        measured=measured, operator=operator, kernel=kernel,
        iterations=hp["iterations"],
        multiplicative=np.full_like(measured, projection_scale, dtype=np.float64),
        save_iterations=[hp["iterations"]],
    )[hp["iterations"]]
    kem_seconds = time.perf_counter() - started
    reconstruction = np.maximum(np.asarray(reconstruction, dtype=np.float32), 0)
    metrics = {
        "nrmse": nrmse(target, reconstruction, mask=support),
        "ssim": ssim(target, reconstruction, data_range=1.0, mask=support),
    }
    return {
        "decoder_seconds": decoder_seconds, "decoder_peak_gpu_bytes": decoder_peak,
        "kernel_seconds": kernel_seconds, "kem_seconds": kem_seconds, **metrics,
    }, reconstruction


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection", required=True, type=Path)
    parser.add_argument("--main-output", required=True, type=Path)
    parser.add_argument("--asset-project", required=True, type=Path)
    parser.add_argument("--brainweb-array-root", required=True, type=Path)
    parser.add_argument("--hecktor-array-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--gpu-index", type=int, default=0)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite efficiency namespace: {args.output}")
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_index)
    rows = list(csv.DictReader(args.selection.open(newline="", encoding="utf-8")))
    if len(rows) != 20:
        raise ValueError("efficiency selection must contain exactly 20 cases")
    operator = build_discrete_radon(repo_root=args.asset_project, size=128, angles=180)
    support = operator_support_mask(operator)
    predictors = {dataset: _predictor(args.asset_project, dataset)[0]
                  for dataset in ("brainweb", "hecktor")}
    output_rows = []
    failures = []
    for row in rows:
        case_dir = _case_dir(args.main_output, row)
        source = json.loads((case_dir / "result.json").read_text(encoding="utf-8"))
        dataset = row["dataset"]
        mlem = np.load(_source_path(row, "mlem10_input_path", args.brainweb_array_root,
                                    args.hecktor_array_root)).astype(np.float32)
        target = np.load(_source_path(row, "target_path", args.brainweb_array_root,
                                      args.hecktor_array_root)).astype(np.float32)
        measured = np.load(_source_path(row, "sinogram_path", args.brainweb_array_root,
                                        args.hecktor_array_root)).astype(np.float64)
        started = time.perf_counter()
        render_pet_views(mlem)
        render_seconds = time.perf_counter() - started
        arms = (
            ("policy", _objects(source["policy_objects"])),
            ("det_component", tuple(det_component_objects(mlem))),
        )
        for arm, objects in arms:
            timing, reconstruction = _measure_arm(
                predictor=predictors[dataset], objects=objects, mlem=mlem,
                measured=measured, target=target,
                projection_scale=float(row["projection_scale"]), operator=operator,
                support=support, dataset=dataset)
            source_endpoint = next(item for item in source["endpoints"] if (
                item["role"] == "policy" if arm == "policy"
                else item["role"] == "paired_comparator"))
            metric_match = (abs(timing["nrmse"] - source_endpoint["nrmse"]) <= 1e-12
                            and abs(timing["ssim"] - source_endpoint["ssim"]) <= 1e-12)
            if not metric_match:
                failures.append(f"metric mismatch {source['case_key']} {arm}")
            api_seconds = source["api_latency_seconds"] if arm == "policy" else 0.0
            usage = source["usage"] if arm == "policy" and isinstance(
                source["usage"], dict) else {}
            output_rows.append({
                "dataset": dataset, "subject_id": row["subject_id"],
                "slice_id": row["slice_id"], "count_level": int(row["count_level"]),
                "realization_id": int(row["realization_id"]), "arm": arm,
                "source_fallback_used": bool(source["fallback_reason"]),
                "render_seconds": render_seconds if arm == "policy" else 0.0,
                "api_seconds": api_seconds,
                "decoder_seconds": timing["decoder_seconds"],
                "kernel_seconds": timing["kernel_seconds"],
                "kem_seconds": timing["kem_seconds"],
                "compute_seconds": timing["decoder_seconds"] + timing["kernel_seconds"]
                + timing["kem_seconds"],
                "end_to_end_component_sum_seconds": render_seconds + api_seconds
                + timing["decoder_seconds"] + timing["kernel_seconds"]
                + timing["kem_seconds"] if arm == "policy" else timing["decoder_seconds"]
                + timing["kernel_seconds"] + timing["kem_seconds"],
                "decoder_peak_gpu_bytes": timing["decoder_peak_gpu_bytes"],
                "prompt_tokens": usage.get("prompt_tokens", 0),
                "completion_tokens": usage.get("completion_tokens", 0),
                "total_tokens": usage.get("total_tokens", 0),
                "nrmse": timing["nrmse"], "ssim": timing["ssim"],
                "metric_match": metric_match,
                "reconstruction_array_sha256": __import__("hashlib").sha256(
                    reconstruction.tobytes(order="C")).hexdigest(),
            })
    args.output.mkdir(parents=True)
    csv_path = args.output / "component_timing.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(output_rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(output_rows)
    summary = {}
    for arm in ("policy", "det_component"):
        values = [row for row in output_rows if row["arm"] == arm]
        summary[arm] = {
            field: {"median": float(np.median([row[field] for row in values])),
                    "mean": float(np.mean([row[field] for row in values]))}
            for field in ("render_seconds", "api_seconds", "decoder_seconds",
                          "kernel_seconds", "kem_seconds", "compute_seconds",
                          "end_to_end_component_sum_seconds", "decoder_peak_gpu_bytes",
                          "total_tokens")
        }
    audit = {
        "artifact_kind": "stage7_formal_component_efficiency",
        "status": "PASS_complete" if not failures else "FAIL_integrity",
        "case_count": len(rows), "endpoint_count": len(output_rows),
        "physical_gpu_index": args.gpu_index, "all_metrics_match": not failures,
        "component_timing_sha256": _sha(csv_path), "summary": summary,
        "new_api_calls": 0, "locked_test_access": "prohibited",
        "anatomical_inputs_used": False, "failures": failures,
    }
    _write_json(args.output / "integrity_audit.json", audit)
    print(json.dumps(audit, indent=2, sort_keys=True))
    if failures:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
