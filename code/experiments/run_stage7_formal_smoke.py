"""Run the four-case Stage 7 formal Agent-vs-DET smoke transactionally."""
from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import json
import os
import shutil
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path

import numpy as np

from sam_kem.medsam import (
    CheckpointSpec,
    predict_prompt_object_logits,
    predictor_factory,
    sha256_file,
)
from sam_kem.metrics import nrmse, ssim
from sam_kem.operator_factory import build_discrete_radon
from sam_kem.reconstruction import (
    kem,
    normalize_kernel_feature,
    operator_support_mask,
    rbf_kernel,
)
from sam_kem.stage7_formal_runtime import (
    FormalPromptError,
    aggregate_logits,
    parse_formal_prompt,
    prompt_objects_json,
    render_pet_views,
)
from sam_kem.stage7a_prompt_plan import det_component_objects

API_URL = "https://api.x5m5x.com/v1/chat/completions"
MODEL = "qwen3.8-max"
HYPERPARAMETERS = {
    "brainweb": {"sigma": 0.1, "radius": 2, "iterations": 20},
    "hecktor": {"sigma": 0.1, "radius": 5, "iterations": 20},
}


def _json_bytes(value: object) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()


def _write_json(path: Path, value: object) -> None:
    path.write_bytes(_json_bytes(value))


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _api_call(*, token: str, prompt: str, png: bytes) -> tuple[dict, dict, float]:
    request_payload = {
        "model": MODEL,
        "temperature": 0,
        "max_tokens": 500,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url":
                    "data:image/png;base64," + base64.b64encode(png).decode()}},
            ],
        }],
    }
    http_request = urllib.request.Request(
        API_URL, data=json.dumps(request_payload, separators=(",", ":")).encode(),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {token}"},
        method="POST",
    )
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(http_request, timeout=90) as response:
            response_payload = json.loads(response.read())
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as error:
        response_payload = {"transport_error": f"{type(error).__name__}: {error}"}
    latency = time.perf_counter() - started
    return request_payload, response_payload, latency


def _response_content(payload: dict) -> str:
    try:
        content = payload["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as error:
        raise FormalPromptError("API response has no assistant content") from error
    if not isinstance(content, str):
        raise FormalPromptError("API assistant content is not text")
    return content


def _reconstruct(*, prior: np.ndarray, measured: np.ndarray, target: np.ndarray,
                 projection_scale: float, operator, support: np.ndarray,
                 dataset: str) -> tuple[np.ndarray, dict]:
    hp = HYPERPARAMETERS[dataset]
    kernel = rbf_kernel(
        normalize_kernel_feature(prior), sigma=hp["sigma"], radius=hp["radius"],
        support=support,
    )
    reconstruction = kem(
        measured=measured, operator=operator, kernel=kernel,
        iterations=hp["iterations"],
        multiplicative=np.full_like(measured, projection_scale, dtype=np.float64),
        save_iterations=[hp["iterations"]],
    )[hp["iterations"]]
    reconstruction = np.maximum(np.asarray(reconstruction, dtype=np.float32), 0)
    metrics = {
        "nrmse": nrmse(target, reconstruction, mask=support),
        "ssim": ssim(target, reconstruction, data_range=1.0, mask=support),
    }
    if not np.isfinite(reconstruction).all() or not all(np.isfinite(list(metrics.values()))):
        raise RuntimeError("non-finite reconstruction or metric")
    return reconstruction, metrics


def _case_name(row: dict[str, str]) -> str:
    return "__".join((row["dataset"], row["subject_id"], row["slice_id"],
                       f"c{row['count_level']}", f"r{row['realization_id']}"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", required=True, type=Path)
    parser.add_argument("--asset-project", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--api-token-file", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite output namespace: {args.output}")
    token = args.api_token_file.read_text(encoding="utf-8").strip()
    if not token:
        raise SystemExit("API token file is empty")
    rows = list(csv.DictReader(args.manifest.open(newline="", encoding="utf-8")))
    if len(rows) != 4:
        raise SystemExit("formal smoke requires exactly four manifest rows")

    temporary = args.output.with_name(f".{args.output.name}.{uuid.uuid4().hex}.tmp")
    temporary.mkdir(parents=True)
    try:
        operator = build_discrete_radon(repo_root=args.asset_project, size=128, angles=180)
        support = operator_support_mask(operator)
        base_checkpoint = args.asset_project / "checkpoints/MedSAM2_latest.pt"
        medsam_source = args.asset_project / "external/MedSAM2"
        predictors = {}
        checkpoint_bindings = {}
        for dataset in ("brainweb", "hecktor"):
            checkpoint = args.asset_project / f"checkpoints/dev_trainmax_v1/medsam2_decoder_{dataset}.pt"
            digest = sha256_file(checkpoint)
            predictors[dataset] = predictor_factory(
                medsam_source=medsam_source, pretrained_checkpoint=base_checkpoint,
            )(CheckpointSpec(
                dataset=dataset, path=checkpoint, sha256=digest,
                normalization_id="train_max_fixed_C", run_id="dev_trainmax_v1",
            ))
            checkpoint_bindings[dataset] = digest

        result_rows = []
        total_api_calls = 0
        started_run = time.perf_counter()
        for row in rows:
            case_started = time.perf_counter()
            case_dir = temporary / _case_name(row)
            case_dir.mkdir()
            dataset = row["dataset"]
            mlem10 = np.load(row["mlem10_path"]).astype(np.float32)
            target = np.load(row["target_path"]).astype(np.float32)
            measured = np.load(row["sinogram_path"]).astype(np.float64)
            png, image_sha = render_pet_views(mlem10)
            (case_dir / "vlm_input.png").write_bytes(png)
            prompt_path = args.repo_root / f"code/configs/prompts/stage7_formal_{dataset}_v1.txt"
            prompt = prompt_path.read_text(encoding="utf-8")
            request_payload, response_payload, latency = _api_call(
                token=token, prompt=prompt, png=png,
            )
            total_api_calls += 1
            _write_json(case_dir / "raw_request.json", request_payload)
            _write_json(case_dir / "raw_response.json", response_payload)
            fallback_reason = ""
            try:
                agent_objects = parse_formal_prompt(
                    _response_content(response_payload), dataset=dataset,
                )
                receipt_ok = isinstance(response_payload.get("model"), str) and isinstance(
                    response_payload.get("usage"), dict
                )
                if not receipt_ok:
                    raise FormalPromptError("API receipt lacks returned model or usage")
                executed_method = (
                    "BW_AGENT_LOCAL_POS" if dataset == "brainweb"
                    else "HE_AGENT_LESION_BPN"
                )
            except FormalPromptError as error:
                fallback_reason = str(error)
                agent_objects = tuple(det_component_objects(mlem10))
                executed_method = "DET_COMPONENT_FALLBACK"

            det_objects = tuple(det_component_objects(mlem10))
            for requested_method, executed, objects in (
                ("DET_COMPONENT", "DET_COMPONENT", det_objects),
                (
                    "BW_AGENT_LOCAL_POS" if dataset == "brainweb" else "HE_AGENT_LESION_BPN",
                    executed_method, agent_objects,
                ),
            ):
                decode_started = time.perf_counter()
                object_logits = predict_prompt_object_logits(
                    predictors[dataset], mlem10, objects, output_shape=(128, 128),
                )
                prior = aggregate_logits(object_logits)
                decode_seconds = time.perf_counter() - decode_started
                reconstruction, metrics = _reconstruct(
                    prior=prior, measured=measured, target=target,
                    projection_scale=float(row["projection_scale"]),
                    operator=operator, support=support, dataset=dataset,
                )
                stem = "det" if requested_method == "DET_COMPONENT" else "policy"
                np.save(case_dir / f"{stem}_prior.npy", prior, allow_pickle=False)
                np.save(case_dir / f"{stem}_reconstruction.npy", reconstruction, allow_pickle=False)
                result_rows.append({
                    "dataset": dataset, "subject_id": row["subject_id"],
                    "slice_id": row["slice_id"], "count_level": int(row["count_level"]),
                    "realization_id": int(row["realization_id"]),
                    "requested_method": requested_method, "executed_method": executed,
                    "fallback_used": executed.endswith("FALLBACK"),
                    "fallback_reason": fallback_reason if stem == "policy" else "",
                    "object_count": len(objects), "nrmse": metrics["nrmse"],
                    "ssim": metrics["ssim"], "decode_seconds": decode_seconds,
                    "prior_sha256": _sha(case_dir / f"{stem}_prior.npy"),
                    "reconstruction_sha256": _sha(case_dir / f"{stem}_reconstruction.npy"),
                })
            receipt = {
                "case": _case_name(row), "api_calls": 1, "retries": 0,
                "requested_model": MODEL, "returned_model": response_payload.get("model"),
                "usage": response_payload.get("usage"), "latency_seconds": latency,
                "image_sha256": image_sha, "raw_request_sha256": _sha(case_dir / "raw_request.json"),
                "raw_response_sha256": _sha(case_dir / "raw_response.json"),
                "authorization_recorded": False, "executed_method": executed_method,
                "fallback_reason": fallback_reason,
                "objects": prompt_objects_json(agent_objects),
                "elapsed_seconds": time.perf_counter() - case_started,
            }
            _write_json(case_dir / "receipt.json", receipt)
            _write_json(case_dir / "_SUCCESS.json", {
                "case": _case_name(row), "receipt_sha256": _sha(case_dir / "receipt.json"),
            })

        fields = list(result_rows[0])
        with (temporary / "per_case.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
            writer.writeheader()
            writer.writerows(result_rows)
        policy_rows = [row for row in result_rows if row["requested_method"] != "DET_COMPONENT"]
        det_by_case = {
            (row["dataset"], row["subject_id"], row["slice_id"]): row
            for row in result_rows if row["requested_method"] == "DET_COMPONENT"
        }
        deltas = []
        for row in policy_rows:
            det = det_by_case[(row["dataset"], row["subject_id"], row["slice_id"])]
            deltas.append({
                "case": "/".join((row["dataset"], row["subject_id"], row["slice_id"])),
                "delta_nrmse_policy_minus_det": row["nrmse"] - det["nrmse"],
                "delta_ssim_policy_minus_det": row["ssim"] - det["ssim"],
            })
        audit = {
            "artifact_kind": "stage7_formal_four_case_smoke",
            "status": "PASS" if len(result_rows) == 8 else "FAIL",
            "case_count": 4, "endpoint_count": len(result_rows),
            "api_calls": total_api_calls, "api_retries": 0,
            "valid_agent_responses": sum(not row["fallback_used"] for row in policy_rows),
            "fallback_responses": sum(row["fallback_used"] for row in policy_rows),
            "all_metrics_finite": all(
                np.isfinite(row["nrmse"]) and np.isfinite(row["ssim"]) for row in result_rows
            ),
            "deltas": deltas, "checkpoint_sha256": checkpoint_bindings,
            "base_checkpoint_sha256": sha256_file(base_checkpoint),
            "manifest_sha256": sha256_file(args.manifest),
            "per_case_csv_sha256": _sha(temporary / "per_case.csv"),
            "elapsed_seconds": time.perf_counter() - started_run,
            "locked_test_access": "prohibited", "post_fusion_used": False,
            "manual_or_ground_truth_prompts_used": False,
            "anatomical_inputs_used": False, "authorization_secret_recorded": False,
        }
        _write_json(temporary / "integrity_audit.json", audit)
        _write_json(temporary / "_SUCCESS.json", {
            "status": audit["status"],
            "integrity_audit_sha256": _sha(temporary / "integrity_audit.json"),
        })
        os.replace(temporary, args.output)
        print(json.dumps(audit, indent=2, sort_keys=True))
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


if __name__ == "__main__":
    main()
