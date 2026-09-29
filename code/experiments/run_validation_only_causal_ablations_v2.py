"""Step 6BM Stage A — corrected 8-variant runner using canonical KEM endpoint."""
from __future__ import annotations

import argparse
import csv
import json
import multiprocessing as mp
import re
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from sam_kem.causal_ablation_v2 import (
    VARIANTS,
    load_prior_for_variant,
    compute_corrected_shuffle_assignment,
    build_uniform_kernel,
)


# ============================================================================
# Frozen SAM-KEM hyperparameters (per dataset × count)
# ============================================================================

def load_frozen_sam_hyperparameters(repo_root: Path) -> dict:
    df = pd.read_csv(repo_root / "results/dev_trainmax_v1/frozen_hyperparameters.csv")
    df = df[df["method"] == "SAM-KEM-MedSAM2-Qwen3-full"]
    out = {}
    for _, row in df.iterrows():
        out[(row["dataset"], int(row["count_level"]))] = (
            float(row["sigma"]), int(row["radius"]), int(row["neighbours"]),
            int(row["iterations"]), float(row["alpha"]),
        )
    return out


# ============================================================================
# Per-case worker (canonical pipeline)
# ============================================================================

def _worker_init_predictors(repo_root_str: str, qwen_box_csv_str: str,
                            zip_path_str: str, shuffle_seed: int):
    """Lazy predictor init + operator/manifest/contract + frozen params + shuffle map."""
    del qwen_box_csv_str  # currently unused; kept for API symmetry
    repo_root = Path(repo_root_str)
    zip_path = Path(zip_path_str)

    from sam_kem.experiments import load_case_manifest, validate_case_contract
    from sam_kem.config import load_contract
    from sam_kem.medsam import sha256_file

    config_path = repo_root / "code/configs/generated/dev_trainmax_v1_validation_registered.yaml"
    cfg, contract = load_contract(config_path)
    manifest_path = repo_root / str(cfg["outputs"]["validation_manifest"])
    cases = load_case_manifest(repo_root, manifest_path, split="validation")
    validate_case_contract(cases, contract)
    case_by_key = {
        (c.dataset, c.subject_id, c.slice_id, int(c.realization_id), int(c.count_level)): c
        for c in cases
    }

    froz = load_frozen_sam_hyperparameters(repo_root)
    bw_path = repo_root / "checkpoints/dev_trainmax_v1/medsam2_decoder_brainweb.pt"
    he_path = repo_root / "checkpoints/dev_trainmax_v1/medsam2_decoder_hecktor.pt"
    bw_sha = sha256_file(bw_path)
    he_sha = sha256_file(he_path)
    pw = repo_root / "checkpoints/MedSAM2_latest.pt"

    shuffle, shuffle_sha, shuffle_summary = compute_corrected_shuffle_assignment(zip_path, shuffle_seed)

    return {
        "repo_root": str(repo_root),
        "cases": case_by_key,
        "froz": froz,
        "pw": str(pw),
        "bw_path": str(bw_path), "bw_sha": bw_sha,
        "he_path": str(he_path), "he_sha": he_sha,
        "shuffle": shuffle,
        "shuffle_sha": shuffle_sha,
        "shuffle_summary": shuffle_summary,
    }


def _process_case_worker(args):
    """Process one (variant, case) using canonical KEM endpoint."""
    (variant, ds, subj, slc, count, real, repo_root_str, zip_path_str,
     qwen_box_csv_str, shuffle_seed) = args
    repo_root = Path(repo_root_str)
    zip_path = Path(zip_path_str)
    qwen_box_csv = Path(qwen_box_csv_str)

    if not hasattr(_process_case_worker, "_ctx"):
        _process_case_worker._ctx = _worker_init_predictors(
            repo_root_str, qwen_box_csv_str, zip_path_str, shuffle_seed)
    ctx = _process_case_worker._ctx
    case_key = (ds, subj, slc, int(real), int(count))
    case = ctx["cases"].get(case_key)
    if case is None:
        return {"variant": variant, "error": f"case not found: {case_key}"}

    froz = ctx["froz"]
    hp = froz.get((ds, int(count)))
    if hp is None:
        return {"variant": variant, "error": f"no frozen SAM hp for ({ds}, {count})"}
    sigma, radius, neighbours, iterations, alpha = hp

    measured = np.load(case.sinogram_path).astype(np.float64)
    target = np.load(case.target_path).astype(np.float32)

    from sam_kem.experiments import compute_mlem_input
    from sam_kem.config import load_contract as _lc
    from sam_kem.operator_factory import build_discrete_radon

    cfg, _ = _lc(repo_root / "code/configs/generated/dev_trainmax_v1_validation_registered.yaml")
    m_size = re.search(r"size(\d+)", case.operator_id)
    m_angles = re.search(r"n(\d+)", case.operator_id)
    if not m_size or not m_angles:
        return {"variant": variant, "error": f"can't parse operator_id: {case.operator_id}"}
    size = int(m_size.group(1))
    angles = int(m_angles.group(1))
    op = build_discrete_radon(repo_root=repo_root, size=size, angles=angles)
    measured = measured.astype(np.float64)
    mlem10 = compute_mlem_input(case, measured, op, cfg)
    if mlem10 is None:
        return {"variant": variant, "error": "compute_mlem_input returned None"}

    feature = None
    source = "n/a"
    if variant == "UNIFORM_FEATURE_MATCHED":
        feature = np.ones((128, 128), dtype=np.float32)
        source = "constant_unit_128"
    elif variant == "MLEM_FEATURE_MATCHED":
        feature = mlem10.astype(np.float32)
        source = "mlem10_cached"
    else:
        if variant in ("FULL_ADAPT", "CENTER_ADAPT", "CENTROID_ADAPT", "Q_RELEASED"):
            from sam_kem.medsam import predictor_factory, build_released_predictor, CheckpointSpec
            if "_predictors_cache" not in ctx:
                ctx["_predictors_cache"] = {}
            cache = ctx["_predictors_cache"]
            if "pet_bw" not in cache:
                cache["pet_bw"] = predictor_factory(
                    medsam_source=repo_root / "external/MedSAM2",
                    pretrained_checkpoint=Path(ctx["pw"]),
                )(CheckpointSpec(
                    dataset="brainweb", path=Path(ctx["bw_path"]), sha256=ctx["bw_sha"],
                    normalization_id="train_max_fixed_C", run_id="dev_trainmax_v1",
                ))
                cache["pet_he"] = predictor_factory(
                    medsam_source=repo_root / "external/MedSAM2",
                    pretrained_checkpoint=Path(ctx["pw"]),
                )(CheckpointSpec(
                    dataset="hecktor", path=Path(ctx["he_path"]), sha256=ctx["he_sha"],
                    normalization_id="train_max_fixed_C", run_id="dev_trainmax_v1",
                ))
                cache["released"] = build_released_predictor(
                    medsam_source=repo_root / "external/MedSAM2",
                    pretrained_checkpoint=Path(ctx["pw"]),
                )()
            predictors = {"pet_adapted_brainweb": cache["pet_bw"],
                          "pet_adapted_hecktor": cache["pet_he"],
                          "released": cache["released"]}
        else:
            predictors = None

        pet_input = mlem10.astype(np.float32)
        if variant == "Q_SHUFFLED_MATCHED":
            prior, source = load_prior_for_variant(
                variant, repo_root=repo_root, zip_path=zip_path,
                ds=ds, subj=subj, slc=slc, count=int(count), real=int(real),
                pet_image=pet_input, qwen_box_csv=qwen_box_csv,
                predictors=predictors, shuffle_assignment=ctx["shuffle"],
            )
        else:
            prior, source = load_prior_for_variant(
                variant, repo_root=repo_root, zip_path=zip_path,
                ds=ds, subj=subj, slc=slc, count=int(count), real=int(real),
                pet_image=pet_input, qwen_box_csv=qwen_box_csv,
                predictors=predictors,
            )
        from sam_kem.reconstruction import normalize_kernel_feature
        feature = normalize_kernel_feature(prior)

    from sam_kem.reconstruction import rbf_kernel, kem, operator_support_mask
    support = operator_support_mask(op)
    if variant == "UNIFORM_FEATURE_MATCHED":
        K = build_uniform_kernel(feature, sigma=sigma, radius=radius, support=support)
    else:
        K = rbf_kernel(feature, sigma=sigma, radius=radius, support=support)

    gain = np.full_like(measured, case.projection_scale, dtype=np.float64)
    out_dict = kem(measured=measured, operator=op, kernel=K,
                   iterations=iterations, multiplicative=gain)
    image = out_dict[max(out_dict.keys())]
    image = np.asarray(image, dtype=np.float32)
    image = np.maximum(image, 0.0)
    if not np.all(np.isfinite(image)):
        return {"variant": variant, "error": "non-finite reconstruction"}

    from sam_kem.metrics import nrmse as _nrmse, ssim as _ssim
    nm = _nrmse(target.astype(np.float64), image.astype(np.float64),
                mask=support.astype(bool))
    sm = _ssim(target.astype(np.float32), image.astype(np.float32),
               data_range=1.0, mask=support.astype(bool))

    return {
        "variant": variant,
        "dataset": ds, "subject_id": subj, "slice_id": slc,
        "count_level": int(count), "realization_id": int(real),
        "nrmse": float(nm), "ssim": float(sm),
        "sigma": sigma, "radius": radius, "neighbours": neighbours,
        "iterations": iterations, "alpha": alpha,
        "source": source,
    }


# ============================================================================
# Driver
# ============================================================================

def collect_cases(repo_root: Path, shards: Optional[list[str]] = None,
                  max_count: Optional[int] = None) -> list[tuple]:
    from sam_kem.experiments import load_case_manifest, validate_case_contract
    from sam_kem.config import load_contract

    cfg, contract = load_contract(repo_root / "code/configs/generated/dev_trainmax_v1_validation_registered.yaml")
    manifest_path = repo_root / str(cfg["outputs"]["validation_manifest"])
    cases = load_case_manifest(repo_root, manifest_path, split="validation")
    validate_case_contract(cases, contract)
    out = []
    for c in cases:
        if max_count is not None and int(c.count_level) > max_count:
            continue
        if shards:
            sh = f"{c.dataset}/{c.subject_id}/{c.slice_id}"
            if sh not in shards:
                continue
        out.append((c.dataset, c.subject_id, c.slice_id, int(c.count_level), int(c.realization_id)))
    return sorted(set(out))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--variants", nargs="*", default=list(VARIANTS))
    parser.add_argument("--shards", nargs="*", default=None)
    parser.add_argument("--max-count", type=int, default=None)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--qwen-box-csv", type=Path,
                        default=Path("prompts/dev_trainmax_v1/train_validation_box_cache.csv"))
    parser.add_argument("--shuffle-seed", type=int, default=20260915)
    parser.add_argument("--release-check-only", action="store_true")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.release_check_only:
        ctx = _worker_init_predictors(
            str(args.root),
            str(args.root / args.qwen_box_csv),
            str(args.root / "delivery/dev_trainmax_v1_validation_priors.zip"),
            args.shuffle_seed,
        )
        summary = {
            "artifact_kind": "validation_only_causal_ablations_v2_release_check",
            "schema_version": 1,
            "status": "PASS_ready_for_smoke",
            "variants": list(args.variants),
            "release_check_only": True,
            "formal_output_created": False,
            "frozen_hyperparameters_created": False,
            "locked_config_created": False,
            "immutable_archive_uri_modified": False,
            "test_locked_access": "prohibited",
            "qwen_called": False,
            "new_reconstruction_executed": False,
            "shuffle_derangement_sha256": ctx["shuffle_sha"],
            "shuffle_summary": ctx["shuffle_summary"],
        }
        with open(args.summary, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        return 0

    cases = collect_cases(args.root, shards=args.shards, max_count=args.max_count)
    print(f"  cases: {len(cases)}")

    repo_root_str = str(args.root)
    zip_path_str = str(args.root / "delivery/dev_trainmax_v1_validation_priors.zip")
    qwen_box_csv_str = str(args.root / args.qwen_box_csv)
    jobs = []
    for variant in args.variants:
        for (ds, subj, slc, count, real) in cases:
            jobs.append((variant, ds, subj, slc, count, real,
                         repo_root_str, zip_path_str, qwen_box_csv_str,
                         args.shuffle_seed))

    t0 = time.time()
    results = []
    with mp.Pool(processes=args.workers) as pool:
        for i, r in enumerate(pool.imap_unordered(_process_case_worker, jobs, chunksize=1)):
            results.append(r)
            if (i + 1) % 50 == 0:
                print(f"  [{i+1}/{len(jobs)}] elapsed {time.time()-t0:.1f}s")
    elapsed = time.time() - t0
    print(f"  done in {elapsed:.1f}s")

    errors = [r for r in results if "error" in r]
    print(f"  errors: {len(errors)}/{len(results)}")
    if errors:
        print(f"  first error: {errors[0]}")

    by_variant = defaultdict(list)
    for r in results:
        if "error" in r:
            continue
        by_variant[r["variant"]].append(r)

    for variant, rows in by_variant.items():
        vdir = args.output_dir / variant
        vdir.mkdir(parents=True, exist_ok=True)
        csv_path = vdir / "per_case.csv"
        with open(csv_path, "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"  {variant}: {len(rows)} rows -> {csv_path}")

    summary = {
        "artifact_kind": "validation_only_causal_ablations_v2",
        "schema_version": 1,
        "status": "complete" if not errors else "complete_with_errors",
        "wall_time_seconds": elapsed,
        "variants": list(args.variants),
        "total_cases": len(cases),
        "errors": len(errors),
        "shards": args.shards,
    }
    with open(args.summary, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())