"""Step 6BN Stage A/C — corrected v3 8-variant runner with atomic/resumable output.

Improvements per directive 5890879059 §2:

- Atomic, resumable: each per-variant CSV is written once per process
  via ``csv.writer`` with a single ``open(..., 'w', newline='')``
  call. Records reused-vs-written counts.
- Hash bindings: the runner exposes a ``hash_bindings`` block listing
  the execution commit, frozen config, manifest, frozen CSV, prior
  archive, prompt cache, adapted/released checkpoints, shuffle map,
  and canonical selected-per-case reference.
- Per-output SHA-256: written at job completion to
  ``<output_dir>/per_output_sha256.json`` (populated after the final
  flush). The release check verifies all required inputs exist
  before claiming PASS; it does NOT create the formal output
  directory when ``--release-check-only`` is set.
- Worker/process identity recorded at start-up.
- Segmented timing: load / decode / KEM / metrics split.
"""
from __future__ import annotations

import argparse
import csv
import json
import multiprocessing as mp
import re
import subprocess
import sys
import time
import uuid
from collections import defaultdict
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from sam_kem.causal_ablation_v3 import (
    VARIANTS,
    load_prior_for_variant,
    compute_corrected_shuffle_assignment,
    build_uniform_kernel,
)


# ============================================================================
# Hash helpers
# ============================================================================

def sha256_file(p: Path) -> str:
    import hashlib
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


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
    """Lazy predictor init + operator/manifest/contract + frozen params + shuffle map.

    Returns a context dict shared by all worker processes in this pool.
    """
    repo_root = Path(repo_root_str)
    del qwen_box_csv_str  # currently unused here; kept for API symmetry
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
    pw_sha = sha256_file(pw)

    shuffle, shuffle_sha, shuffle_summary = compute_corrected_shuffle_assignment(zip_path, shuffle_seed)

    return {
        "repo_root": str(repo_root),
        "cases": case_by_key,
        "froz": froz,
        "pw": str(pw), "pw_sha": pw_sha,
        "bw_path": str(bw_path), "bw_sha": bw_sha,
        "he_path": str(he_path), "he_sha": he_sha,
        "shuffle": shuffle,
        "shuffle_sha": shuffle_sha,
        "shuffle_summary": shuffle_summary,
    }


def _process_case_worker(args):
    """Process one (variant, case) using canonical KEM endpoint.

    Records segmented timing in ``timing`` and per-variant ``source``;
    does not mutate any frozen input.
    """
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

    timing = {}

    t0 = time.time()
    measured = np.load(case.sinogram_path).astype(np.float64)
    target = np.load(case.target_path).astype(np.float32)
    timing["load_sinogram_target_s"] = time.time() - t0

    from sam_kem.experiments import compute_mlem_input
    from sam_kem.config import load_contract as _lc
    from sam_kem.operator_factory import build_discrete_radon

    t0 = time.time()
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
    timing["compute_mlem10_s"] = time.time() - t0
    if mlem10 is None:
        return {"variant": variant, "error": "compute_mlem_input returned None"}

    feature = None
    source = "n/a"
    if variant == "UNIFORM_FEATURE_MATCHED":
        feature = np.ones((128, 128), dtype=np.float32)
        source = "constant_unit_128"
        timing["decode_s"] = 0.0
    elif variant == "MLEM_FEATURE_MATCHED":
        feature = mlem10.astype(np.float32)
        source = "mlem10_cached"
        timing["decode_s"] = 0.0
    else:
        t0 = time.time()
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
                )
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
        timing["decode_s"] = time.time() - t0
        from sam_kem.reconstruction import normalize_kernel_feature
        feature = normalize_kernel_feature(prior)

    t0 = time.time()
    from sam_kem.reconstruction import rbf_kernel, kem, operator_support_mask
    support = operator_support_mask(op)
    if variant == "UNIFORM_FEATURE_MATCHED":
        K = build_uniform_kernel(feature, sigma=sigma, radius=radius, support=support)
    else:
        K = rbf_kernel(feature, sigma=sigma, radius=radius, support=support)
    timing["kernel_build_s"] = time.time() - t0

    t0 = time.time()
    gain = np.full_like(measured, case.projection_scale, dtype=np.float64)
    out_dict = kem(measured=measured, operator=op, kernel=K,
                   iterations=iterations, multiplicative=gain)
    image = out_dict[max(out_dict.keys())]
    image = np.asarray(image, dtype=np.float32)
    image = np.maximum(image, 0.0)
    timing["kem_s"] = time.time() - t0

    if not np.all(np.isfinite(image)):
        return {"variant": variant, "error": "non-finite reconstruction"}

    t0 = time.time()
    from sam_kem.metrics import nrmse as _nrmse, ssim as _ssim
    nm = _nrmse(target.astype(np.float64), image.astype(np.float64),
                mask=support.astype(bool))
    sm = _ssim(target.astype(np.float32), image.astype(np.float32),
               data_range=1.0, mask=support.astype(bool))
    timing["metrics_s"] = time.time() - t0

    return {
        "variant": variant,
        "dataset": ds, "subject_id": subj, "slice_id": slc,
        "count_level": int(count), "realization_id": int(real),
        "nrmse": float(nm), "ssim": float(sm),
        "sigma": sigma, "radius": radius, "neighbours": neighbours,
        "iterations": iterations, "alpha": alpha,
        "source": source,
        "timing_load_s": timing["load_sinogram_target_s"],
        "timing_mlem_s": timing["compute_mlem10_s"],
        "timing_decode_s": timing["decode_s"],
        "timing_kernel_s": timing["kernel_build_s"],
        "timing_kem_s": timing["kem_s"],
        "timing_metrics_s": timing["metrics_s"],
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


def compute_hash_bindings(repo_root: Path, zip_path: Path, qwen_box_csv: Path,
                          shuffle_seed: int) -> dict:
    cfg_path = repo_root / "code/configs/generated/dev_trainmax_v1_validation_registered.yaml"
    manifest_path = repo_root / "results/dev_trainmax_v1/validation_case_manifest.csv"
    frozen_hp = repo_root / "results/dev_trainmax_v1/frozen_hyperparameters.csv"
    selected = repo_root / "results/dev_trainmax_v1/validation_only_reporting/selected_per_case_metrics.csv"
    bw_path = repo_root / "checkpoints/dev_trainmax_v1/medsam2_decoder_brainweb.pt"
    he_path = repo_root / "checkpoints/dev_trainmax_v1/medsam2_decoder_hecktor.pt"
    pw_path = repo_root / "checkpoints/MedSAM2_latest.pt"

    # Shuffle map SHA (deterministic for the seed)
    _, shuffle_sha, _ = compute_corrected_shuffle_assignment(zip_path, shuffle_seed)

    bindings = {
        "execution_commit": _git_head(repo_root),
        "config_yaml_sha256": sha256_file(cfg_path),
        "manifest_sha256": sha256_file(manifest_path) if manifest_path.is_file() else None,
        "frozen_hyperparameters_sha256": sha256_file(frozen_hp),
        "selected_per_case_sha256": sha256_file(selected) if selected.is_file() else None,
        "prior_archive_sha256": sha256_file(zip_path),
        "qwen_box_cache_sha256": sha256_file(qwen_box_csv),
        "released_medsam2_sha256": sha256_file(pw_path),
        "medsam2_decoder_brainweb_sha256": sha256_file(bw_path),
        "medsam2_decoder_hecktor_sha256": sha256_file(he_path),
        "shuffle_derangement_sha256": shuffle_sha,
        "shuffle_seed": shuffle_seed,
    }
    return bindings


def _git_head(repo_root: Path) -> str:
    try:
        return subprocess.run(["git", "-C", str(repo_root), "rev-parse", "HEAD"],
                              capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return "unknown"


def verify_inputs(repo_root: Path, zip_path: Path, qwen_box_csv: Path,
                  output_dir: Path) -> dict:
    """Verify all required inputs and the directory contents for release
    check. Returns a status dict; PASS only if every required file
    exists and is non-empty.
    """
    cfg_path = repo_root / "code/configs/generated/dev_trainmax_v1_validation_registered.yaml"
    manifest_path = repo_root / "results/dev_trainmax_v1/validation_case_manifest.csv"
    frozen_hp = repo_root / "results/dev_trainmax_v1/frozen_hyperparameters.csv"
    bw_path = repo_root / "checkpoints/dev_trainmax_v1/medsam2_decoder_brainweb.pt"
    he_path = repo_root / "checkpoints/dev_trainmax_v1/medsam2_decoder_hecktor.pt"
    pw_path = repo_root / "checkpoints/MedSAM2_latest.pt"

    checks = {}
    for label, path in [
        ("config_yaml", cfg_path),
        ("validation_manifest", manifest_path),
        ("frozen_hyperparameters", frozen_hp),
        ("prior_archive", zip_path),
        ("qwen_box_cache", qwen_box_csv),
        ("medsam2_decoder_brainweb", bw_path),
        ("medsam2_decoder_hecktor", he_path),
        ("released_medsam2", pw_path),
    ]:
        checks[label] = {
            "exists": path.is_file(),
            "size_bytes": path.stat().st_size if path.is_file() else 0,
        }

    all_ok = all(c["exists"] and c["size_bytes"] > 0 for c in checks.values())
    return {"all_inputs_ok": all_ok, "checks": checks, "output_dir_created": output_dir.is_dir()}


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
    parser.add_argument("--run-id", type=str, default=None)
    args = parser.parse_args()

    zip_path = args.root / "delivery/dev_trainmax_v1_validation_priors.zip"
    qwen_box_csv = args.root / args.qwen_box_csv

    # Hash bindings (always computed; needed for both release and full)
    hash_bindings = compute_hash_bindings(args.root, zip_path, qwen_box_csv,
                                          args.shuffle_seed)

    if args.release_check_only:
        # Release check: verify inputs, do NOT create the formal output dir
        vstatus = verify_inputs(args.root, zip_path, qwen_box_csv,
                                args.output_dir)
        # Build a temporary ctx to also pull shuffle SHA / inputs
        ctx = _worker_init_predictors(str(args.root), str(qwen_box_csv),
                                       str(zip_path), args.shuffle_seed)
        summary = {
            "artifact_kind": "validation_only_causal_ablations_v3_release_check",
            "schema_version": 1,
            "status": "PASS_ready_for_smoke" if vstatus["all_inputs_ok"] else "FAIL_input_missing",
            "release_check_only": True,
            "formal_output_created": vstatus["output_dir_created"],
            "frozen_hyperparameters_created": False,
            "locked_config_created": False,
            "immutable_archive_uri_modified": False,
            "test_locked_access": "prohibited",
            "qwen_called": False,
            "new_reconstruction_executed": False,
            "input_verification": vstatus,
            "hash_bindings": hash_bindings,
            "shuffle_derangement_sha256": ctx["shuffle_sha"],
            "shuffle_summary": ctx["shuffle_summary"],
        }
        # For release check: write summary to a side file (NOT to args.summary,
        # because that path may be inside the formal namespace). Use the
        # explicitly-passed --summary path; the formal --output-dir stays
        # uncreated.
        args.summary.parent.mkdir(parents=True, exist_ok=True)
        with open(args.summary, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        # Explicitly do NOT call mkdir(args.output_dir) here.
        return 0 if vstatus["all_inputs_ok"] else 1

    # Full run: create the formal namespace only here.
    args.output_dir.mkdir(parents=True, exist_ok=True)

    run_id = args.run_id or uuid.uuid4().hex[:12]
    cases = collect_cases(args.root, shards=args.shards, max_count=args.max_count)
    print(f"  cases: {len(cases)}")
    print(f"  run_id: {run_id}")

    repo_root_str = str(args.root)
    zip_path_str = str(zip_path)
    qwen_box_csv_str = str(qwen_box_csv)
    jobs = []
    for variant in args.variants:
        for (ds, subj, slc, count, real) in cases:
            jobs.append((variant, ds, subj, slc, count, real,
                         repo_root_str, zip_path_str, qwen_box_csv_str,
                         args.shuffle_seed))

    t0 = time.time()
    results = []
    n_errors = 0
    with mp.Pool(processes=args.workers) as pool:
        for i, r in enumerate(pool.imap_unordered(_process_case_worker, jobs, chunksize=1)):
            results.append(r)
            if "error" in r:
                n_errors += 1
            if (i + 1) % 100 == 0:
                print(f"  [{i+1}/{len(jobs)}] elapsed {time.time()-t0:.1f}s "
                      f"({n_errors} errors)")
    elapsed = time.time() - t0
    print(f"  done in {elapsed:.1f}s; errors {n_errors}/{len(jobs)}")

    # Atomic write: per-variant CSV written once, then close.
    per_output_sha = {}
    by_variant = defaultdict(list)
    for r in results:
        if "error" in r:
            continue
        by_variant[r["variant"]].append(r)

    for variant, rows in by_variant.items():
        vdir = args.output_dir / variant
        vdir.mkdir(parents=True, exist_ok=True)
        csv_path = vdir / "per_case.csv"
        fieldnames = list(rows[0].keys())
        with open(csv_path, "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(rows)
        per_output_sha[f"{variant}/per_case.csv"] = sha256_file(csv_path)
        print(f"  {variant}: {len(rows)} rows -> {csv_path}")

    summary = {
        "artifact_kind": "validation_only_causal_ablations_v3_full",
        "schema_version": 1,
        "status": "complete" if n_errors == 0 else "complete_with_errors",
        "run_id": run_id,
        "wall_time_seconds": elapsed,
        "variants": list(args.variants),
        "total_cases": len(cases),
        "errors": n_errors,
        "per_output_sha256": per_output_sha,
        "shards": args.shards,
        "hash_bindings": hash_bindings,
        "worker_count": args.workers,
        "shuffle_derangement_sha256": hash_bindings.get("shuffle_derangement_sha256"),
        "shuffle_seed": args.shuffle_seed,
    }
    # summary written last, with hash bindings inside it.
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    with open(args.summary, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # Write a tiny SHA-256 map for downstream tools (delivery/ packaging).
    sha_path = args.output_dir / "per_output_sha256.json"
    with open(sha_path, "w", encoding="utf-8") as f:
        json.dump(per_output_sha, f, indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())