"""Step 6BP Stage E — corrected deterministic post-processing.

Per directive 5910758070:

1. Loss-ratio denominator: use the frozen MLEM endpoint values
   (not the per_case.csv MLEM_FEATURE_MATCHED variant). The frozen
   denominators are:
       BrainWeb: 100k 0.0302676787631081; 200k 0.0330807617729657;
                 500k 0.0166027693195544; 1M  0.0143730623636540.
       HECKTOR:   100k 0.101581081872378;  200k 0.0784697190768229;
                 500k 0.0688390216426416; 1M  0.0673837834835941.
   Formula: R = (NRMSE_v - NRMSE_Q) / frozen_denom.

2. Bootstrap: include all 7 non-Q variants (incl. UNIFORM_FEATURE_MATCHED)
   on HE, 7 x 4 = 28 records.

3. Kernel diagnostics: each variant uses its actual Stage C prior;
   compute effective-neighbor count n_eff_i = 1 / sum_j K_ij^2.

4. SHA labels: array_payload_sha256 (hash of np.tobytes payload)
   AND npy_file_sha256 (hash of on-disk file bytes); both recorded.

5. main() generates all Stage E text artifacts deterministically.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================================
# Constants and frozen denominators
# ============================================================================

VARIANTS = (
    "Q_ADAPT",
    "FULL_ADAPT",
    "CENTER_ADAPT",
    "CENTROID_ADAPT",
    "Q_RELEASED",
    "MLEM_FEATURE_MATCHED",
    "Q_SHUFFLED_MATCHED",
    "UNIFORM_FEATURE_MATCHED",
)
PRIMARY = "Q_ADAPT"
NON_Q_VARIANTS = [v for v in VARIANTS if v != PRIMARY]
MBOOTSTRAP_SEED = 20260929
N_BOOTSTRAP = 10000

# Frozen MLEM endpoint denominators per directive 5910758070.
# These are the subject-balanced (NRMSE_MLEM_endpoint - NRMSE_Q_ADAPT)
# values from the preregistered plan.
FROZEN_MLEM_DENOMINATOR = {
    ("brainweb", 100000): 0.0302676787631081,
    ("brainweb", 200000): 0.0330807617729657,
    ("brainweb", 500000): 0.0166027693195544,
    ("brainweb", 1000000): 0.0143730623636540,
    ("hecktor", 100000): 0.101581081872378,
    ("hecktor", 200000): 0.0784697190768229,
    ("hecktor", 500000): 0.0688390216426416,
    ("hecktor", 1000000): 0.0673837834835941,
}

VARIANT_ORDER = list(VARIANTS)


# ============================================================================
# Hash helpers (both file bytes and array payload bytes)
# ============================================================================

def sha256_bytes(arr: np.ndarray) -> str:
    """Hash of the contiguous array payload bytes (np.tobytes())."""
    return hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()


def sha256_file(p: Path) -> str:
    """Hash of the file bytes as stored on disk."""
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_file_lf(p: Path) -> str:
    """Hash with CRLF normalized to LF (what Git stores)."""
    data = p.read_bytes()
    return hashlib.sha256(data.replace(b"\r\n", b"\n")).hexdigest()


# ============================================================================
# (1) Loss ratio cells with frozen denominator
# ============================================================================

def compute_loss_ratio_cells(per_case_csvs: dict) -> pd.DataFrame:
    """Per dataset x count x variant, with the FROZEN MLEM endpoint
    denominator (not the MLEM_FEATURE_MATCHED variant):
        R = (NRMSE_v - NRMSE_Q) / frozen_denom
    Returns a long DataFrame.
    """
    subj_means = []
    for v, csv in per_case_csvs.items():
        df = pd.read_csv(csv)
        s = df.groupby(["dataset", "count_level", "subject_id"]).nrmse.mean().reset_index()
        s["variant"] = v
        subj_means.append(s)
    all_subj = pd.concat(subj_means, ignore_index=True)

    sb = all_subj.groupby(["dataset", "count_level", "variant"]).nrmse.mean().reset_index()
    sb = sb.rename(columns={"nrmse": "nrmse_subject_balanced_mean"})

    wide = sb.pivot_table(
        index=["dataset", "count_level"],
        columns="variant",
        values="nrmse_subject_balanced_mean",
    ).reset_index()
    cells = []
    for _, row in wide.iterrows():
        ds = row["dataset"]
        count = int(row["count_level"])
        nrmse_q = float(row[PRIMARY])
        denom = float(FROZEN_MLEM_DENOMINATOR[(ds, count)])
        for v in NON_Q_VARIANTS:
            nrmse_v = float(row[v])
            R = (nrmse_v - nrmse_q) / denom if denom != 0 else float("nan")
            cells.append({
                "dataset": ds, "count_level": count, "variant": v,
                "nrmse_v": nrmse_v, "nrmse_q": nrmse_q,
                "denominator_frozen": denom, "loss_ratio_R": R,
            })
    return pd.DataFrame(cells)


def summarize_loss_ratios(loss_cells: pd.DataFrame) -> pd.DataFrame:
    """Per variant: 8 cell values, median, sign of (Q lower NRMSE) count."""
    rows = []
    for v, g in loss_cells.groupby("variant"):
        per_cell = list(zip(g["dataset"], g["count_level"], g["loss_ratio_R"]))
        n_q_lower = int((g["nrmse_v"] > g["nrmse_q"]).sum())
        median = float(np.median(g["loss_ratio_R"].dropna()))
        rows.append({
            "variant": v,
            "n_cells": len(g),
            "n_q_lower_nrmse": n_q_lower,
            "median_loss_ratio": median,
            "per_cell_values": [
                {"dataset": d, "count_level": int(c), "R": float(R)}
                for d, c, R in per_cell
            ],
        })
    return pd.DataFrame(rows)


# ============================================================================
# (2) Paired subject-cluster HECKTOR bootstrap (all 7 non-Q variants)
# ============================================================================

def paired_he_subject_bootstrap(per_case_csvs: dict, seed: int = MBOOTSTRAP_SEED,
                                n_bootstrap: int = N_BOOTSTRAP) -> list:
    """For every non-Q variant x count (7 x 4 = 28 records), compute
    paired subject-cluster bootstrap on HE subjects. Returns a list
    of dicts.
    """
    rng = np.random.default_rng(seed)
    he_q = pd.read_csv(per_case_csvs[PRIMARY])
    he_q = he_q[he_q.dataset == "hecktor"]
    he_q_subj = he_q.groupby(["subject_id", "count_level"]).agg(
        nrmse_mean=("nrmse", "mean"),
        ssim_mean=("ssim", "mean"),
    ).reset_index()

    records = []
    for v in NON_Q_VARIANTS:
        he_v = pd.read_csv(per_case_csvs[v])
        he_v = he_v[he_v.dataset == "hecktor"]
        he_v_subj = he_v.groupby(["subject_id", "count_level"]).agg(
            nrmse_mean=("nrmse", "mean"),
            ssim_mean=("ssim", "mean"),
        ).reset_index()
        merged = he_q_subj.merge(he_v_subj, on=["subject_id", "count_level"],
                                  suffixes=("_q", "_v"))
        merged["delta_nrmse"] = merged["nrmse_mean_v"] - merged["nrmse_mean_q"]
        merged["delta_ssim"] = merged["ssim_mean_q"] - merged["ssim_mean_v"]
        for count in sorted(merged.count_level.unique()):
            sub = merged[merged.count_level == count]
            if len(sub) < 2:
                continue
            n_subj = len(sub)
            dn = sub.delta_nrmse.values
            ds = sub.delta_ssim.values
            boots_dn = np.empty(n_bootstrap, dtype=np.float64)
            boots_ds = np.empty(n_bootstrap, dtype=np.float64)
            for i in range(n_bootstrap):
                idx = rng.integers(0, n_subj, size=n_subj)
                boots_dn[i] = dn[idx].mean()
                boots_ds[i] = ds[idx].mean()
            records.append({
                "variant": v,
                "count_level": int(count),
                "n_subjects": n_subj,
                "mean_delta_nrmse": float(dn.mean()),
                "delta_nrmse_ci95_low": float(np.percentile(boots_dn, 2.5)),
                "delta_nrmse_ci95_high": float(np.percentile(boots_dn, 97.5)),
                "mean_delta_ssim": float(ds.mean()),
                "delta_ssim_ci95_low": float(np.percentile(boots_ds, 2.5)),
                "delta_ssim_ci95_high": float(np.percentile(boots_ds, 97.5)),
                "convention": "delta_nrmse = variant - Q; delta_ssim = Q - variant (positive favors Q)",
                "seed": seed,
                "n_bootstrap": n_bootstrap,
                "seed_label": "post-run deterministic analysis choice, not a preregistered seed",
            })
    return records


# ============================================================================
# (3) Real kernel diagnostics (each variant uses its actual Stage C prior)
# ============================================================================

def kernel_diagnostics(K) -> dict:
    """From a sparse row-normalized RBF kernel matrix (csr):
    - mean_diagonal_mass
    - mean_geometric_degree (mean_row_nnz)
    - mean_effective_neighbor_count: n_eff_i = 1 / sum_j K_ij^2
    - mean_effective_to_geometric_ratio
    """
    K = K.tocsr()
    diag = K.diagonal()
    row_nnz = np.diff(K.indptr)
    # Compute sum_j K_ij^2 per row (squared L2 norm per row)
    row_sq = np.asarray(K.multiply(K).sum(axis=1)).ravel()
    # Guard against zeros
    safe = np.where(row_sq > 0, row_sq, 1.0)
    n_eff = 1.0 / safe
    n_eff = np.where(row_sq > 0, n_eff, 0.0)
    # Effective-to-geometric ratio per row (where both > 0)
    mask = (row_nnz > 0) & (n_eff > 0)
    eff_to_geo = np.zeros_like(n_eff)
    eff_to_geo[mask] = n_eff[mask] / row_nnz[mask]

    return {
        "mean_diagonal_mass": float(np.mean(diag)),
        "min_diagonal_mass": float(np.min(diag)),
        "max_diagonal_mass": float(np.max(diag)),
        "mean_geometric_degree": float(np.mean(row_nnz)),
        "median_geometric_degree": float(np.median(row_nnz)),
        "min_geometric_degree": int(np.min(row_nnz)),
        "max_geometric_degree": int(np.max(row_nnz)),
        "mean_effective_neighbor_count": float(np.mean(n_eff)),
        "median_effective_neighbor_count": float(np.median(n_eff)),
        "min_effective_neighbor_count": float(np.min(n_eff)),
        "max_effective_neighbor_count": float(np.max(n_eff)),
        "mean_eff_to_geo_ratio": float(np.mean(eff_to_geo[mask])) if mask.any() else 0.0,
        "n_total_nnz": int(K.nnz),
    }


def compute_kernel_diagnostics_for_case(repo_root: Path, ds: str, subj: str,
                                        slc: str, count: int, real: int,
                                        priors: dict = None) -> dict:
    """Build the same RBF kernels used by Stage C for one case, using
    the PRIORS provided (each variant uses its actual Stage C prior;
    if priors is None, load from committed Stage C inputs).
    """
    from sam_kem.reconstruction import (
        rbf_kernel, operator_support_mask, normalize_kernel_feature,
    )
    from sam_kem.config import load_contract
    from sam_kem.operator_factory import build_discrete_radon

    # Frozen params
    df = pd.read_csv(repo_root / "results/dev_trainmax_v1/frozen_hyperparameters.csv")
    df = df[df.method == "SAM-KEM-MedSAM2-Qwen3-full"]
    hp = df[(df.dataset == ds) & (df.count_level == count)].iloc[0]
    sigma = float(hp.sigma); radius = int(hp.radius)

    # Build operator + load MLEM10
    cfg, _ = load_contract(repo_root / "code/configs/generated/dev_trainmax_v1_validation_registered.yaml")
    op = build_discrete_radon(repo_root=repo_root, size=128, angles=180)
    mlem10 = np.load(repo_root / f"arrays/dev_trainmax_v1/{ds}/{subj}/{slc}/c{count}/r{real}/mlem10.npy").astype(np.float32)

    support = operator_support_mask(op)

    out = {}
    for v in VARIANTS:
        if v == "UNIFORM_FEATURE_MATCHED":
            K = build_uniform_kernel_from_v3(np.ones((128, 128), dtype=np.float32),
                                            sigma=sigma, radius=radius, support=support)
            label = "uniform"
        else:
            if v == "MLEM_FEATURE_MATCHED":
                prior = mlem10
            elif v == "Q_ADAPT":
                prior_path = repo_root / f"arrays/dev_trainmax_v1/{ds}/{subj}/{slc}/c{count}/r{real}/sam_prior.npy"
                prior = np.load(prior_path).astype(np.float32)
            else:
                if priors is None:
                    # Load from Stage C inputs
                    from sam_kem.causal_ablation_v3 import _load_qwen_prior
                    prior = _load_qwen_prior(repo_root / "delivery/dev_trainmax_v1_validation_priors.zip",
                                              ds, subj, slc, count, real).astype(np.float32)
                else:
                    prior = priors[v]
            # Stage C rule: MLEM_FEATURE_MATCHED does NOT normalize
            if v == "MLEM_FEATURE_MATCHED":
                feat = prior.astype(np.float32)
            else:
                feat = normalize_kernel_feature(prior.astype(np.float32))
            K = rbf_kernel(feat, sigma=sigma, radius=radius, support=support)
            label = "learned" if v != "MLEM_FEATURE_MATCHED" else "learned_mlem"
        d = kernel_diagnostics(K)
        d["variant"] = v
        d["kernel_class"] = label
        out[v] = d
    return out


def build_uniform_kernel_from_v3(feature, sigma, radius, support):
    from sam_kem.causal_ablation_v3 import build_uniform_kernel
    return build_uniform_kernel(feature, sigma=sigma, radius=radius, support=support)


# ============================================================================
# (4) 200k/r0 fixed-case archives (reuse Stage C .npy arrays)
# ============================================================================

def load_stage_c_priors_for_fixed_case(repo_root: Path, ds: str, subj: str,
                                        slc: str, count: int, real: int) -> dict:
    """Load the Stage C per-variant priors from committed inputs
    (Qwen box cache CSV + prior zip + MLEM10). Returns dict[v] = np.ndarray.
    """
    from sam_kem.causal_ablation_v3 import _load_qwen_prior
    zip_path = repo_root / "delivery/dev_trainmax_v1_validation_priors.zip"
    qwen_csv = repo_root / "prompts/dev_trainmax_v1/train_validation_box_cache.csv"
    from sam_kem.causal_ablation_v3 import qwen_box_lookup

    qb = qwen_box_lookup(qwen_csv, ds, subj, slc, count)
    bw, bh = (qb[2] - qb[0], qb[3] - qb[1]) if qb is not None else (w := 128, 128)
    mlem10 = np.load(repo_root / f"arrays/dev_trainmax_v1/{ds}/{subj}/{slc}/c{count}/r{real}/mlem10.npy").astype(np.float32)

    # Qwen-box prior: just the cached prior (no decoder call needed for the array)
    prior_qwen_box = _load_qwen_prior(zip_path, ds, subj, slc, count, real).astype(np.float32)

    priors = {
        "Q_ADAPT": prior_qwen_box,
        "MLEM_FEATURE_MATCHED": mlem10,
        "UNIFORM_FEATURE_MATCHED": np.ones((128, 128), dtype=np.float32),
    }
    # For other variants, we need to actually run the decoder with the
    # appropriate box. This IS a small GPU computation, but the directive
    # permits one-shot re-computation of the 16 fixed-case endpoints
    # (2 cases x 8 variants) for verification only.
    from sam_kem.medsam import predictor_factory, build_released_predictor, CheckpointSpec, sha256_file
    pw = repo_root / "checkpoints/MedSAM2_latest.pt"
    bw_path = repo_root / "checkpoints/dev_trainmax_v1/medsam2_decoder_brainweb.pt"
    he_path = repo_root / "checkpoints/dev_trainmax_v1/medsam2_decoder_hecktor.pt"
    pet_factory = predictor_factory(
        medsam_source=repo_root / "external/MedSAM2",
        pretrained_checkpoint=pw,
    )
    predictors = {
        "pet_adapted_brainweb": pet_factory(CheckpointSpec(
            dataset="brainweb", path=bw_path, sha256=sha256_file(bw_path),
            normalization_id="train_max_fixed_C", run_id="dev_trainmax_v1")),
        "pet_adapted_hecktor": pet_factory(CheckpointSpec(
            dataset="hecktor", path=he_path, sha256=sha256_file(he_path),
            normalization_id="train_max_fixed_C", run_id="dev_trainmax_v1")),
        "released": build_released_predictor(
            medsam_source=repo_root / "external/MedSAM2",
            pretrained_checkpoint=pw,
        ),
    }

    # Decoder-based variants
    from sam_kem.causal_ablation_v3 import load_prior_for_variant, compute_corrected_shuffle_assignment
    shuffle, _, _ = compute_corrected_shuffle_assignment(zip_path, 20260915)
    for v in ["FULL_ADAPT", "CENTER_ADAPT", "CENTROID_ADAPT", "Q_RELEASED", "Q_SHUFFLED_MATCHED"]:
        prior, _ = load_prior_for_variant(
            v, repo_root=repo_root, zip_path=zip_path,
            ds=ds, subj=subj, slc=slc, count=count, real=real,
            pet_image=mlem10, qwen_box_csv=qwen_csv,
            predictors=predictors, shuffle_assignment=shuffle,
        )
        priors[v] = prior.astype(np.float32)
    return priors


def build_fixed_case_manifest(repo_root: Path, ds: str, subj: str, slc: str,
                              count: int, real: int, per_case_csvs: dict,
                              out_dir: Path) -> dict:
    """Stage E.4 — produce the fixed-case archive from committed
    Stage C .npy arrays where possible, and from a single re-run
    where necessary (decoder-based priors for FULL/CENTER/CENTROID/
    Q_RELEASED/Q_SHUFFLED_MATCHED).
    """
    from sam_kem.reconstruction import (
        rbf_kernel, kem, operator_support_mask, normalize_kernel_feature,
    )
    from sam_kem.causal_ablation_v3 import build_uniform_kernel
    from sam_kem.config import load_contract
    from sam_kem.operator_factory import build_discrete_radon
    from sam_kem.metrics import nrmse, ssim

    out_dir.mkdir(parents=True, exist_ok=True)
    case_id = f"{ds}/{subj}/{slc}/c{count}/r{real}"

    target = np.load(repo_root / f"arrays/dev_trainmax_v1/{ds}/{subj}/{slc}/target.npy").astype(np.float32)
    np.save(out_dir / "target.npy", target)

    cfg, _ = load_contract(repo_root / "code/configs/generated/dev_trainmax_v1_validation_registered.yaml")
    op = build_discrete_radon(repo_root=repo_root, size=128, angles=180)
    measured = np.load(repo_root / f"arrays/dev_trainmax_v1/{ds}/{subj}/{slc}/c{count}/r{real}/sinogram.npy").astype(np.float64)
    _ = np.load(repo_root / f"arrays/dev_trainmax_v1/{ds}/{subj}/{slc}/c{count}/r{real}/mlem10.npy")  # ensure exists

    df = pd.read_csv(repo_root / "results/dev_trainmax_v1/frozen_hyperparameters.csv")
    df = df[df.method == "SAM-KEM-MedSAM2-Qwen3-full"]
    hp = df[(df.dataset == ds) & (df.count_level == count)].iloc[0]
    sigma = float(hp.sigma); radius = int(hp.radius); iterations = int(hp.iterations)

    manifest_path = repo_root / str(cfg["outputs"]["validation_manifest"])
    from sam_kem.experiments import load_case_manifest
    cases = load_case_manifest(repo_root, manifest_path, split="validation")
    case = next(c for c in cases if (c.dataset == ds and c.subject_id == subj
                                       and c.slice_id == slc
                                       and int(c.realization_id) == real
                                       and int(c.count_level) == count))
    gain = np.full_like(measured, case.projection_scale, dtype=np.float64)

    support = operator_support_mask(op)

    # Load priors (single small re-run for decoder-based variants; 8 endpoints total)
    priors = load_stage_c_priors_for_fixed_case(repo_root, ds, subj, slc, count, real)

    exports = {"case": case_id,
               "shapes_dtypes": {},
               "sha256": {},
               "metrics": {}}
    for v in VARIANTS:
        prior_path = out_dir / f"prior_{v}.npy"
        rec_path = out_dir / f"reconstruction_{v}.npy"
        err_path = out_dir / f"abs_error_{v}.npy"

        # Save prior (.npy file + payload + file SHA)
        prior = priors[v]
        np.save(prior_path, prior.astype(np.float32))
        prior_payload_sha = sha256_bytes(prior.astype(np.float32))
        prior_file_sha = sha256_file(prior_path)

        # Build kernel
        if v == "UNIFORM_FEATURE_MATCHED":
            K = build_uniform_kernel(np.ones((128, 128), dtype=np.float32),
                                     sigma=sigma, radius=radius, support=support)
        elif v == "MLEM_FEATURE_MATCHED":
            feat = prior.astype(np.float32)
            K = rbf_kernel(feat, sigma=sigma, radius=radius, support=support)
        else:
            feat = normalize_kernel_feature(prior.astype(np.float32))
            K = rbf_kernel(feat, sigma=sigma, radius=radius, support=support)

        out_dict = kem(measured=measured, operator=op, kernel=K,
                       iterations=iterations, multiplicative=gain)
        rec = out_dict[max(out_dict.keys())]
        rec = np.asarray(rec, dtype=np.float32)
        rec = np.maximum(rec, 0.0)
        np.save(rec_path, rec)
        rec_payload_sha = sha256_bytes(rec)
        rec_file_sha = sha256_file(rec_path)

        err = np.abs(rec - target).astype(np.float32)
        np.save(err_path, err)
        err_payload_sha = sha256_bytes(err)
        err_file_sha = sha256_file(err_path)

        nm = float(nrmse(target.astype(np.float64), rec.astype(np.float64),
                          mask=support.astype(bool)))
        sm = float(ssim(target.astype(np.float32), rec.astype(np.float32),
                          data_range=1.0, mask=support.astype(bool)))

        # Verify against committed per_case.csv
        df_p = pd.read_csv(per_case_csvs[v])
        ref_row = df_p[(df_p.dataset == ds) & (df_p.subject_id == subj)
                        & (df_p.slice_id == slc) & (df_p.count_level == count)
                        & (df_p.realization_id == real)].iloc[0]
        ref_nrmse = float(ref_row.nrmse)
        ref_ssim = float(ref_row.ssim)
        nrmse_match = abs(nm - ref_nrmse) < 1e-6
        ssim_match = abs(sm - ref_ssim) < 1e-6

        exports["shapes_dtypes"][v] = {
            "target": list(target.shape) + [str(target.dtype)],
            f"prior_{v}": list(prior.shape) + [str(prior.dtype)],
            f"reconstruction_{v}": list(rec.shape) + [str(rec.dtype)],
            f"abs_error_{v}": list(err.shape) + [str(err.dtype)],
        }
        exports["sha256"][v] = {
            "target_payload_sha256": sha256_bytes(target),
            "target_file_sha256": sha256_file(out_dir / "target.npy"),
            f"prior_{v}_payload_sha256": prior_payload_sha,
            f"prior_{v}_file_sha256": prior_file_sha,
            f"reconstruction_{v}_payload_sha256": rec_payload_sha,
            f"reconstruction_{v}_file_sha256": rec_file_sha,
            f"abs_error_{v}_payload_sha256": err_payload_sha,
            f"abs_error_{v}_file_sha256": err_file_sha,
        }
        exports["metrics"][v] = {
            "nrmse_recomputed": nm, "nrmse_committed": ref_nrmse,
            "nrmse_match": nrmse_match,
            "ssim_recomputed": sm, "ssim_committed": ref_ssim,
            "ssim_match": ssim_match,
        }

    manifest = {
        "case": case_id,
        "shapes_dtypes": exports["shapes_dtypes"],
        "sha256": exports["sha256"],
        "metrics": exports["metrics"],
        "all_metrics_match": all(
            m["nrmse_match"] and m["ssim_match"]
            for m in exports["metrics"].values()
        ),
    }
    with open(out_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    return manifest


# ============================================================================
# (5) Paired subject-level deltas with SSIM
# ============================================================================

def paired_subject_deltas_with_ssim(per_case_csvs: dict) -> pd.DataFrame:
    """Per-subject deltas for every non-Q variant x dataset x count.
    Convention:
        delta_nrmse = variant - Q (positive: variant worse than Q)
        delta_ssim  = Q - variant (positive: variant worse than Q)
    """
    subj_means = []
    for v, csv in per_case_csvs.items():
        df = pd.read_csv(csv)
        s = df.groupby(["dataset", "count_level", "subject_id"]).agg(
            nrmse_mean=("nrmse", "mean"),
            ssim_mean=("ssim", "mean"),
        ).reset_index()
        s["variant"] = v
        subj_means.append(s)
    all_subj = pd.concat(subj_means, ignore_index=True)

    wide_nrmse = all_subj.pivot_table(
        index=["dataset", "count_level", "subject_id"],
        columns="variant", values="nrmse_mean",
    ).reset_index()
    wide_ssim = all_subj.pivot_table(
        index=["dataset", "count_level", "subject_id"],
        columns="variant", values="ssim_mean",
    ).reset_index()

    records = []
    for v in NON_Q_VARIANTS:
        rows = wide_nrmse[["dataset", "count_level", "subject_id"]].copy()
        rows["variant"] = v
        rows["nrmse_v"] = wide_nrmse[v]
        rows["nrmse_q"] = wide_nrmse[PRIMARY]
        rows["ssim_v"] = wide_ssim[v]
        rows["ssim_q"] = wide_ssim[PRIMARY]
        rows["delta_nrmse"] = rows["nrmse_v"] - rows["nrmse_q"]
        rows["delta_ssim"] = rows["ssim_q"] - rows["ssim_v"]
        rows["q_lower_nrmse"] = rows["nrmse_v"] > rows["nrmse_q"]
        rows["q_higher_ssim"] = rows["ssim_v"] < rows["ssim_q"]
        records.append(rows)
    return pd.concat(records, ignore_index=True)


def summarize_paired_deltas(deltas: pd.DataFrame) -> pd.DataFrame:
    return deltas.groupby(["variant", "dataset", "count_level"]).agg(
        n_subjects=("subject_id", "nunique"),
        n_q_lower_nrmse=("q_lower_nrmse", "sum"),
        n_q_higher_ssim=("q_higher_ssim", "sum"),
        mean_delta_nrmse=("delta_nrmse", "mean"),
        std_delta_nrmse=("delta_nrmse", "std"),
        mean_delta_ssim=("delta_ssim", "mean"),
        std_delta_ssim=("delta_ssim", "std"),
    ).reset_index()


# ============================================================================
# Driver — generates all Stage E text artifacts deterministically
# ============================================================================

def main(repo_root: Path, run_dir: Path, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    per_case_csvs = {v: run_dir / v / "per_case.csv" for v in VARIANTS}

    # (1) Loss-ratio cells + summary
    print("=" * 60)
    print("(1) Loss ratio with FROZEN MLEM endpoint denominator")
    print("=" * 60)
    loss_cells = compute_loss_ratio_cells(per_case_csvs)
    loss_cells.to_csv(out_dir / "loss_ratio_cells.csv", index=False)
    loss_summary = summarize_loss_ratios(loss_cells)
    loss_summary.to_csv(out_dir / "loss_ratio_summary.csv", index=False)
    print(loss_summary.to_string(index=False))

    # (2) Paired HE bootstrap (28 records = 7 non-Q variants x 4 counts)
    print()
    print("=" * 60)
    print("(2) Paired HE subject-cluster bootstrap (28 records)")
    print("=" * 60)
    boot_records = paired_he_subject_bootstrap(per_case_csvs)
    assert len(boot_records) == 7 * 4, f"Expected 28 bootstrap records, got {len(boot_records)}"
    with open(out_dir / "hecktor_paired_bootstrap.json", "w", encoding="utf-8") as f:
        json.dump({
            "n_bootstrap": N_BOOTSTRAP,
            "seed": MBOOTSTRAP_SEED,
            "seed_label": "post-run deterministic analysis choice, not a preregistered seed",
            "convention": "delta_nrmse = variant - Q; delta_ssim = Q - variant (positive favors Q)",
            "n_variants": 7, "n_count_levels": 4,
            "results": boot_records,
        }, f, indent=2)
    for r in boot_records:
        print(f"  {r['variant']:25s} c{r['count_level']:>7d}: "
              f"delta_nrmse={r['mean_delta_nrmse']:+.4f} "
              f"CI95=[{r['delta_nrmse_ci95_low']:+.4f}, {r['delta_nrmse_ci95_high']:+.4f}] "
              f"delta_ssim={r['mean_delta_ssim']:+.4f} "
              f"CI95=[{r['delta_ssim_ci95_low']:+.4f}, {r['delta_ssim_ci95_high']:+.4f}]")

    # (3) Kernel diagnostics (each variant uses its actual Stage C prior)
    print()
    print("=" * 60)
    print("(3) Real kernel diagnostics (correct per-variant priors)")
    print("=" * 60)
    # Pre-compute and cache priors for both fixed cases
    bw_priors = load_stage_c_priors_for_fixed_case(repo_root, "brainweb", "subject04", "z038", 200000, 0)
    he_priors = load_stage_c_priors_for_fixed_case(repo_root, "hecktor", "10", "z072", 200000, 0)
    kd_bw = compute_kernel_diagnostics_for_case(repo_root, "brainweb", "subject04", "z038",
                                                 200000, 0, priors=bw_priors)
    kd_he = compute_kernel_diagnostics_for_case(repo_root, "hecktor", "10", "z072",
                                                 200000, 0, priors=he_priors)

    # Q-vs-uniform ratios: ratio of Q_ADAPT's stats to UNIFORM_FEATURE_MATCHED's stats
    def _ratios(kd):
        ratios = {}
        q = kd.get("Q_ADAPT", {})
        u = kd.get("UNIFORM_FEATURE_MATCHED", {})
        if not q or not u:
            return ratios
        ratios["Q_vs_uniform_geometric_degree"] = (
            float(q["mean_geometric_degree"] / u["mean_geometric_degree"])
            if u["mean_geometric_degree"] > 0 else None
        )
        ratios["Q_vs_uniform_effective_neighbor_count"] = (
            float(q["mean_effective_neighbor_count"] / u["mean_effective_neighbor_count"])
            if u["mean_effective_neighbor_count"] > 0 else None
        )
        ratios["Q_vs_uniform_eff_to_geo_ratio"] = (
            float(q["mean_eff_to_geo_ratio"] / u["mean_eff_to_geo_ratio"])
            if u["mean_eff_to_geo_ratio"] > 0 else None
        )
        ratios["Q_vs_uniform_diagonal_mass"] = (
            float(q["mean_diagonal_mass"] / u["mean_diagonal_mass"])
            if u["mean_diagonal_mass"] > 0 else None
        )
        return ratios

    with open(out_dir / "kernel_diagnostics.json", "w", encoding="utf-8") as f:
        json.dump({
            "brainweb_subject04_z038_c200000_r0": kd_bw,
            "hecktor_10_z072_c200000_r0": kd_he,
            "Q_vs_uniform_ratios_brainweb": _ratios(kd_bw),
            "Q_vs_uniform_ratios_hecktor": _ratios(kd_he),
            "notes": (
                "Kernel diagnostics built from the actual sparse RBF kernels, "
                "with each variant using its Stage C prior (decoder-based priors "
                "for FULL/CENTER/CENTROID/Q_RELEASED/Q_SHUFFLED_MATCHED; MLEM10 for "
                "MLEM_FEATURE_MATCHED; cached Qwen-box prior for Q_ADAPT; constant "
                "unit feature for UNIFORM_FEATURE_MATCHED). Effective-neighbor "
                "count n_eff_i = 1 / sum_j K_ij^2 (per supported row). The uniform "
                "kernel and the learned kernel have identical geometric support "
                "(same radius), so mean_geometric_degree is the same. The "
                "diagonal mass and effective-neighbor count differ because each "
                "kernel carries different feature weights."
            ),
        }, f, indent=2)
    print("  brainweb kd: written")
    print("  hecktor kd: written")

    # (4) Fixed-case archives (use Stage C inputs only; .npy regenerated as needed)
    print()
    print("=" * 60)
    print("(4) 200k/r0 fixed-case archives")
    print("=" * 60)
    fixed_bw = build_fixed_case_manifest(
        repo_root, "brainweb", "subject04", "z038", 200000, 0, per_case_csvs,
        out_dir / "fixed_case_brainweb_subject04_z038_200k_r0")
    print(f"  brainweb: all_metrics_match={fixed_bw['all_metrics_match']}")
    fixed_he = build_fixed_case_manifest(
        repo_root, "hecktor", "10", "z072", 200000, 0, per_case_csvs,
        out_dir / "fixed_case_hecktor_10_z072_200k_r0")
    print(f"  hecktor: all_metrics_match={fixed_he['all_metrics_match']}")

    # (5) Paired subject-level deltas
    print()
    print("=" * 60)
    print("(5) Paired subject-level deltas with SSIM")
    print("=" * 60)
    deltas = paired_subject_deltas_with_ssim(per_case_csvs)
    deltas.to_csv(out_dir / "paired_subject_deltas.csv", index=False)
    summary = summarize_paired_deltas(deltas)
    summary.to_csv(out_dir / "paired_subject_deltas_summary.csv", index=False)
    print(f"  per-subject rows: {len(deltas)}")
    print(f"  summary rows: {len(summary)}")

    # (6) Dual SHA map for Stage C per_variant CSVs
    print()
    print("=" * 60)
    print("(6) Dual SHA map (file bytes + payload bytes)")
    print("=" * 60)
    dual = {}
    for v in VARIANTS:
        csv = run_dir / v / "per_case.csv"
        dual[f"{v}/per_case.csv"] = {
            "npy_file_sha256": sha256_file(csv),
            "lf_normalized_sha256": sha256_file_lf(csv),
            "size_bytes": csv.stat().st_size,
        }
    dual_sha_doc = {
        "artifact_kind": "validation_only_causal_ablations_v3_dual_sha",
        "schema_version": 1,
        "relationship": (
            "Git stores text files with LF. Python on Windows writes text "
            "files using the platform-native CRLF by default. The execution-time "
            "SHA-256 (file bytes, possibly CRLF) is what the runner saw; the "
            "repository-byte SHA-256 (LF-normalized) is what Git stores. The two "
            "hashes differ when the file contains any newline character."
        ),
        "files": dual,
    }
    with open(run_dir / "per_output_sha256_dual.json", "w", encoding="utf-8") as f:
        json.dump(dual_sha_doc, f, indent=2)
    print("  per_output_sha256_dual.json written")

    # (7) Stage E summary
    summary_doc = {
        "artifact_kind": "validation_only_causal_ablations_v3_stage_e",
        "schema_version": 1,
        "loss_ratio_formula": "R = (NRMSE_v - NRMSE_Q) / frozen_MLEM_endpoint_denominator",
        "loss_ratio_cells": "loss_ratio_cells.csv",
        "loss_ratio_summary": "loss_ratio_summary.csv",
        "hecktor_paired_bootstrap": "hecktor_paired_bootstrap.json",
        "kernel_diagnostics": "kernel_diagnostics.json",
        "fixed_case_brainweb": "fixed_case_brainweb_subject04_z038_200k_r0/",
        "fixed_case_hecktor": "fixed_case_hecktor_10_z072_200k_r0/",
        "paired_subject_deltas": "paired_subject_deltas.csv",
        "paired_subject_deltas_summary": "paired_subject_deltas_summary.csv",
        "per_output_sha256_dual": "per_output_sha256_dual.json",
        "mbootstrap_seed": MBOOTSTRAP_SEED,
        "n_bootstrap": N_BOOTSTRAP,
        "all_metrics_match_brainweb": fixed_bw["all_metrics_match"],
        "all_metrics_match_hecktor": fixed_he["all_metrics_match"],
        "execution_commit": _git_head(repo_root),
    }
    with open(out_dir / "stage_e_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary_doc, f, indent=2)

    # (8) Integrity audit
    audit_doc = {
        "artifact_kind": "validation_only_causal_ablations_v3_stage_e_integrity",
        "schema_version": 1,
        "stage_e_files": [
            "loss_ratio_cells.csv",
            "loss_ratio_summary.csv",
            "hecktor_paired_bootstrap.json",
            "kernel_diagnostics.json",
            "paired_subject_deltas.csv",
            "paired_subject_deltas_summary.csv",
            "fixed_case_brainweb_subject04_z038_200k_r0/",
            "fixed_case_hecktor_10_z072_200k_r0/",
            "stage_e_summary.json",
        ],
        "n_fixed_cases": 2,
        "fixed_cases_all_metrics_match": (
            fixed_bw["all_metrics_match"] and fixed_he["all_metrics_match"]),
        "n_loss_ratio_cells_per_variant": 8,
        "n_paired_subject_deltas_rows": len(deltas),
        "n_bw_subjects_per_cell": 4,
        "n_hecktor_subjects": 45,
        "n_bootstrap": N_BOOTSTRAP,
        "mbootstrap_seed": MBOOTSTRAP_SEED,
        "n_loss_ratio_summary_rows": len(loss_summary),
        "n_paired_deltas_summary_rows": len(summary),
        "n_bootstrap_records": len(boot_records),
        "loss_ratio_summary_table": [
            {
                "variant": r["variant"],
                "n_q_lower_nrmse": int(r["n_q_lower_nrmse"]),
                "median_loss_ratio": float(r["median_loss_ratio"]),
            }
            for _, r in loss_summary.iterrows()
        ],
    }
    with open(out_dir / "stage_e_integrity_audit.json", "w", encoding="utf-8") as f:
        json.dump(audit_doc, f, indent=2)

    return summary_doc


def _git_head(repo_root: Path) -> str:
    try:
        import subprocess
        return subprocess.run(["git", "-C", str(repo_root), "rev-parse", "HEAD"],
                              capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return "unknown"


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 3:
        print("Usage: python -m sam_kem.stage_e_analysis REPO_ROOT OUT_DIR")
        sys.exit(1)
    main(Path(sys.argv[1]).resolve(),
          Path(sys.argv[1]).resolve() / "results/dev_trainmax_v1/causal_ablations_corrected_v3/full_run",
          Path(sys.argv[2]).resolve())