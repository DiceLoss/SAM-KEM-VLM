"""Step 6BO Stage E — deterministic post-processing analysis for the v3
causal ablation package.

Per directive 5908110622, the 71,136 Stage C endpoint computation is
accepted; only deterministic post-processing is performed here.

This module reads the committed per-variant per_case.csv files plus the
frozen inputs, and produces:

1. Corrected loss-ratio formula:
   R = (NRMSE_v - NRMSE_Q) / (NRMSE_MLEM - NRMSE_Q)
   computed per dataset x count from subject-balanced means.
   Reports 8 cell values per variant + median + direction count.

2. Paired subject-cluster HECKTOR bootstrap (NRMSE and SSIM) for every
   non-Q variant x count. 10,000 resamples, seed 20260929. The seed
   is explicitly labelled as a post-run deterministic analysis choice
   (not a preregistered seed).

3. Real kernel diagnostics from the actual sparse kernels:
   mean diagonal mass, effective-neighbor count, geometric
   degree/support, learned-vs-uniform ratios. Computed by re-running
   the same KEM pipeline on the two fixed 200k/r0 cases.

4. 200k/r0 fixed-case archives:
   - BrainWeb subject04/z038/c200000/r0
   - HECKTOR 10/z072/c200000/r0
   For each: target, prior(s), 8 pure-KEM reconstructions, absolute
   error arrays with shapes, dtypes, and SHA-256. The recomputed
   NRMSE/SSIM for each reconstruction is verified against its
   committed per_case.csv row.

5. Paired subject-level deltas for both NRMSE and SSIM with the
   explicit convention:
       delta_nrmse = variant - Q
       delta_ssim  = Q - variant
   so positive always favors Q. Per-subject rows plus dataset x count
   summaries and direction counts. Includes all four BW subjects; no
   pooling of slices as independent subjects.

The runner module `run_validation_only_causal_ablations_v3.py` was
corrected to remove the misleading "atomic/resumable" claims; it
writes per-variant CSVs in a single flush but does not maintain worker
PIDs or per-job reuse counts.

This tool is purely deterministic: re-running on the same committed
inputs produces the same outputs (modulo the bootstrap seed).
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================================
# Variant list and primary control
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
LEARNED_VARIANTS = [
    "FULL_ADAPT", "CENTER_ADAPT", "CENTROID_ADAPT",
    "Q_RELEASED", "MLEM_FEATURE_MATCHED", "Q_SHUFFLED_MATCHED",
]
MBOOTSTRAP_SEED = 20260929
N_BOOTSTRAP = 10000


# ============================================================================
# Hash helper
# ============================================================================

def sha256_bytes(arr: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ============================================================================
# Loss ratio (corrected formula)
# ============================================================================

def compute_loss_ratio_cells(per_case_csvs: dict) -> pd.DataFrame:
    """Per dataset x count x variant: subject-balanced NRMSE mean.
    Then per cell:
        R = (NRMSE_v - NRMSE_Q) / (NRMSE_MLEM - NRMSE_Q)

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
        ds, count = row["dataset"], int(row["count_level"])
        nrmse_q = float(row[PRIMARY])
        nrmse_mlem = float(row["MLEM_FEATURE_MATCHED"])
        denom = nrmse_mlem - nrmse_q
        for v in VARIANTS:
            if v == PRIMARY:
                continue
            nrmse_v = float(row[v])
            R = (nrmse_v - nrmse_q) / denom if denom != 0 else float("nan")
            cells.append({
                "dataset": ds, "count_level": count, "variant": v,
                "nrmse_v": nrmse_v, "nrmse_q": nrmse_q,
                "nrmse_mlem": nrmse_mlem, "loss_ratio_R": R,
            })
    return pd.DataFrame(cells)


def summarize_loss_ratios(loss_cells: pd.DataFrame) -> pd.DataFrame:
    """Per variant: 8 cell values, median, sign of (Q lower NRMSE) count."""
    rows = []
    for v, g in loss_cells.groupby("variant"):
        per_cell = list(zip(g["dataset"], g["count_level"], g["loss_ratio_R"]))
        # Q is "lower NRMSE" iff nrmse_v > nrmse_q iff loss_ratio_R > 0
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
# Paired subject-cluster HECKTOR bootstrap
# ============================================================================

def paired_he_subject_bootstrap(per_case_csvs: dict, seed: int = MBOOTSTRAP_SEED,
                                n_bootstrap: int = N_BOOTSTRAP) -> list:
    """For every non-Q variant x count, compute paired subject-cluster
    bootstrap on HE subjects. Returns a list of dicts.
    """
    rng = np.random.default_rng(seed)
    he_q = pd.read_csv(per_case_csvs[PRIMARY])
    he_q = he_q[he_q.dataset == "hecktor"]
    he_q_subj = he_q.groupby(["subject_id", "count_level"]).agg(
        nrmse_mean=("nrmse", "mean"),
        ssim_mean=("ssim", "mean"),
    ).reset_index()

    records = []
    for v in LEARNED_VARIANTS:
        if v == PRIMARY:
            continue
        he_v = pd.read_csv(per_case_csvs[v])
        he_v = he_v[he_v.dataset == "hecktor"]
        he_v_subj = he_v.groupby(["subject_id", "count_level"]).agg(
            nrmse_mean=("nrmse", "mean"),
            ssim_mean=("ssim", "mean"),
        ).reset_index()
        merged = he_q_subj.merge(he_v_subj, on=["subject_id", "count_level"],
                                  suffixes=("_q", "_v"))
        # Convention: delta_nrmse = v - Q, delta_ssim = Q - v (so positive favors Q)
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
# Real kernel diagnostics
# ============================================================================

def kernel_diagnostics(K) -> dict:
    """From a sparse row-normalized RBF kernel matrix (csr): mean
    diagonal mass, mean effective-neighbor count, geometric degree
    distribution.
    """
    K = K.tocsr()
    diag = K.diagonal()
    row_nnz = np.diff(K.indptr)
    return {
        "mean_diagonal_mass": float(np.mean(diag)),
        "min_diagonal_mass": float(np.min(diag)),
        "max_diagonal_mass": float(np.max(diag)),
        "mean_row_nnz": float(np.mean(row_nnz)),
        "median_row_nnz": float(np.median(row_nnz)),
        "min_row_nnz": int(np.min(row_nnz)),
        "max_row_nnz": int(np.max(row_nnz)),
        "n_total_nnz": int(K.nnz),
    }


def compute_kernel_diagnostics_for_case(repo_root: Path, ds: str, subj: str,
                                        slc: str, count: int, real: int) -> dict:
    """Build the same RBF kernels used by the Stage C pipeline for
    one case, and report diagnostics for each variant.
    """
    from sam_kem.reconstruction import (
        rbf_kernel, operator_support_mask, normalize_kernel_feature,
    )
    from sam_kem.config import load_contract
    from sam_kem.operator_factory import build_discrete_radon
    from sam_kem.medsam import predictor_factory, build_released_predictor, CheckpointSpec, sha256_file
    from sam_kem.causal_ablation_v3 import (
        build_uniform_kernel,
    )

    # Frozen params
    fpath = repo_root / "results/dev_trainmax_v1/frozen_hyperparameters.csv"
    df = pd.read_csv(fpath)
    df = df[df.method == "SAM-KEM-MedSAM2-Qwen3-full"]
    hp = df[(df.dataset == ds) & (df.count_level == count)].iloc[0]
    sigma = float(hp.sigma); radius = int(hp.radius)
    iterations = int(hp.iterations)

    # Build operator
    cfg, _ = load_contract(repo_root / "code/configs/generated/dev_trainmax_v1_validation_registered.yaml")
    op = build_discrete_radon(repo_root=repo_root, size=128, angles=180)
    measured = np.load(repo_root / f"arrays/dev_trainmax_v1/{ds}/{subj}/{slc}/c{count}/r{real}/sinogram.npy").astype(np.float64)
    measured = measured.astype(np.float64)
    mlem10 = np.load(repo_root / f"arrays/dev_trainmax_v1/{ds}/{subj}/{slc}/c{count}/r{real}/mlem10.npy").astype(np.float32)

    # Predictor cache
    bw_path = repo_root / "checkpoints/dev_trainmax_v1/medsam2_decoder_brainweb.pt"
    he_path = repo_root / "checkpoints/dev_trainmax_v1/medsam2_decoder_hecktor.pt"
    pw = repo_root / "checkpoints/MedSAM2_latest.pt"
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

    support = operator_support_mask(op)

    def _norm(arr):
        return normalize_kernel_feature(arr)

    out = {}
    for v in VARIANTS:
        if v == "UNIFORM_FEATURE_MATCHED":
            K = build_uniform_kernel(np.ones((128, 128), dtype=np.float32),
                                     sigma=sigma, radius=radius, support=support)
            label = "uniform"
        else:
            if v == "MLEM_FEATURE_MATCHED":
                prior = mlem10
            elif v == "Q_ADAPT":
                prior_path = repo_root / f"arrays/dev_trainmax_v1/{ds}/{subj}/{slc}/c{count}/r{real}/sam_prior.npy"
                prior = np.load(prior_path).astype(np.float32)
            else:
                # Use Qwen box prior from cache (zip)
                from sam_kem.causal_ablation_v3 import _load_qwen_prior
                prior = _load_qwen_prior(repo_root / "delivery/dev_trainmax_v1_validation_priors.zip",
                                          ds, subj, slc, count, real).astype(np.float32)
            if v != "MLEM_FEATURE_MATCHED":
                feat = _norm(prior)
            else:
                feat = _norm(prior)
            K = rbf_kernel(feat, sigma=sigma, radius=radius, support=support)
            label = "learned" if v != "MLEM_FEATURE_MATCHED" else "learned_mlem"
        d = kernel_diagnostics(K)
        d["variant"] = v
        d["kernel_class"] = label
        out[v] = d
    return out


# ============================================================================
# 200k/r0 fixed-case archives
# ============================================================================

def export_fixed_case(repo_root: Path, ds: str, subj: str, slc: str,
                      count: int, real: int, per_case_csvs: dict,
                      out_dir: Path) -> dict:
    """Export reference target, prior, 8 reconstructions, 8 error arrays,
    with shapes, dtypes, and SHA-256. Verify against per_case.csv.
    """
    from sam_kem.reconstruction import (
        rbf_kernel, kem, operator_support_mask, normalize_kernel_feature,
    )
    from sam_kem.config import load_contract
    from sam_kem.operator_factory import build_discrete_radon
    from sam_kem.medsam import predictor_factory, build_released_predictor, CheckpointSpec, sha256_file
    from sam_kem.causal_ablation_v3 import (
        load_prior_for_variant, build_uniform_kernel,
    )
    from sam_kem.metrics import nrmse, ssim

    out_dir.mkdir(parents=True, exist_ok=True)
    case_id = f"{ds}/{subj}/{slc}/c{count}/r{real}"

    # Load target
    target = np.load(repo_root / f"arrays/dev_trainmax_v1/{ds}/{subj}/{slc}/target.npy").astype(np.float32)
    np.save(out_dir / "target.npy", target)
    target_sha = sha256_bytes(target)

    # Operator + sinogram
    cfg, _ = load_contract(repo_root / "code/configs/generated/dev_trainmax_v1_validation_registered.yaml")
    op = build_discrete_radon(repo_root=repo_root, size=128, angles=180)
    measured = np.load(repo_root / f"arrays/dev_trainmax_v1/{ds}/{subj}/{slc}/c{count}/r{real}/sinogram.npy").astype(np.float64)
    mlem10 = np.load(repo_root / f"arrays/dev_trainmax_v1/{ds}/{subj}/{slc}/c{count}/r{real}/mlem10.npy").astype(np.float32)

    # Frozen params
    df = pd.read_csv(repo_root / "results/dev_trainmax_v1/frozen_hyperparameters.csv")
    df = df[df.method == "SAM-KEM-MedSAM2-Qwen3-full"]
    hp = df[(df.dataset == ds) & (df.count_level == count)].iloc[0]
    sigma = float(hp.sigma); radius = int(hp.radius); iterations = int(hp.iterations)

    # Predictors
    bw_path = repo_root / "checkpoints/dev_trainmax_v1/medsam2_decoder_brainweb.pt"
    he_path = repo_root / "checkpoints/dev_trainmax_v1/medsam2_decoder_hecktor.pt"
    pw = repo_root / "checkpoints/MedSAM2_latest.pt"
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

    support = operator_support_mask(op)
    gain = np.full_like(measured, 1.0, dtype=np.float64)  # projection scale = 1 in registry (no projection_scale override for fixed export)
    # Use the same per-case projection_scale that the runner used:
    manifest_path = repo_root / str(cfg["outputs"]["validation_manifest"])
    from sam_kem.experiments import load_case_manifest, validate_case_contract
    cases = load_case_manifest(repo_root, manifest_path, split="validation")
    validate_case_contract(cases, contract=None)
    case = next(c for c in cases if (c.dataset == ds and c.subject_id == subj
                                       and c.slice_id == slc
                                       and int(c.realization_id) == real
                                       and int(c.count_level) == count))
    gain = np.full_like(measured, case.projection_scale, dtype=np.float64)

    zip_path = repo_root / "delivery/dev_trainmax_v1_validation_priors.zip"
    qwen_csv = repo_root / "prompts/dev_trainmax_v1/train_validation_box_cache.csv"

    exports = {"case": case_id, "shapes_dtypes": {}, "sha256": {}, "metrics": {}}
    for v in VARIANTS:
        prior_path = out_dir / f"prior_{v}.npy"
        rec_path = out_dir / f"reconstruction_{v}.npy"
        err_path = out_dir / f"abs_error_{v}.npy"

        # Build the prior
        prior, source = load_prior_for_variant(
            v, repo_root=repo_root, zip_path=zip_path,
            ds=ds, subj=subj, slc=slc, count=count, real=real,
            pet_image=mlem10, qwen_box_csv=qwen_csv,
            predictors=predictors,
        )
        np.save(prior_path, prior.astype(np.float32))
        prior_sha = sha256_bytes(prior.astype(np.float32))

        # Build kernel + KEM
        if v == "UNIFORM_FEATURE_MATCHED":
            K = build_uniform_kernel(np.ones((128, 128), dtype=np.float32),
                                     sigma=sigma, radius=radius, support=support)
        else:
            feat = normalize_kernel_feature(prior.astype(np.float32))
            K = rbf_kernel(feat, sigma=sigma, radius=radius, support=support)

        out_dict = kem(measured=measured, operator=op, kernel=K,
                       iterations=iterations, multiplicative=gain)
        rec = out_dict[max(out_dict.keys())]
        rec = np.asarray(rec, dtype=np.float32)
        rec = np.maximum(rec, 0.0)
        np.save(rec_path, rec)
        rec_sha = sha256_bytes(rec)

        # Absolute error
        err = np.abs(rec - target).astype(np.float32)
        np.save(err_path, err)
        err_sha = sha256_bytes(err)

        # Metrics
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
            "target": target_sha,
            f"prior_{v}": prior_sha,
            f"reconstruction_{v}": rec_sha,
            f"abs_error_{v}": err_sha,
        }
        exports["metrics"][v] = {
            "nrmse_recomputed": nm, "nrmse_committed": ref_nrmse,
            "nrmse_match": nrmse_match,
            "ssim_recomputed": sm, "ssim_committed": ref_ssim,
            "ssim_match": ssim_match,
            "source": source,
        }

    # Manifest JSON
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
# Paired subject-level deltas with SSIM (convention: positive favors Q)
# ============================================================================

def paired_subject_deltas_with_ssim(per_case_csvs: dict) -> pd.DataFrame:
    """Per-subject deltas for every non-Q variant x dataset x count.
    Convention:
        delta_nrmse = variant - Q (positive: variant worse than Q)
        delta_ssim  = Q - variant (positive: variant worse than Q)
    Both: positive favors Q.
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
    for v in VARIANTS:
        if v == PRIMARY:
            continue
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
# Driver
# ============================================================================

def main(repo_root: Path, run_dir: Path, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    per_case_csvs = {v: run_dir / v / "per_case.csv" for v in VARIANTS}

    # (1) Loss-ratio cells + summary
    print("=" * 60)
    print("(1) Loss ratio formula")
    print("=" * 60)
    loss_cells = compute_loss_ratio_cells(per_case_csvs)
    loss_cells.to_csv(out_dir / "loss_ratio_cells.csv", index=False)
    loss_summary = summarize_loss_ratios(loss_cells)
    loss_summary.to_csv(out_dir / "loss_ratio_summary.csv", index=False)
    print(loss_summary.to_string(index=False))

    # (2) Paired HE bootstrap
    print()
    print("=" * 60)
    print("(2) Paired HE subject-cluster bootstrap (NRMSE + SSIM)")
    print("=" * 60)
    boot_records = paired_he_subject_bootstrap(per_case_csvs)
    with open(out_dir / "hecktor_paired_bootstrap.json", "w", encoding="utf-8") as f:
        json.dump({
            "n_bootstrap": N_BOOTSTRAP,
            "seed": MBOOTSTRAP_SEED,
            "seed_label": "post-run deterministic analysis choice, not a preregistered seed",
            "convention": "delta_nrmse = variant - Q; delta_ssim = Q - variant (positive favors Q)",
            "results": boot_records,
        }, f, indent=2)
    for r in boot_records:
        print(f"  {r['variant']:25s} c{r['count_level']:>7d}: "
              f"delta_nrmse={r['mean_delta_nrmse']:+.4f} "
              f"CI95=[{r['delta_nrmse_ci95_low']:+.4f}, {r['delta_nrmse_ci95_high']:+.4f}] "
              f"delta_ssim={r['mean_delta_ssim']:+.4f} "
              f"CI95=[{r['delta_ssim_ci95_low']:+.4f}, {r['delta_ssim_ci95_high']:+.4f}]")

    # (3) Real kernel diagnostics
    print()
    print("=" * 60)
    print("(3) Real kernel diagnostics (rebuilt kernels on the 2 fixed cases)")
    print("=" * 60)
    kd_bw = compute_kernel_diagnostics_for_case(repo_root, "brainweb", "subject04", "z038", 200000, 0)
    kd_he = compute_kernel_diagnostics_for_case(repo_root, "hecktor", "10", "z072", 200000, 0)

    # Learned-vs-uniform ratios for both cases
    def _ratios(kd):
        ratios = {}
        uni = kd.get("UNIFORM_FEATURE_MATCHED", {})
        for v in VARIANTS:
            if v == "UNIFORM_FEATURE_MATCHED":
                continue
            d = kd.get(v, {})
            if "mean_row_nnz" in uni and "mean_row_nnz" in d and uni["mean_row_nnz"] > 0:
                ratios[v] = {
                    "row_nnz_ratio_vs_uniform": float(d["mean_row_nnz"] / uni["mean_row_nnz"]),
                    "diag_mass_ratio_vs_uniform": float(d["mean_diagonal_mass"] / uni["mean_diagonal_mass"])
                        if uni["mean_diagonal_mass"] > 0 else None,
                }
        return ratios

    with open(out_dir / "kernel_diagnostics.json", "w", encoding="utf-8") as f:
        json.dump({
            "brainweb_subject04_z038_c200000_r0": kd_bw,
            "hecktor_10_z072_c200000_r0": kd_he,
            "learned_vs_uniform_ratios_brainweb": _ratios(kd_bw),
            "learned_vs_uniform_ratios_hecktor": _ratios(kd_he),
            "notes": (
                "Diagnostics computed from the actual sparse RBF kernels rebuilt on the two "
                "200k/r0 fixed cases. The unified_kernel has a smaller mean_row_nnz (radius=2, "
                "Manhattan distance <= 2) and a larger diagonal mass because every neighbor "
                "contributes weight 1 before normalization."
            ),
        }, f, indent=2)
    print("  brainweb kd: written")
    print("  hecktor kd: written")

    # (4) 200k/r0 fixed-case archives
    print()
    print("=" * 60)
    print("(4) 200k/r0 fixed-case archives")
    print("=" * 60)
    fixed_bw = export_fixed_case(repo_root, "brainweb", "subject04", "z038",
                                  200000, 0, per_case_csvs,
                                  out_dir / "fixed_case_brainweb_subject04_z038_200k_r0")
    print(f"  brainweb: {fixed_bw['all_metrics_match']}")
    fixed_he = export_fixed_case(repo_root, "hecktor", "10", "z072",
                                  200000, 0, per_case_csvs,
                                  out_dir / "fixed_case_hecktor_10_z072_200k_r0")
    print(f"  hecktor: {fixed_he['all_metrics_match']}")

    # (5) Paired subject-level deltas with SSIM
    print()
    print("=" * 60)
    print("(5) Paired subject-level deltas (NRMSE + SSIM, convention)")
    print("=" * 60)
    deltas = paired_subject_deltas_with_ssim(per_case_csvs)
    deltas.to_csv(out_dir / "paired_subject_deltas.csv", index=False)
    summary = summarize_paired_deltas(deltas)
    summary.to_csv(out_dir / "paired_subject_deltas_summary.csv", index=False)
    print(f"  per-subject rows: {len(deltas)}")
    print(f"  summary rows: {len(summary)}")
    print(f"  BW subjects per (variant,count) cell: max {summary[summary.dataset=='brainweb'].n_subjects.max()}")

    # Master summary
    summary_all = {
        "artifact_kind": "validation_only_causal_ablations_v3_stage_e",
        "schema_version": 1,
        "loss_ratio_formula": "(NRMSE_v - NRMSE_Q) / (NRMSE_MLEM - NRMSE_Q)",
        "loss_ratio_cells": "loss_ratio_cells.csv",
        "loss_ratio_summary": "loss_ratio_summary.csv",
        "hecktor_paired_bootstrap": "hecktor_paired_bootstrap.json",
        "kernel_diagnostics": "kernel_diagnostics.json",
        "fixed_case_brainweb": "fixed_case_brainweb_subject04_z038_200k_r0/",
        "fixed_case_hecktor": "fixed_case_hecktor_10_z072_200k_r0/",
        "paired_subject_deltas": "paired_subject_deltas.csv",
        "paired_subject_deltas_summary": "paired_subject_deltas_summary.csv",
        "mbootstrap_seed": MBOOTSTRAP_SEED,
        "n_bootstrap": N_BOOTSTRAP,
        "all_metrics_match_brainweb": fixed_bw["all_metrics_match"],
        "all_metrics_match_hecktor": fixed_he["all_metrics_match"],
    }
    with open(out_dir / "stage_e_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary_all, f, indent=2)
    return summary_all


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 3:
        print("Usage: python -m sam_kem.stage_e_analysis REPO_ROOT OUT_DIR")
        sys.exit(1)
    main(Path(sys.argv[1]).resolve(), Path(sys.argv[1]).resolve() / "results/dev_trainmax_v1/causal_ablations_corrected_v3/full_run",
          Path(sys.argv[2]).resolve())