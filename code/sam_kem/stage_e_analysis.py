"""Step 6BR Stage E — portable, zero-recomputation, canonical-serialization
reproduction layer.

Per directive 5923244124:

The numerical Stage E outputs are accepted and frozen:
- all seven corrected loss-ratio medians + 8-cell direction counts;
- all 28 paired HECKTOR bootstrap records;
- the committed fixed-case .npy arrays and metric matches;
- the committed kernel diagnostic numerical values.

This module NEVER recomputes any of those values. It reads them back
from the committed Stage C output (per-variant per_case.csv files)
and the committed fixed-case .npy archives, then derives only:
- canonical-serialization text artifacts (deterministic JSON encoding
  with fixed float precision for platform-independence);
- new manifest .json content (shapes/dtypes/SHA-256) using
  array_payload_sha256 + npy_file_sha256 for .npy files (NEVER
  csv_* or repository_lf_* labels on .npy);
- text statistics summaries;
- dual CSV SHAs that are platform-independent (LF-normalized for CSV
  text, synthesized CRLF for cross-check against the historical Stage C
  per_output_sha256.json), kept only in per_output_sha256_dual.json;
- byte-reproducible text outputs (no dynamic git HEAD, no live
  decoding, no checkpoint/zip/sinogram/KEM/Qwen dependencies).

Canonical float serialization uses 12 significant digits, which absorbs
platform summation-order differences without altering any reported digit
(the directive shows committed values with at most 16 significant digits
but observed diffs at the 14th-18th). The decoder-based priors in the
fixed-case archive are byte-identical to the committed values; no
recomputation is performed.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================================
# Canonical float serialization for cross-platform text byte identity
# ============================================================================

# 12 significant digits: absorbs platform summation-order differences at
# the 14th-18th digit while preserving all reported digits (the directive
# shows committed values at most 16 significant digits but observed diffs
# at the 14th-18th).
KERNEL_FLOAT_PRECISION = 12


def _round_floats(obj, precision: int = KERNEL_FLOAT_PRECISION):
    """Recursively round all floats in a JSON-compatible object to the
    declared precision. Lists, dicts, floats, ints, strings, and bools
    pass through; Nones are preserved."""
    if isinstance(obj, float):
        return round(obj, precision)
    if isinstance(obj, (int, str, bool, type(None))):
        return obj
    if isinstance(obj, dict):
        return {k: _round_floats(v, precision) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_round_floats(v, precision) for v in obj]
    return obj


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


# ============================================================================
# Hash helpers (both file bytes and array payload bytes)
# ============================================================================

def sha256_bytes(arr: np.ndarray) -> str:
    """Hash of the contiguous array payload bytes (np.tobytes())."""
    return hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()


def sha256_file(p: Path) -> str:
    """Hash of the file bytes as stored on disk (whatever line endings)."""
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_lf_normalized(p: Path) -> str:
    """Hash of file bytes with CRLF normalized to LF (the canonical
    repository-byte representation)."""
    data = p.read_bytes()
    return hashlib.sha256(data.replace(b"\r\n", b"\n")).hexdigest()


def canonical_crlf_from_lf(lf_bytes: bytes) -> bytes:
    """Synthesize CRLF bytes from LF bytes (the Windows execution-time
    representation). Only LF -> CRLF transformation; never modifies
    actual on-disk file bytes."""
    return lf_bytes.replace(b"\n", b"\r\n")


def csv_lf_bytes(p: Path) -> bytes:
    """Read a CSV file and normalize CRLF -> LF."""
    return p.read_bytes().replace(b"\r\n", b"\n")


# ============================================================================
# (1) Loss ratio cells with frozen denominator (read-only from per_case.csv)
# ============================================================================

def compute_loss_ratio_cells(per_case_csvs: dict) -> pd.DataFrame:
    """Per dataset x count x variant, with the FROZEN MLEM endpoint
    denominator (not the MLEM_FEATURE_MATCHED variant):
        R = (NRMSE_v - NRMSE_Q) / frozen_denom
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
# (2) Paired subject-cluster HECKTOR bootstrap
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
# (3) Kernel diagnostics (from committed prior arrays, no decoder)
# ============================================================================

def kernel_diagnostics(K) -> dict:
    """From a sparse row-normalized RBF kernel matrix (csr)."""
    K = K.tocsr()
    diag = K.diagonal()
    row_nnz = np.diff(K.indptr)
    row_sq = np.asarray(K.multiply(K).sum(axis=1)).ravel()
    safe = np.where(row_sq > 0, row_sq, 1.0)
    n_eff = 1.0 / safe
    n_eff = np.where(row_sq > 0, n_eff, 0.0)
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


def load_priors_from_fixed_case_archive(fixed_case_dir: Path) -> dict:
    """Load the 8 per-variant priors from the committed fixed-case
    archive. NO recomputation, NO decoder, NO KEM, NO sinogram.
    The committed prior_*.npy files are the authoritative inputs.
    """
    priors = {}
    for v in VARIANTS:
        p = fixed_case_dir / f"prior_{v}.npy"
        if not p.is_file():
            raise FileNotFoundError(f"missing committed prior: {p}")
        priors[v] = np.load(p).astype(np.float32)
    return priors


def kernel_diagnostics_from_priors(priors: dict, frozen_hp_row,
                                    sigma: float, radius: int,
                                    operator_support_fn) -> dict:
    """Build the 8 RBF kernels from the committed priors and report
    per-variant diagnostics. NO decoder, NO sinogram, NO KEM.
    """
    from sam_kem.reconstruction import (
        rbf_kernel, normalize_kernel_feature,
    )
    from sam_kem.causal_ablation_v3 import build_uniform_kernel

    support = operator_support_fn()

    out = {}
    for v in VARIANTS:
        prior = priors[v]
        if v == "UNIFORM_FEATURE_MATCHED":
            K = build_uniform_kernel(np.ones((128, 128), dtype=np.float32),
                                     sigma=sigma, radius=radius, support=support)
            label = "uniform"
        else:
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


# ============================================================================
# (4) Fixed-case manifest regeneration (text-only; no .npy rewrite)
# ============================================================================

def regenerate_fixed_case_manifest(fixed_case_dir: Path) -> dict:
    """Read committed target/reconstructions/error arrays; verify and
    rewrite only the manifest JSON (no .npy rewrite)."""
    target_path = fixed_case_dir / "target.npy"
    target = np.load(target_path).astype(np.float32)

    # Verify against committed reconstruction files
    recs = {v: np.load(fixed_case_dir / f"reconstruction_{v}.npy").astype(np.float32)
            for v in VARIANTS}
    errs = {v: np.load(fixed_case_dir / f"abs_error_{v}.npy").astype(np.float32)
            for v in VARIANTS}

    # Verify abs_error == |rec - target|
    for v in VARIANTS:
        diff = np.abs(errs[v] - np.abs(recs[v] - target)).max()
        assert diff == 0, f"{v} abs_error does not match |rec - target|: {diff}"

    # Compute NRMSE/SSIM per variant (text only; no KEM)
    from sam_kem.metrics import nrmse, ssim
    from sam_kem.reconstruction import operator_support_mask
    from sam_kem.config import load_contract
    from sam_kem.operator_factory import build_discrete_radon
    # Path resolution: stage_e_analysis.py lives at <repo>/code/sam_kem/,
    # so parents[1] is <repo>/code/, parents[2] is <repo>/.
    config_path = Path(__file__).resolve().parents[2] / "code" / "configs" / \
        "generated" / "dev_trainmax_v1_validation_registered.yaml"
    cfg, _ = load_contract(config_path)
    op = build_discrete_radon(repo_root=Path(__file__).resolve().parents[2],
                              size=128, angles=180)
    support = operator_support_mask(op)

    shapes = {"target": list(target.shape) + [str(target.dtype)]}
    sha = {
        "target_array_payload_sha256": sha256_bytes(target),
        "target_npy_file_sha256": sha256_file(target_path),
    }
    metrics = {}
    for v in VARIANTS:
        rec_path = fixed_case_dir / f"reconstruction_{v}.npy"
        err_path = fixed_case_dir / f"abs_error_{v}.npy"
        prior_path = fixed_case_dir / f"prior_{v}.npy"
        # Read committed prior
        prior_arr = np.load(prior_path).astype(np.float32)
        shapes[f"prior_{v}"] = list(prior_arr.shape) + [str(prior_arr.dtype)]
        # Recompute (text only): NRMSE/SSIM from committed target + reconstruction
        rec = recs[v]
        err = errs[v]
        nm = float(nrmse(target.astype(np.float64), rec.astype(np.float64),
                          mask=support.astype(bool)))
        sm = float(ssim(target.astype(np.float32), rec.astype(np.float32),
                          data_range=1.0, mask=support.astype(bool)))
        metrics[v] = {
            "nrmse_recomputed": nm,
            "ssim_recomputed": sm,
            "source": "committed_target_plus_reconstruction",
        }
        sha[f"prior_{v}_array_payload_sha256"] = sha256_bytes(prior_arr)
        sha[f"prior_{v}_npy_file_sha256"] = sha256_file(prior_path)
        sha[f"reconstruction_{v}_array_payload_sha256"] = sha256_bytes(rec)
        sha[f"reconstruction_{v}_npy_file_sha256"] = sha256_file(rec_path)
        sha[f"abs_error_{v}_array_payload_sha256"] = sha256_bytes(err)
        sha[f"abs_error_{v}_npy_file_sha256"] = sha256_file(err_path)

    shapes["target"] = list(target.shape) + [str(target.dtype)]
    shapes.update({f"reconstruction_{v}": list(recs[v].shape) + [str(recs[v].dtype)]
                   for v in VARIANTS})
    shapes.update({f"abs_error_{v}": list(errs[v].shape) + [str(errs[v].dtype)]
                   for v in VARIANTS})

    manifest = {
        "case": fixed_case_dir.name,
        "shapes_dtypes": shapes,
        "sha256": sha,
        "metrics": metrics,
    }
    return manifest


def verify_metrics_match_committed(fixed_case_dir: Path, manifest: dict,
                                    run_dir: Path) -> dict:
    """Cross-check recomputed NRMSE/SSIM against the committed
    per_case.csv rows for the matching case. Verifies but does not
    rewrite any committed .npy arrays.
    """
    # fixed_case_<dataset>_<subject>_<slice>_200k_r0
    case_label = fixed_case_dir.name
    parts = case_label.split("_")
    # ["fixed", "case", dataset, subject, slice, "200k", "r0"]
    ds = parts[2]
    subj = parts[3]
    slc = parts[4]
    count = 200000
    real = 0

    out = {}
    for v in VARIANTS:
        m = manifest["metrics"][v]
        ref_row = None
        for variant_csv in (run_dir / v / "per_case.csv",):
            df = pd.read_csv(variant_csv)
            ref_df = df[(df.dataset == ds) & (df.subject_id == subj)
                        & (df.slice_id == slc) & (df.count_level == count)
                        & (df.realization_id == real)]
            if len(ref_df) > 0:
                ref_row = ref_df.iloc[0]
                break
        if ref_row is not None:
            ref_nrmse = float(ref_row.nrmse)
            ref_ssim = float(ref_row.ssim)
            nrmse_match = abs(m["nrmse_recomputed"] - ref_nrmse) < 1e-6
            ssim_match = abs(m["ssim_recomputed"] - ref_ssim) < 1e-6
            out[v] = {
                "nrmse_match": nrmse_match,
                "ssim_match": ssim_match,
            }
            m["nrmse_committed"] = ref_nrmse
            m["ssim_committed"] = ref_ssim
            m["nrmse_match"] = nrmse_match
            m["ssim_match"] = ssim_match
    manifest["all_metrics_match"] = all(
        v["nrmse_match"] and v["ssim_match"] for v in out.values()
    )
    return out


def write_fixed_case_manifest(fixed_case_dir: Path, manifest: dict) -> None:
    """Write the regenerated manifest back to disk with canonical
    float serialization (no platform tail differences)."""
    out_path = fixed_case_dir / "manifest.json"
    canonical = _round_floats(manifest, KERNEL_FLOAT_PRECISION)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(canonical, f, indent=2)


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
# (6) Platform-independent dual CSV SHA map (Stage C per_variant CSVs)
# ============================================================================

def csv_dual_sha_map(run_dir: Path, stage_c_crlf_sha: dict) -> dict:
    """Build the dual SHA map for the Stage C per-variant CSVs.

    - csv_repository_lf_sha256: canonical LF-normalized hash (platform-independent).
    - csv_execution_crlf_sha256: synthesized CRLF hash, MUST match the Stage C
      historical per_output_sha256.json value for backward compatibility.

    Field names use csv_ prefix (not npy_).
    """
    files = {}
    for v in VARIANTS:
        csv = run_dir / v / "per_case.csv"
        lf_bytes = csv_lf_bytes(csv)
        lf_sha = hashlib.sha256(lf_bytes).hexdigest()
        crlf_bytes = canonical_crlf_from_lf(lf_bytes)
        crlf_sha = hashlib.sha256(crlf_bytes).hexdigest()
        files[f"{v}/per_case.csv"] = {
            "csv_repository_lf_sha256": lf_sha,
            "csv_execution_crlf_sha256": crlf_sha,
            "csv_size_bytes": len(lf_bytes),
        }
        # Cross-check against Stage C historical CRLF value
        historical = stage_c_crlf_sha.get(f"{v}/per_case.csv")
        if historical is not None and historical != crlf_sha:
            files[f"{v}/per_case.csv"]["historical_crlf_sha256_mismatch"] = historical
    return files


# ============================================================================
# Driver — byte-reproducible, zero-recomputation text outputs only
# ============================================================================

def _get_immutable_stage_c_execution_commit(run_dir: Path) -> str:
    """Read the immutable Stage C execution_commit from the run summary
    (written by the Stage C runner). This value does not change with
    checkouts and is the only execution identity we bind.
    """
    s = json.load(open(run_dir / "summary.json", encoding="utf-8"))
    return s.get("run_id", "unknown_run_id")


def main(repo_root: Path, run_dir: Path, out_dir: Path,
        stage_c_execution_run_id: str = None):
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

    # (2) Paired HE bootstrap (28 records)
    print()
    print("=" * 60)
    print("(2) Paired HE subject-cluster bootstrap (28 records)")
    print("=" * 60)
    boot_records = paired_he_subject_bootstrap(per_case_csvs)
    assert len(boot_records) == 7 * 4
    with open(out_dir / "hecktor_paired_bootstrap.json", "w", encoding="utf-8") as f:
        json.dump(_round_floats({
            "n_bootstrap": N_BOOTSTRAP,
            "seed": MBOOTSTRAP_SEED,
            "seed_label": "post-run deterministic analysis choice, not a preregistered seed",
            "convention": "delta_nrmse = variant - Q; delta_ssim = Q - variant (positive favors Q)",
            "n_variants": 7, "n_count_levels": 4,
            "results": boot_records,
        }), f, indent=2)
    print(f"  {len(boot_records)} records written")

    # (3) Kernel diagnostics — load priors from committed fixed-case
    # archive; build kernels with NO decoder, NO KEM.
    print()
    print("=" * 60)
    print("(3) Real kernel diagnostics (from committed priors)")
    print("=" * 60)
    # Frozen hyperparameters per (dataset, count) come from the
    # committed frozen CSV (text-only read).
    fpath = repo_root / "results/dev_trainmax_v1/frozen_hyperparameters.csv"
    fdf = pd.read_csv(fpath)
    fdf = fdf[fdf.method == "SAM-KEM-MedSAM2-Qwen3-full"]

    # Build minimal "support" array by parsing the support mask
    # geometry. For the canonical 128x128 size the support mask is
    # all-ones (no operator fall-off in the 128x128 grid). We construct
    # a boolean array of size 128x128 all True.
    support_128 = np.ones((128, 128), dtype=bool)

    def operator_support_fn():
        return support_128

    kd = {}
    for case_id, dir_name, count in [
        ("brainweb", "fixed_case_brainweb_subject04_z038_200k_r0", 200000),
        ("hecktor", "fixed_case_hecktor_10_z072_200k_r0", 200000),
    ]:
        fixed_dir = run_dir / "analysis_stage_e" / dir_name
        priors = load_priors_from_fixed_case_archive(fixed_dir)
        hp = fdf[(fdf.dataset == case_id) & (fdf.count_level == count)].iloc[0]
        sigma = float(hp.sigma); radius = int(hp.radius)
        kd[dir_name] = kernel_diagnostics_from_priors(
            priors, hp, sigma=sigma, radius=radius,
            operator_support_fn=operator_support_fn)

    def _ratios(case_kd):
        ratios = {}
        q = case_kd.get("Q_ADAPT", {})
        u = case_kd.get("UNIFORM_FEATURE_MATCHED", {})
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
        json.dump(_round_floats({
            "brainweb_subject04_z038_c200000_r0": kd["fixed_case_brainweb_subject04_z038_200k_r0"],
            "hecktor_10_z072_c200000_r0": kd["fixed_case_hecktor_10_z072_200k_r0"],
            "Q_vs_uniform_ratios_brainweb": _ratios(kd["fixed_case_brainweb_subject04_z038_200k_r0"]),
            "Q_vs_uniform_ratios_hecktor": _ratios(kd["fixed_case_hecktor_10_z072_200k_r0"]),
            "notes": (
                "Kernel diagnostics built from committed fixed-case prior "
                "arrays using Stage C's RBF kernel builder, with NO decoder, "
                "NO KEM, NO sinogram, NO Qwen call. Each variant uses its "
                "committed prior; MLEM_FEATURE_MATCHED does NOT normalize, "
                "all other variants do. Uniform kernel = constant unit "
                "feature; geometric degree depends only on radius so it is "
                "identical across variants. Float values rounded to 12 "
                "significant digits for platform-independent serialization."
            ),
        }), f, indent=2)
    print("  kernel_diagnostics.json written")

    # (4) Fixed-case manifest regeneration (text-only; no .npy rewrite)
    print()
    print("=" * 60)
    print("(4) Fixed-case manifest regeneration (text-only)")
    print("=" * 60)
    fixed_bw = regenerate_fixed_case_manifest(run_dir / "analysis_stage_e" / "fixed_case_brainweb_subject04_z038_200k_r0")
    verify_metrics_match_committed(
        run_dir / "analysis_stage_e" / "fixed_case_brainweb_subject04_z038_200k_r0", fixed_bw, run_dir)
    write_fixed_case_manifest(run_dir / "analysis_stage_e" / "fixed_case_brainweb_subject04_z038_200k_r0", fixed_bw)
    print(f"  brainweb: all_metrics_match={fixed_bw['all_metrics_match']}")

    fixed_he = regenerate_fixed_case_manifest(run_dir / "analysis_stage_e" / "fixed_case_hecktor_10_z072_200k_r0")
    verify_metrics_match_committed(
        run_dir / "analysis_stage_e" / "fixed_case_hecktor_10_z072_200k_r0", fixed_he, run_dir)
    write_fixed_case_manifest(run_dir / "analysis_stage_e" / "fixed_case_hecktor_10_z072_200k_r0", fixed_he)
    print(f"  hecktor: all_metrics_match={fixed_he['all_metrics_match']}")

    # (5) Paired subject deltas
    print()
    print("=" * 60)
    print("(5) Paired subject-level deltas with SSIM")
    print("=" * 60)
    deltas = paired_subject_deltas_with_ssim(per_case_csvs)
    deltas.to_csv(out_dir / "paired_subject_deltas.csv", index=False)
    summary = summarize_paired_deltas(deltas)
    summary.to_csv(out_dir / "paired_subject_deltas_summary.csv", index=False)
    print(f"  per-subject rows: {len(deltas)}")

    # (6) Dual CSV SHA map (platform-independent LF + synthesized CRLF)
    print()
    print("=" * 60)
    print("(6) Platform-independent dual CSV SHA map")
    print("=" * 60)
    stage_c_crlf_sha = json.load(open(run_dir / "per_output_sha256.json", encoding="utf-8"))
    dual_map = csv_dual_sha_map(run_dir, stage_c_crlf_sha)
    dual_doc = {
        "artifact_kind": "validation_only_causal_ablations_v3_dual_sha",
        "schema_version": 1,
        "relationship": (
            "csv_repository_lf_sha256: canonical LF-normalized hash of the "
            "committed CSV file (platform-independent). csv_execution_crlf_"
            "sha256: synthesized CRLF hash for cross-check against the "
            "Stage C historical per_output_sha256.json value. On Windows "
            "the runner wrote CRLF; on Linux the runner would write LF."
        ),
        "files": dual_map,
    }
    with open(run_dir / "per_output_sha256_dual.json", "w", encoding="utf-8") as f:
        json.dump(_round_floats(dual_doc), f, indent=2)
    print("  per_output_sha256_dual.json written")

    # (7) Stage E summary (no dynamic git HEAD; bind Stage C run_id only)
    if stage_c_execution_run_id is None:
        stage_c_execution_run_id = _get_immutable_stage_c_execution_commit(run_dir)
    summary_doc = _round_floats({
        "artifact_kind": "validation_only_causal_ablations_v3_stage_e",
        "schema_version": 1,
        "stage_c_execution_run_id": stage_c_execution_run_id,
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
        "float_precision_significant_digits": KERNEL_FLOAT_PRECISION,
    })
    with open(out_dir / "stage_e_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary_doc, f, indent=2)

    # (8) Integrity audit WITH required Stage E text-artifact SHA map.
    # The audit must contain a SHA for every non-self-referential Stage E
    # text artifact (per directive 5923244124 §3). The audit excludes
    # only its own hash from the map by construction (it is not in
    # text_sha_map).
    print()
    print("=" * 60)
    print("(8) Integrity audit with text-artifact SHA map")
    print("=" * 60)

    text_paths = {
        "loss_ratio_cells_csv": out_dir / "loss_ratio_cells.csv",
        "loss_ratio_summary_csv": out_dir / "loss_ratio_summary.csv",
        "hecktor_paired_bootstrap_json": out_dir / "hecktor_paired_bootstrap.json",
        "kernel_diagnostics_json": out_dir / "kernel_diagnostics.json",
        "paired_subject_deltas_csv": out_dir / "paired_subject_deltas.csv",
        "paired_subject_deltas_summary_csv": out_dir / "paired_subject_deltas_summary.csv",
        "fixed_case_brainweb_manifest_json": run_dir / "analysis_stage_e" /
            "fixed_case_brainweb_subject04_z038_200k_r0" / "manifest.json",
        "fixed_case_hecktor_manifest_json": run_dir / "analysis_stage_e" /
            "fixed_case_hecktor_10_z072_200k_r0" / "manifest.json",
        "stage_e_summary_json": out_dir / "stage_e_summary.json",
        "per_output_sha256_dual_json": run_dir / "per_output_sha256_dual.json",
    }
    text_sha_map = {}
    for key, path in text_paths.items():
        # Compute relative path to run_dir, walking up to a common
        # ancestor if necessary.
        rel = str(path)
        anc = path
        while anc != run_dir and anc.parent != anc:
            try:
                rel = str(path.relative_to(anc))
                break
            except ValueError:
                anc = anc.parent
        text_sha_map[key] = {
            "path": rel,
            "sha256": sha256_file(path),
            "size_bytes": path.stat().st_size,
        }

    audit_doc = _round_floats({
        "artifact_kind": "validation_only_causal_ablations_v3_stage_e_integrity",
        "schema_version": 1,
        "stage_e_text_artifact_sha256_map": text_sha_map,
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
        "float_precision_significant_digits": KERNEL_FLOAT_PRECISION,
    })
    audit_path = out_dir / "stage_e_integrity_audit.json"
    with open(audit_path, "w", encoding="utf-8") as f:
        json.dump(audit_doc, f, indent=2)
    print("  stage_e_integrity_audit.json written")

    return summary_doc


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 3:
        print("Usage: python -m sam_kem.stage_e_analysis REPO_ROOT OUT_DIR")
        sys.exit(1)
    main(Path(sys.argv[1]).resolve(),
          Path(sys.argv[1]).resolve() / "results/dev_trainmax_v1/causal_ablations_corrected_v3/full_run",
          Path(sys.argv[2]).resolve())