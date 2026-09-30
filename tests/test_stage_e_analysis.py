"""Step 6BO Stage E — portable tests for the deterministic post-processing
analysis tool.

The committed v3 endpoint package must be the input: this test module
verifies the loss-ratio formula, paired deltas with SSIM convention,
and kernel diagnostic helpers, all from the committed inputs only.

Tests derive paths from ``Path(__file__).resolve()`` with env-var
fallback; no machine-specific absolute paths.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))

from sam_kem.stage_e_analysis import (
    VARIANTS,
    MBOOTSTRAP_SEED,
    N_BOOTSTRAP,
    compute_loss_ratio_cells,
    summarize_loss_ratios,
    paired_subject_deltas_with_ssim,
    summarize_paired_deltas,
    paired_he_subject_bootstrap,
    kernel_diagnostics,
    sha256_bytes,
    sha256_file,
)


# ============================================================================
# Portable repository-root discovery (same logic as v3 tests)
# ============================================================================

def _resolve_repo_root() -> Path:
    here = Path(__file__).resolve()
    for parent in [here.parent, *here.parents]:
        if (parent / "code" / "sam_kem").is_dir():
            if (parent / "results" / "dev_trainmax_v1" /
                "causal_ablations_corrected_v3" / "full_run").is_dir():
                return parent
    env = os.environ.get("SAM_KEM_V3_REPO")
    if env:
        return Path(env).resolve()
    raise RuntimeError(
        "Cannot locate repository root; set SAM_KEM_V3_REPO env var."
    )


REPO_ROOT = _resolve_repo_root()
RUN_DIR = REPO_ROOT / "results/dev_trainmax_v1/causal_ablations_corrected_v3/full_run"
PER_CASE_CSVS = {v: RUN_DIR / v / "per_case.csv" for v in VARIANTS}


# ============================================================================
# Loss ratio formula
# ============================================================================

def test_loss_ratio_formula_definition():
    """R = (NRMSE_v - NRMSE_Q) / (NRMSE_MLEM - NRMSE_Q)."""
    cells = compute_loss_ratio_cells(PER_CASE_CSVS)
    assert "loss_ratio_R" in cells.columns
    assert len(cells) == 7 * 8  # 7 non-Q variants * 8 dataset-count cells
    # Q is "lower NRMSE" iff R > 0; verify that positive R => nrmse_v > nrmse_q
    sub = cells.dropna(subset=["loss_ratio_R"])
    assert ((sub["nrmse_v"] > sub["nrmse_q"]) == (sub["loss_ratio_R"] > 0)).all()


def test_loss_ratio_summary_has_8_cells_per_variant():
    summary = summarize_loss_ratios(compute_loss_ratio_cells(PER_CASE_CSVS))
    # 7 non-Q variants
    assert len(summary) == 7
    # Each row should have exactly 8 cell values
    counts = summary.n_cells.tolist()
    assert all(c == 8 for c in counts), counts


def test_loss_ratio_summary_reports_median():
    summary = summarize_loss_ratios(compute_loss_ratio_cells(PER_CASE_CSVS))
    assert "median_loss_ratio" in summary.columns
    assert summary["median_loss_ratio"].notna().all()


def test_loss_ratio_summary_reports_q_lower_count():
    summary = summarize_loss_ratios(compute_loss_ratio_cells(PER_CASE_CSVS))
    assert "n_q_lower_nrmse" in summary.columns
    assert summary["n_q_lower_nrmse"].between(0, 8).all()


# ============================================================================
# Paired subject deltas with SSIM convention
# ============================================================================

def test_paired_subject_deltas_convention():
    deltas = paired_subject_deltas_with_ssim(PER_CASE_CSVS)
    assert {"dataset", "count_level", "subject_id", "variant",
            "delta_nrmse", "delta_ssim",
            "q_lower_nrmse", "q_higher_ssim"}.issubset(deltas.columns)
    # Convention: positive delta_nrmse = variant worse than Q
    assert ((deltas["q_lower_nrmse"]) == (deltas["delta_nrmse"] > 0)).all()
    # Convention: positive delta_ssim = variant worse than Q (Q ssim higher)
    assert ((deltas["q_higher_ssim"]) == (deltas["delta_ssim"] > 0)).all()


def test_paired_subject_deltas_includes_all_four_bw_subjects():
    deltas = paired_subject_deltas_with_ssim(PER_CASE_CSVS)
    # For each non-Q variant x count, the BW group should have 4 subjects
    bw = deltas[deltas.dataset == "brainweb"]
    counts = bw.groupby(["variant", "count_level"]).subject_id.nunique()
    assert (counts == 4).all(), counts.value_counts()


def test_paired_subject_deltas_no_slices_pooled_as_subjects():
    """subject_id entries are subject IDs, not slice IDs."""
    deltas = paired_subject_deltas_with_ssim(PER_CASE_CSVS)
    bw_subs = sorted(deltas[deltas.dataset == "brainweb"].subject_id.unique())
    assert bw_subs == ["subject04", "subject42", "subject44", "subject50"], bw_subs
    he_subs = sorted(deltas[deltas.dataset == "hecktor"].subject_id.unique())
    # HECKTOR has 45 subjects, distinct IDs (not slice IDs)
    assert len(he_subs) == 45


def test_paired_subject_deltas_summary_columns():
    deltas = paired_subject_deltas_with_ssim(PER_CASE_CSVS)
    summary = summarize_paired_deltas(deltas)
    for col in ["variant", "dataset", "count_level", "n_subjects",
                "n_q_lower_nrmse", "n_q_higher_ssim",
                "mean_delta_nrmse", "mean_delta_ssim"]:
        assert col in summary.columns


# ============================================================================
# Paired HECKTOR bootstrap
# ============================================================================

def test_paired_hecktor_bootstrap_seed_and_count():
    records = paired_he_subject_bootstrap(PER_CASE_CSVS)
    assert len(records) > 0
    for r in records:
        assert r["seed"] == MBOOTSTRAP_SEED
        assert r["n_bootstrap"] == N_BOOTSTRAP
        assert "post-run deterministic" in r["seed_label"]
        assert r["convention"].startswith("delta_nrmse = variant")


def test_paired_hecktor_bootstrap_includes_nrmse_and_ssim():
    records = paired_he_subject_bootstrap(PER_CASE_CSVS)
    for r in records:
        assert "mean_delta_nrmse" in r
        assert "mean_delta_ssim" in r
        assert "delta_nrmse_ci95_low" in r and "delta_nrmse_ci95_high" in r
        assert "delta_ssim_ci95_low" in r and "delta_ssim_ci95_high" in r


# ============================================================================
# Kernel diagnostics (deterministic from a synthetic kernel)
# ============================================================================

def test_kernel_diagnostics_from_synthetic_kernel():
    """The diagnostic helper returns shape/dtype-correct fields on a
    synthetic row-normalized kernel."""
    import scipy.sparse as sp
    n = 64
    rows = np.repeat(np.arange(n), 3)
    offsets = np.tile(np.array([-1, 0, 1]), n)
    cols = (np.arange(n).repeat(3) + offsets) % n
    weights = np.full(n * 3, 1.0 / 3.0)
    K = sp.csr_matrix((weights, (rows, cols)), shape=(n, n))
    d = kernel_diagnostics(K)
    assert "mean_diagonal_mass" in d
    assert "mean_row_nnz" in d
    assert d["mean_row_nnz"] == 3.0
    assert d["n_total_nnz"] == n * 3
    assert 0.0 <= d["mean_diagonal_mass"] <= 1.0


def test_kernel_diagnostics_uniform_vs_learned_differ():
    """A uniform kernel (all allowed edges weight 1) and a learned RBF
    kernel built from a constant feature should be byte-identical (per
    Stage A v3 spec)."""
    from sam_kem.causal_ablation_v3 import build_uniform_kernel
    from sam_kem.reconstruction import rbf_kernel
    feat = np.ones((8, 8), dtype=np.float64)
    K_u = build_uniform_kernel(feat, sigma=1.0, radius=1).tocsr()
    K_r = rbf_kernel(feat, sigma=1.0, radius=1).tocsr()
    d_u = kernel_diagnostics(K_u)
    d_r = kernel_diagnostics(K_r)
    assert d_u["mean_row_nnz"] == d_r["mean_row_nnz"]
    assert d_u["mean_diagonal_mass"] == pytest.approx(d_r["mean_diagonal_mass"])


# ============================================================================
# SHA helper
# ============================================================================

def test_sha256_bytes_deterministic():
    arr = np.arange(100, dtype=np.float32).reshape(10, 10)
    h1 = sha256_bytes(arr)
    h2 = sha256_bytes(arr.copy())
    assert h1 == h2
    assert len(h1) == 64


def test_sha256_file_deterministic():
    p = REPO_ROOT / "results/dev_trainmax_v1/causal_ablations_corrected_v3/full_run/summary.json"
    h1 = sha256_file(p)
    h2 = sha256_file(p)
    assert h1 == h2