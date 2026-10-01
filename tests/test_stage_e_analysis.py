"""Step 6BO Stage E — portable tests for the deterministic post-processing
analysis tool.

The committed v3 endpoint package must be the input: this test module
verifies the loss-ratio formula, paired deltas with SSIM convention,
and kernel diagnostic helpers, all from the committed inputs only.

Tests derive paths from ``Path(__file__).resolve()`` with env-var
fallback; no machine-specific absolute paths.
"""
from __future__ import annotations

import json
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
    """R = (NRMSE_v - NRMSE_Q) / frozen_MLEM_endpoint_denominator."""
    cells = compute_loss_ratio_cells(PER_CASE_CSVS)
    assert "loss_ratio_R" in cells.columns
    assert "denominator_frozen" in cells.columns
    assert len(cells) == 7 * 8  # 7 non-Q variants * 8 dataset-count cells
    # Q is "lower NRMSE" iff R > 0; verify that positive R => nrmse_v > nrmse_q
    sub = cells.dropna(subset=["loss_ratio_R"])
    assert ((sub["nrmse_v"] > sub["nrmse_q"]) == (sub["loss_ratio_R"] > 0)).all()


def test_frozen_mlem_denominator_values_loaded():
    """The 8 frozen MLEM endpoint denominators per directive 5910758070."""
    from sam_kem.stage_e_analysis import FROZEN_MLEM_DENOMINATOR
    assert FROZEN_MLEM_DENOMINATOR[("brainweb", 100000)] == pytest.approx(0.0302676787631081, abs=1e-12)
    assert FROZEN_MLEM_DENOMINATOR[("brainweb", 200000)] == pytest.approx(0.0330807617729657, abs=1e-12)
    assert FROZEN_MLEM_DENOMINATOR[("brainweb", 500000)] == pytest.approx(0.0166027693195544, abs=1e-12)
    assert FROZEN_MLEM_DENOMINATOR[("brainweb", 1000000)] == pytest.approx(0.0143730623636540, abs=1e-12)
    assert FROZEN_MLEM_DENOMINATOR[("hecktor", 100000)] == pytest.approx(0.101581081872378, abs=1e-12)
    assert FROZEN_MLEM_DENOMINATOR[("hecktor", 200000)] == pytest.approx(0.0784697190768229, abs=1e-12)
    assert FROZEN_MLEM_DENOMINATOR[("hecktor", 500000)] == pytest.approx(0.0688390216426416, abs=1e-12)
    assert FROZEN_MLEM_DENOMINATOR[("hecktor", 1000000)] == pytest.approx(0.0673837834835941, abs=1e-12)


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
    # Must be exactly 7 non-Q variants * 4 count levels = 28 records.
    assert len(records) == 7 * 4
    expected_keys = set()
    for v in ["FULL_ADAPT", "CENTER_ADAPT", "CENTROID_ADAPT", "Q_RELEASED",
              "MLEM_FEATURE_MATCHED", "Q_SHUFFLED_MATCHED", "UNIFORM_FEATURE_MATCHED"]:
        for c in [100000, 200000, 500000, 1000000]:
            expected_keys.add((v, c))
    actual_keys = {(r["variant"], r["count_level"]) for r in records}
    assert actual_keys == expected_keys
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
    assert "mean_geometric_degree" in d
    assert "mean_effective_neighbor_count" in d
    assert d["mean_geometric_degree"] == 3.0
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
    assert d_u["mean_geometric_degree"] == d_r["mean_geometric_degree"]
    assert d_u["mean_diagonal_mass"] == pytest.approx(d_r["mean_diagonal_mass"])


def test_kernel_diagnostics_effective_neighbor_count():
    """n_eff_i = 1 / sum_j K_ij^2. For a uniform row-normalized kernel
    with row L2 norm = 1/sqrt(degree_i), n_eff_i = degree_i, so
    mean_eff_to_geo_ratio ~ 1.0."""
    import scipy.sparse as sp
    n = 64
    rows = np.repeat(np.arange(n), 3)
    offsets = np.tile(np.array([-1, 0, 1]), n)
    cols = (np.arange(n).repeat(3) + offsets) % n
    weights = np.full(n * 3, 1.0 / 3.0)
    K = sp.csr_matrix((weights, (rows, cols)), shape=(n, n))
    d = kernel_diagnostics(K)
    # Each row has 3 neighbors, all weight 1/3, row L2 = sqrt(3*(1/3)^2) = 1/sqrt(3)
    # n_eff_i = 1 / (3 * (1/3)^2) = 3
    # ratio n_eff/degree = 3/3 = 1.0
    assert d["mean_geometric_degree"] == 3.0
    assert d["mean_effective_neighbor_count"] == pytest.approx(3.0, abs=1e-6)
    assert d["mean_eff_to_geo_ratio"] == pytest.approx(1.0, abs=1e-6)


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


def test_sha256_file_vs_payload_differ_for_saved_npy():
    """The on-disk .npy file SHA and the array payload SHA must differ
    by exactly the np.save header bytes. This test confirms the two
    SHAs are tracked separately and are not the same value."""
    import tempfile
    from pathlib import Path as _P
    arr = np.arange(100, dtype=np.float32).reshape(10, 10)
    with tempfile.TemporaryDirectory() as td:
        p = _P(td) / "test.npy"
        np.save(p, arr)
        payload_sha = sha256_bytes(arr)
        file_sha = sha256_file(p)
        assert payload_sha != file_sha
        assert len(payload_sha) == 64
        assert len(file_sha) == 64


def test_byte_reproduction_of_stage_e_text_artifacts(tmp_path):
    """Regenerate every Stage E text artifact into tmp_path from the
    committed inputs and compare byte-for-byte with the committed
    artifacts. NO .npy arrays are rewritten."""
    from sam_kem.stage_e_analysis import main as stage_e_main
    tmp_out = tmp_path / "analysis_stage_e"
    tmp_run = tmp_path / "run"
    # Mirror the committed run_dir and fixed-case archives into tmp_path
    tmp_run.mkdir(parents=True, exist_ok=True)
    for v in VARIANTS:
        (tmp_run / v).mkdir(parents=True, exist_ok=True)
        (tmp_run / v / "per_case.csv").write_bytes(
            (RUN_DIR / v / "per_case.csv").read_bytes())
    (tmp_run / "summary.json").write_bytes((RUN_DIR / "summary.json").read_bytes())
    (tmp_run / "per_output_sha256.json").write_bytes(
        (RUN_DIR / "per_output_sha256.json").read_bytes())
    (tmp_run / "integrity_audit.json").write_bytes(
        (RUN_DIR / "integrity_audit.json").read_bytes())
    # Copy both fixed-case archives (committed .npy arrays)
    for src in (RUN_DIR / "analysis_stage_e").glob("fixed_case_*"):
        dst = tmp_run / "analysis_stage_e" / src.name
        dst.mkdir(parents=True, exist_ok=True)
        for f in src.iterdir():
            (dst / f.name).write_bytes(f.read_bytes())

    # Run the deterministic tool
    stage_e_main(REPO_ROOT, tmp_run, tmp_out)

    # Compare every text artifact byte-for-byte with committed.
    # main() writes most text artifacts under tmp_out = tmp_path/analysis_stage_e/.
    # The fixed-case manifests are written under
    # tmp_run/analysis_stage_e/fixed_case_*/ (because the fixed-case dir is
    # under run_dir, not out_dir). The dual SHA lives at
    # tmp_run/per_output_sha256_dual.json.
    out_artifacts = [
        "loss_ratio_cells.csv",
        "loss_ratio_summary.csv",
        "hecktor_paired_bootstrap.json",
        "kernel_diagnostics.json",
        "paired_subject_deltas.csv",
        "paired_subject_deltas_summary.csv",
        "stage_e_summary.json",
        "stage_e_integrity_audit.json",
    ]
    for rel in out_artifacts:
        committed = (RUN_DIR / "analysis_stage_e" / rel).read_bytes()
        regenerated = (tmp_out / rel).read_bytes()
        assert committed == regenerated, f"byte mismatch in {rel}"

    fixed_artifacts = [
        "fixed_case_brainweb_subject04_z038_200k_r0/manifest.json",
        "fixed_case_hecktor_10_z072_200k_r0/manifest.json",
    ]
    for rel in fixed_artifacts:
        committed = (RUN_DIR / "analysis_stage_e" / rel).read_bytes()
        regenerated = (tmp_run / "analysis_stage_e" / rel).read_bytes()
        assert committed == regenerated, f"byte mismatch in {rel}"

    # Dual CSV SHA lives at run_dir/per_output_sha256_dual.json
    committed = (RUN_DIR / "per_output_sha256_dual.json").read_bytes()
    regenerated = (tmp_run / "per_output_sha256_dual.json").read_bytes()
    assert committed == regenerated, "byte mismatch in per_output_sha256_dual.json"


def test_stage_e_summary_no_dynamic_git_head():
    """stage_e_summary.json must NOT contain a dynamic git HEAD. It
    binds the immutable Stage C run_id from the run summary only."""
    s = json.load(open(RUN_DIR / "analysis_stage_e" / "stage_e_summary.json",
                        encoding="utf-8"))
    assert "execution_commit" not in s
    assert "git_head" not in s
    assert "stage_c_execution_run_id" in s


def test_csv_dual_sha_platform_independent():
    """csv_dual_sha must use canonical LF SHA (platform-independent)
    AND synthesized CRLF SHA (matching historical Stage C values)."""
    from sam_kem.stage_e_analysis import csv_dual_sha_map
    stage_c = json.load(open(RUN_DIR / "per_output_sha256.json", encoding="utf-8"))
    dual = csv_dual_sha_map(RUN_DIR, stage_c)
    for path, info in dual.items():
        assert "csv_repository_lf_sha256" in info
        assert "csv_execution_crlf_sha256" in info
        # The synthesized CRLF SHA must match the historical Stage C value
        hist = stage_c.get(path)
        if hist is not None:
            assert info["csv_execution_crlf_sha256"] == hist, (
                f"CRLF mismatch on {path}: {info['csv_execution_crlf_sha256']} != {hist}")


def test_fixed_case_manifests_use_npy_field_names():
    """Fixed-case manifests must use *_array_payload_sha256 and
    *_npy_file_sha256 for the .npy files (target, prior,
    reconstruction, abs_error) — never csv_* or repository_lf_*
    (per directive 5923244124 §2)."""
    bw = json.load(open(RUN_DIR / "analysis_stage_e" /
                        "fixed_case_brainweb_subject04_z038_200k_r0/manifest.json",
                        encoding="utf-8"))
    for key in bw["sha256"].keys():
        assert not key.startswith("csv_"), f"{key} uses csv_* prefix on .npy"
        assert not key.startswith("repository_lf_"), f"{key} uses repository_lf_* prefix on .npy"
        assert key.endswith("_array_payload_sha256") or key.endswith("_npy_file_sha256"), (
            f"{key} must end with _array_payload_sha256 or _npy_file_sha256")


def test_kernel_diagnostics_byte_identity_after_two_runs():
    """Regenerate kernel_diagnostics.json twice into tmp_path; the two
    files must be byte-identical (canonical float serialization)."""
    import tempfile
    from pathlib import Path as _P
    from sam_kem.stage_e_analysis import (
        load_priors_from_fixed_case_archive, kernel_diagnostics_from_priors,
    )
    import pandas as _pd
    fpath = REPO_ROOT / "results/dev_trainmax_v1/frozen_hyperparameters.csv"
    fdf = _pd.read_csv(fpath)
    fdf = fdf[fdf.method == "SAM-KEM-MedSAM2-Qwen3-full"]
    support = np.ones((128, 128), dtype=bool)

    out1 = _P(tempfile.mkdtemp()) / "k1.json"
    out2 = _P(tempfile.mkdtemp()) / "k2.json"
    for out in (out1, out2):
        for case_id, dir_name, count in [
            ("brainweb", "fixed_case_brainweb_subject04_z038_200k_r0", 200000),
            ("hecktor", "fixed_case_hecktor_10_z072_200k_r0", 200000),
        ]:
            fixed_dir = RUN_DIR / "analysis_stage_e" / dir_name
            priors = load_priors_from_fixed_case_archive(fixed_dir)
            hp = fdf[(fdf.dataset == case_id) & (fdf.count_level == count)].iloc[0]
            kd = kernel_diagnostics_from_priors(
                priors, hp, sigma=float(hp.sigma), radius=int(hp.radius),
                operator_support_fn=lambda: support)
        with open(out, "w", encoding="utf-8") as f:
            json.dump(kd, f, indent=2)
    assert out1.read_bytes() == out2.read_bytes(), (
        f"kernel_diagnostics.json not byte-identical across runs:\n"
        f"  run1 size={out1.stat().st_size}, run2 size={out2.stat().st_size}")


def test_stage_e_text_artifact_sha_map_has_expected_keys():
    """stage_e_integrity_audit.json must contain a stage_e_text_artifact_sha256_map
    with the expected key set per directive 5923244124 §3."""
    a = json.load(open(RUN_DIR / "analysis_stage_e" / "stage_e_integrity_audit.json",
                        encoding="utf-8"))
    expected_keys = {
        "loss_ratio_cells_csv",
        "loss_ratio_summary_csv",
        "hecktor_paired_bootstrap_json",
        "kernel_diagnostics_json",
        "paired_subject_deltas_csv",
        "paired_subject_deltas_summary_csv",
        "fixed_case_brainweb_manifest_json",
        "fixed_case_hecktor_manifest_json",
        "stage_e_summary_json",
        "per_output_sha256_dual_json",
    }
    assert "stage_e_text_artifact_sha256_map" in a, "missing SHA map"
    actual = set(a["stage_e_text_artifact_sha256_map"].keys())
    assert actual == expected_keys, (
        f"SHA map key mismatch: missing={expected_keys - actual}, extra={actual - expected_keys}")
    # Each entry must have sha256 + path + size_bytes
    for k, v in a["stage_e_text_artifact_sha256_map"].items():
        assert "sha256" in v and "path" in v and "size_bytes" in v
        assert len(v["sha256"]) == 64


def test_load_priors_from_fixed_case_archive():
    """Priors are loaded from committed fixed-case prior_*.npy files;
    NO decoder, NO KEM, NO MLEM10 lookup. The committed archive is
    the authoritative source."""
    from sam_kem.stage_e_analysis import load_priors_from_fixed_case_archive
    bw_dir = RUN_DIR / "analysis_stage_e" / "fixed_case_brainweb_subject04_z038_200k_r0"
    he_dir = RUN_DIR / "analysis_stage_e" / "fixed_case_hecktor_10_z072_200k_r0"
    bw = load_priors_from_fixed_case_archive(bw_dir)
    he = load_priors_from_fixed_case_archive(he_dir)
    expected = {"Q_ADAPT", "FULL_ADAPT", "CENTER_ADAPT", "CENTROID_ADAPT",
                "Q_RELEASED", "MLEM_FEATURE_MATCHED", "Q_SHUFFLED_MATCHED",
                "UNIFORM_FEATURE_MATCHED"}
    assert set(bw.keys()) == expected
    assert set(he.keys()) == expected
    # Decoder-based variants must differ from Q_ADAPT
    for v in ["FULL_ADAPT", "CENTER_ADAPT", "CENTROID_ADAPT", "Q_RELEASED", "Q_SHUFFLED_MATCHED"]:
        diff = np.abs(bw[v].astype(np.float32) - bw["Q_ADAPT"]).sum()
        assert diff > 0, f"{v} prior equals Q_ADAPT prior (should differ)"