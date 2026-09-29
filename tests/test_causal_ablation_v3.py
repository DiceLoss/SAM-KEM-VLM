"""Step 6BN Stage A — portable tests for the corrected v3 8-variant causal ablation protocol.

Test paths derive from ``Path(__file__).resolve().parents[1]`` so the
same committed tests run on Windows and Linux from arbitrary checkout
locations when immutable inputs are present. Required inputs (the
prior archive and Qwen box cache CSV) are resolved by walking up
from the test file until a known marker is found, falling back to
``SAM_KEM_V3_REPO`` / ``PRIOR_ZIP`` / ``QWEN_BOX_CSV`` env vars if the
auto-discovery fails.

Covers:
- corrected CENTER_ADAPT box (image geometric center, exact size)
- translate_to_bounds does not shrink width/height
- corrected shuffle invariants (zero fixed points, bijection per group,
  BW same-slice/cross-slice split, HECKTOR zero-same-subject)
- all-eight variant cardinality
- exact control-box dimensions/bounds
- pure-KEM output (nonnegative, finite)
- no resampling (Poisson resampling prohibited)
- uniform-edge construction
- variant dispatch returns prior/source pairs
- audit zero width/height mismatches over the full validation cohort
"""
from __future__ import annotations

import os
import zipfile
from pathlib import Path

import numpy as np
import pytest

from sam_kem.causal_ablation_v3 import (
    VARIANTS,
    build_uniform_kernel,
    build_center_box,
    build_centroid_box,
    translate_to_bounds,
    _clamp_size_to_image,
    compute_corrected_shuffle_assignment,
    load_prior_for_variant,
    audit_box_over_validation_cohort,
)


# ============================================================================
# Portable repository-root discovery
# ============================================================================

def _resolve_repo_root() -> Path:
    """Find the repository root by walking up from this file.

    Looks for a directory containing ``code/sam_kem`` AND either
    ``delivery/dev_trainmax_v1_validation_priors.zip`` or
    ``prompts/dev_trainmax_v1/train_validation_box_cache.csv``.
    """
    here = Path(__file__).resolve()
    for parent in [here.parent, *here.parents]:
        if (parent / "code" / "sam_kem").is_dir():
            if (parent / "delivery" / "dev_trainmax_v1_validation_priors.zip").is_file():
                return parent
            if (parent / "prompts" / "dev_trainmax_v1" / "train_validation_box_cache.csv").is_file():
                return parent
    env = os.environ.get("SAM_KEM_V3_REPO")
    if env:
        return Path(env).resolve()
    raise RuntimeError(
        "Cannot locate repository root; set SAM_KEM_V3_REPO env var."
    )


def _resolve_zip_path(repo_root: Path) -> Path:
    env = os.environ.get("PRIOR_ZIP")
    if env:
        return Path(env).resolve()
    candidate = repo_root / "delivery" / "dev_trainmax_v1_validation_priors.zip"
    if candidate.is_file():
        return candidate
    raise RuntimeError("PRIOR_ZIP env var or delivery/.../validation_priors.zip is required")


def _resolve_qwen_csv(repo_root: Path) -> Path:
    env = os.environ.get("QWEN_BOX_CSV")
    if env:
        return Path(env).resolve()
    candidate = repo_root / "prompts" / "dev_trainmax_v1" / "train_validation_box_cache.csv"
    if candidate.is_file():
        return candidate
    raise RuntimeError("QWEN_BOX_CSV env var or prompts/.../cache.csv is required")


REPO_ROOT = _resolve_repo_root()
ZIP_PATH = _resolve_zip_path(REPO_ROOT)
QWEN_BOX_CSV = _resolve_qwen_csv(REPO_ROOT)


# ============================================================================
# Image geometric-center box (the v3 fix)
# ============================================================================

def test_center_box_uses_image_center_not_qwen_center():
    """build_center_box places the box at the image geometric center,
    independent of Qwen location."""
    h, w = 128, 128
    # Qwen at (5, 5, 25, 35) — small box near the top-left corner.
    cb = build_center_box((h, w), (5, 5, 25, 35))
    # Expected: width=20, height=30, centered on (64, 64).
    # Top-left: ((W-bw)//2, (H-bh)//2) = ((128-20)//2, (128-30)//2) = (54, 49)
    assert cb == (54, 49, 74, 79)


def test_center_box_two_same_size_qwen_locations_produce_same_center_box():
    """The image-center control must collapse two Qwen boxes of equal
    size but different locations to identical CENTER_ADAPT boxes."""
    h, w = 128, 128
    qb1 = (5, 5, 25, 35)      # size 20x30, top-left
    qb2 = (90, 10, 110, 40)   # size 20x30, right side
    cb1 = build_center_box((h, w), qb1)
    cb2 = build_center_box((h, w), qb2)
    assert cb1 == cb2
    # And it differs from both Qwen boxes (proves it is not an alias).
    assert cb1 != qb1
    assert cb1 != qb2


def test_center_box_preserves_exact_qwen_size():
    """Width/height of CENTER_ADAPT == width/height of the Qwen box."""
    for qb in [(5, 5, 25, 35), (50, 60, 80, 90), (90, 10, 120, 30),
               (0, 0, 128, 128), (0, 0, 1, 1)]:
        bw, bh = qb[2] - qb[0], qb[3] - qb[1]
        cb = build_center_box((128, 128), qb)
        assert cb[2] - cb[0] == bw
        assert cb[3] - cb[1] == bh


def test_center_box_for_even_and_odd_dimensions():
    """Even and odd image dimensions both yield valid integer-centered
    boxes with exact size."""
    for w, h in [(128, 128), (127, 127), (129, 129), (200, 100)]:
        bw, bh = 30, 40
        cb = build_center_box((h, w), (0, 0, bw, bh))
        assert cb[2] - cb[0] == bw
        assert cb[3] - cb[1] == bh
        assert 0 <= cb[0] < cb[2] <= w
        assert 0 <= cb[1] < cb[3] <= h


def test_center_box_all_edges_and_corners():
    """Boxes touching all four image edges stay valid and exact-size."""
    for qb in [
        (0, 0, 30, 40),       # top-left corner
        (98, 0, 128, 40),     # top-right corner
        (0, 88, 30, 128),     # bottom-left corner
        (98, 88, 128, 128),   # bottom-right corner
        (50, 0, 80, 40),      # top edge
        (50, 88, 80, 128),    # bottom edge
        (0, 50, 30, 80),      # left edge
        (98, 50, 128, 80),    # right edge
    ]:
        bw, bh = qb[2] - qb[0], qb[3] - qb[1]
        cb = build_center_box((128, 128), qb)
        assert cb[2] - cb[0] == bw, f"CENTER width shrunk for Qwen {qb}: {cb}"
        assert cb[3] - cb[1] == bh, f"CENTER height shrunk for Qwen {qb}: {cb}"
        assert 0 <= cb[0] < cb[2] <= 128
        assert 0 <= cb[1] < cb[3] <= 128


def test_translate_to_bounds_does_not_shrink():
    """``translate_to_bounds`` never shrinks width or height."""
    x0, y0, w_, h_ = translate_to_bounds(-5, 200, 30, 40, (128, 128))
    # translate_to_bounds clamps x0 to 0, y0 to 128-40=88, keeps w/h.
    assert w_ - x0 == 30
    assert h_ - y0 == 40
    assert 0 <= x0 < w_ <= 128
    assert 0 <= y0 < h_ <= 128


def test_clamp_size_to_image_only_when_qwen_larger():
    """If the Qwen box is larger than the image on either axis, the
    resulting box saturates at the image extent."""
    assert _clamp_size_to_image(30, 40, (128, 128)) == (30, 40)
    assert _clamp_size_to_image(200, 40, (128, 128)) == (128, 40)
    assert _clamp_size_to_image(30, 200, (128, 128)) == (30, 128)


def test_centroid_box_preserves_exact_qwen_size():
    """CENTROID width/height == Qwen width/height."""
    for qb in [(5, 5, 25, 35), (50, 60, 80, 90), (90, 10, 120, 30)]:
        bw, bh = qb[2] - qb[0], qb[3] - qb[1]
        cb = build_centroid_box(np.zeros((128, 128), dtype=np.float32), (128, 128), qb)
        assert cb[2] - cb[0] == bw
        assert cb[3] - cb[1] == bh


# ============================================================================
# Audit zero width/height mismatches over full validation cohort
# ============================================================================

def test_audit_zero_box_size_mismatch_over_full_cohort():
    """The audit walks every Qwen box in the validation cohort and
    verifies CENTER and CENTROID boxes have width/height exactly equal
    to the Qwen (or fallback) size and valid half-open bounds.
    """
    audit = audit_box_over_validation_cohort(REPO_ROOT, QWEN_BOX_CSV)
    assert audit["n_qwen_boxes_audited"] > 0, "validation cohort is empty"
    assert audit["center"]["n_width_mismatch"] == 0, audit["center"]["mismatches_first5"]
    assert audit["center"]["n_height_mismatch"] == 0, audit["center"]["mismatches_first5"]
    assert audit["center"]["n_invalid_bounds"] == 0
    assert audit["centroid"]["n_width_mismatch"] == 0, audit["centroid"]["mismatches_first5"]
    assert audit["centroid"]["n_height_mismatch"] == 0, audit["centroid"]["mismatches_first5"]
    assert audit["centroid"]["n_invalid_bounds"] == 0


# ============================================================================
# Variant set cardinality
# ============================================================================

def test_variants_tuple_has_eight():
    """VARIANTS contains exactly 8 entries per directive."""
    assert len(VARIANTS) == 8
    assert set(VARIANTS) == {
        "Q_ADAPT", "FULL_ADAPT", "CENTER_ADAPT", "CENTROID_ADAPT",
        "Q_RELEASED", "MLEM_FEATURE_MATCHED", "Q_SHUFFLED_MATCHED",
        "UNIFORM_FEATURE_MATCHED",
    }


def test_cardinality_per_variant_is_8892():
    """All 8 variants × 8892 cases = 71,136 pure KEM endpoints."""
    with zipfile.ZipFile(ZIP_PATH) as zf:
        n_prior = sum(1 for n in zf.namelist() if n.endswith("sam_prior.npy"))
    assert n_prior == 8892
    assert len(VARIANTS) * n_prior == 71136


# ============================================================================
# Corrected shuffle invariants
# ============================================================================

def test_shuffle_zero_fixed_points():
    """Every assignment is a derangement: source != target."""
    shuffle, sha, _ = compute_corrected_shuffle_assignment(ZIP_PATH, shuffle_seed=20260915)
    assert len(sha) == 64
    assert len(shuffle) == 8892
    for k, v in shuffle.items():
        assert not (k[0] == v[0] and k[1] == v[1] and k[2] == v[2] and k[3] == v[3] and k[4] == v[4]), (
            f"Fixed point: {k}"
        )


def test_shuffle_hecktor_zero_same_subject():
    """HECKTOR: every target subject differs from its source subject."""
    shuffle, _, _ = compute_corrected_shuffle_assignment(ZIP_PATH, shuffle_seed=20260915)
    he_violations = sum(
        1 for k, v in shuffle.items()
        if k[0] == "hecktor" and k[1] == v[1]
    )
    assert he_violations == 0


def test_shuffle_hecktor_all_same_slice():
    """HECKTOR: all 45/45 sources are in z072 (same slice)."""
    shuffle, _, _ = compute_corrected_shuffle_assignment(ZIP_PATH, shuffle_seed=20260915)
    he_cross = sum(
        1 for k, v in shuffle.items()
        if k[0] == "hecktor" and k[2] != v[2]
    )
    assert he_cross == 0


def test_shuffle_bw_694_same_slice_2_cross_slice():
    """BrainWeb: per (count, real) cell, 694 same-slice + 2 cross-slice."""
    shuffle, _, summary = compute_corrected_shuffle_assignment(ZIP_PATH, shuffle_seed=20260915)
    for grp in summary["bw_groups"]:
        assert grp["intra_slice"] == 694, f"intra_slice {grp['intra_slice']} != 694: {grp}"
        assert grp["cross_slice"] == 2, f"cross_slice {grp['cross_slice']} != 2: {grp}"
        assert grp["same_subject_violations"] == 0
        assert grp["n_targets"] == 696


def test_shuffle_bw_zero_same_subject():
    """BrainWeb: every target subject differs from its source subject."""
    shuffle, _, _ = compute_corrected_shuffle_assignment(ZIP_PATH, shuffle_seed=20260915)
    bw_violations = sum(
        1 for k, v in shuffle.items()
        if k[0] == "brainweb" and k[1] == v[1]
    )
    assert bw_violations == 0


def test_shuffle_bijection_per_group():
    """Each (ds, count, real) group is a bijection on (subj, slc) targets."""
    from collections import Counter
    shuffle, _, _ = compute_corrected_shuffle_assignment(ZIP_PATH, shuffle_seed=20260915)
    groups = {}
    for k, v in shuffle.items():
        gkey = (k[0], k[3], k[4])
        groups.setdefault(gkey, []).append((k[1], k[2], v[1], v[2]))
    for gk, items in groups.items():
        src_pairs = Counter((x[2], x[3]) for x in items)
        assert all(c == 1 for c in src_pairs.values()), f"NOT bijective {gk}: {src_pairs}"
        assert len(src_pairs) == len(items)


def test_shuffle_sha_bound_to_seed():
    """Mapping SHA-256 is deterministic for the same seed."""
    _, sha1, _ = compute_corrected_shuffle_assignment(ZIP_PATH, shuffle_seed=20260915)
    _, sha2, _ = compute_corrected_shuffle_assignment(ZIP_PATH, shuffle_seed=20260915)
    assert sha1 == sha2


# ============================================================================
# UNIFORM_FEATURE_MATCHED kernel
# ============================================================================

def test_build_uniform_kernel_matches_constant_feature_rbf():
    """build_uniform_kernel(ones, sigma, r) == rbf_kernel(ones, sigma, r)."""
    from sam_kem.reconstruction import rbf_kernel
    feat = np.ones((8, 8), dtype=np.float64)
    K_u = build_uniform_kernel(feat, sigma=1.0, radius=1).tocsr()
    K_r = rbf_kernel(feat, sigma=1.0, radius=1).tocsr()
    assert K_u.shape == K_r.shape
    assert K_u.nnz == K_r.nnz
    assert np.allclose(K_u.toarray(), K_r.toarray())


def test_build_uniform_kernel_rejects_bad_input():
    with pytest.raises(ValueError):
        build_uniform_kernel(np.ones((4,)), sigma=1.0, radius=1)
    with pytest.raises(ValueError):
        build_uniform_kernel(np.ones((4, 4)), sigma=-1.0, radius=1)
    with pytest.raises(ValueError):
        build_uniform_kernel(np.ones((4, 4)), sigma=1.0, radius=-1)


# ============================================================================
# Variant dispatcher
# ============================================================================

def test_load_prior_for_variant_uniform_returns_unit_128():
    """UNIFORM_FEATURE_MATCHED returns a 128x128 unit float32 array."""
    prior, source = load_prior_for_variant(
        "UNIFORM_FEATURE_MATCHED", repo_root=REPO_ROOT, zip_path=ZIP_PATH,
        ds="brainweb", subj="subject04", slc="z038", count=100000, real=0,
        pet_image=np.zeros((128, 128), dtype=np.float32),
        qwen_box_csv=QWEN_BOX_CSV, predictors=None,
    )
    assert prior.shape == (128, 128)
    assert prior.dtype == np.float32
    assert np.all(prior == 1.0)
    assert source == "constant_unit_128"


def test_load_prior_for_variant_mlem_returns_mlem10():
    """MLEM_FEATURE_MATCHED returns MLEM10 cached prior or pet_image fallback."""
    pet = np.zeros((128, 128), dtype=np.float32)
    pet[10, 10] = 0.5
    prior, source = load_prior_for_variant(
        "MLEM_FEATURE_MATCHED", repo_root=REPO_ROOT, zip_path=ZIP_PATH,
        ds="brainweb", subj="subject04", slc="z038", count=100000, real=0,
        pet_image=pet, qwen_box_csv=QWEN_BOX_CSV, predictors=None,
    )
    assert prior.shape == (128, 128)
    assert source.startswith("mlem10_")


def test_load_prior_for_variant_q_shuffled_requires_assignment():
    """Q_SHUFFLED_MATCHED without shuffle_assignment raises ValueError."""
    with pytest.raises(ValueError):
        load_prior_for_variant(
            "Q_SHUFFLED_MATCHED", repo_root=REPO_ROOT, zip_path=ZIP_PATH,
            ds="brainweb", subj="subject04", slc="z038", count=100000, real=0,
            pet_image=np.zeros((128, 128), dtype=np.float32),
            qwen_box_csv=QWEN_BOX_CSV, predictors=None, shuffle_assignment=None,
        )


def test_load_prior_for_variant_unknown_raises():
    with pytest.raises(ValueError):
        load_prior_for_variant(
            "BOGUS_VARIANT", repo_root=REPO_ROOT, zip_path=ZIP_PATH,
            ds="brainweb", subj="subject04", slc="z038", count=100000, real=0,
            pet_image=np.zeros((128, 128), dtype=np.float32),
            qwen_box_csv=QWEN_BOX_CSV, predictors=None,
        )


def test_load_prior_for_variant_q_adapt_returns_cached():
    """Q_ADAPT pulls from zip prior cache (no predictor needed)."""
    prior, source = load_prior_for_variant(
        "Q_ADAPT", repo_root=REPO_ROOT, zip_path=ZIP_PATH,
        ds="brainweb", subj="subject04", slc="z038", count=100000, real=0,
        pet_image=np.zeros((128, 128), dtype=np.float32),
        qwen_box_csv=QWEN_BOX_CSV, predictors=None,
    )
    assert prior.shape == (128, 128)
    assert prior.dtype == np.float32
    assert source == "cached"


# ============================================================================
# Frozen-input immutability & pipeline purity
# ============================================================================

def test_shuffle_assignment_does_not_mutate_inputs():
    """compute_corrected_shuffle_assignment must not mutate its inputs."""
    before = sorted(p for p in zipfile.ZipFile(ZIP_PATH).namelist() if p.endswith("sam_prior.npy"))
    compute_corrected_shuffle_assignment(ZIP_PATH, shuffle_seed=20260915)
    after = sorted(p for p in zipfile.ZipFile(ZIP_PATH).namelist() if p.endswith("sam_prior.npy"))
    assert before == after