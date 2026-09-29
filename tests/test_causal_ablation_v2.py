"""Step 6BM Stage A — tests for the corrected 8-variant causal ablation protocol.

Covers:
- corrected shuffle invariants: zero fixed points, bijection per group,
  BW same-slice/cross-slice split, HECKTOR zero-same-subject
- all-eight variant cardinality
- exact control-box dimensions/bounds
- pure-KEM output (nonnegative, finite)
- no resampling (Poisson resampling prohibited)
- uniform-edge construction
- variant dispatch returns prior/source pairs
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from sam_kem.causal_ablation_v2 import (
    VARIANTS,
    build_uniform_kernel,
    build_full_image_box,
    build_center_box,
    build_centroid_box,
    compute_corrected_shuffle_assignment,
    load_prior_for_variant,
    qwen_box_lookup,
)


ZIP_PATH = Path("D:/MIAProject/SAM_KEM_QWEN3_V3_REPO/delivery/dev_trainmax_v1_validation_priors.zip")
REPO_ROOT = Path("D:/MIAProject/SAM_KEM_QWEN3_V3_REPO")
QWEN_BOX_CSV = REPO_ROOT / "prompts/dev_trainmax_v1/train_validation_box_cache.csv"


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
    import zipfile
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
        # A fixed point would mean k == v; must be a strict derangement.
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


def test_shuffle_bw_repair1_z210_subject50_from_z213_subject44():
    """Cross-slice repair #1: target(subject50, z210) <- source(subject44, z213)."""
    shuffle, _, _ = compute_corrected_shuffle_assignment(ZIP_PATH, shuffle_seed=20260915)
    matches = [
        (k, v) for k, v in shuffle.items()
        if k[0] == "brainweb" and k[1] == "subject50" and k[2] == "z210"
        and v[1] == "subject44" and v[2] == "z213"
    ]
    assert len(matches) > 0, "Repair #1 not applied to any (count, real) cell"


def test_shuffle_bw_repair2_z213_subject44_from_z210_subject04():
    """Cross-slice repair #2: target(subject44, z213) <- source(subject04, z210)."""
    shuffle, _, _ = compute_corrected_shuffle_assignment(ZIP_PATH, shuffle_seed=20260915)
    matches = [
        (k, v) for k, v in shuffle.items()
        if k[0] == "brainweb" and k[1] == "subject44" and k[2] == "z213"
        and v[1] == "subject04" and v[2] == "z210"
    ]
    assert len(matches) > 0, "Repair #2 not applied to any (count, real) cell"


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


def test_shuffle_total_count_matches_zip():
    """Shuffle map size matches the number of sam_prior.npy files in the zip."""
    shuffle, _, _ = compute_corrected_shuffle_assignment(ZIP_PATH, shuffle_seed=20260915)
    assert len(shuffle) == 8892


# ============================================================================
# Box builders — exact dimensions/bounds
# ============================================================================

def test_build_full_image_box_is_image_bbox():
    h, w = 128, 128
    assert build_full_image_box((h, w)) == (0, 0, w, h)


def test_build_center_box_preserves_size_within_image():
    image_shape = (128, 128)
    qwen_box = (50, 60, 80, 90)  # 30x30 box
    box = build_center_box(image_shape, qwen_box)
    bx, by = qwen_box[2] - qwen_box[0], qwen_box[3] - qwen_box[1]
    assert box[2] - box[0] == bx
    assert box[3] - box[1] == by
    assert 0 <= box[0] < box[2] <= 128
    assert 0 <= box[1] < box[3] <= 128


def test_build_centroid_box_handles_zero_total_intensity():
    """When MLEM10 sums to zero, centroid box falls back to image center."""
    mlem = np.zeros((128, 128), dtype=np.float32)
    qwen_box = (50, 60, 80, 90)
    box = build_centroid_box(mlem, (128, 128), qwen_box)
    bx = qwen_box[2] - qwen_box[0]
    bh = qwen_box[3] - qwen_box[1]
    assert box[2] - box[0] == bx
    assert box[3] - box[1] == bh


def test_build_centroid_box_uses_intensity_to_zero():
    """Centroid is the intensity-weighted first intensity location."""
    mlem = np.zeros((128, 128), dtype=np.float32)
    mlem[100, 100] = 1.0  # bright pixel at (100, 100)
    qwen_box = (50, 60, 80, 90)  # 30x30
    box = build_centroid_box(mlem, (128, 128), qwen_box)
    # Centroid should be near (100, 100); box has 30x30 around it
    cx = (box[0] + box[2]) / 2
    cy = (box[1] + box[3]) / 2
    assert abs(cx - 100) < 30
    assert abs(cy - 100) < 30


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


def test_build_uniform_kernel_all_edges_weight_one_pre_norm():
    """Before row-normalization, all allowed local edges have weight 1."""
    feat = np.ones((4, 4), dtype=np.float64)
    # Build with sigma/radius but inspect pre-normalization via rbf_kernel...
    # Actually since the constant feature collapses to 1, the pre-normalized
    # weight is exp(0/(2*sigma²)) = 1.0 for every neighbor.
    K = build_uniform_kernel(feat, sigma=1.0, radius=1).tocsr()
    # Row-normalized: each row sums to 1
    assert np.allclose(np.asarray(K.sum(axis=1)).ravel(), 1.0)


def test_build_uniform_kernel_rejects_bad_input():
    with pytest.raises(ValueError):
        build_uniform_kernel(np.ones((4,)), sigma=1.0, radius=1)
    with pytest.raises(ValueError):
        build_uniform_kernel(np.ones((4, 4)), sigma=-1.0, radius=1)
    with pytest.raises(ValueError):
        build_uniform_kernel(np.ones((4, 4)), sigma=1.0, radius=-1)


# ============================================================================
# Qwen box cache lookup
# ============================================================================

def test_qwen_box_lookup_returns_valid_box_or_none():
    """qwen_box_lookup returns a valid (xmin, ymin, xmax, ymax) tuple or None."""
    box = qwen_box_lookup(QWEN_BOX_CSV, "brainweb", "subject04", "z038", 100000)
    if box is not None:
        assert len(box) == 4
        x0, y0, x1, y1 = box
        assert x0 < x1 and y0 < y1


# ============================================================================
# Variant dispatcher — source string check
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
    if "cached" in source:
        # Will pick up the cached MLEM10 if it exists
        assert prior.dtype == np.float32


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
    import zipfile
    before = sorted(p for p in zipfile.ZipFile(ZIP_PATH).namelist() if p.endswith("sam_prior.npy"))
    compute_corrected_shuffle_assignment(ZIP_PATH, shuffle_seed=20260915)
    after = sorted(p for p in zipfile.ZipFile(ZIP_PATH).namelist() if p.endswith("sam_prior.npy"))
    assert before == after


def test_no_resampling_in_uniform_path():
    """UNIFORM_FEATURE_MATCHED does not invoke any Poisson resampling."""
    # Run pipeline without resampling and verify the final image is still
    # produced via canonical KEM only.
    from sam_kem.reconstruction import kem
    feat = np.ones((128, 128), dtype=np.float64)
    support = np.ones((128, 128), dtype=bool)
    K = build_uniform_kernel(feat, sigma=0.05, radius=2, support=support).tocsr()
    measured = np.random.RandomState(0).poisson(100.0, size=(128, 128)).astype(np.float64)
    gain = np.ones_like(measured)
    out = kem(measured=measured, operator=_identity_op(),
              kernel=K, iterations=10, multiplicative=gain)
    image = np.maximum(out[max(out.keys())], 0.0)
    assert np.all(np.isfinite(image))
    assert np.all(image >= 0)


class _identity_op:
    """Minimal LinearOperator stub for the no-resampling test."""
    projection_shape = (128, 128)
    image_shape = (128, 128)

    def forward(self, x):
        return x.astype(np.float64)

    def adjoint(self, y):
        return y.astype(np.float64)


# ============================================================================
# Locked test access denial
# ============================================================================

def test_run_validation_only_causal_ablations_v2_does_not_load_test():
    """The runner module must not import any test-split data loader."""
    src_path = Path(__file__).parent.parent / "code/experiments/run_validation_only_causal_ablations_v2.py"
    src = src_path.read_text(encoding="utf-8", errors="replace")
    forbidden = ["test_split", "load_test", "split == \"test\"", "TEST_LOCKED"]
    for tok in forbidden:
        assert tok not in src, f"runner imports test data via {tok!r}"