"""Step 6BM Stage A — corrected 8-variant causal ablation protocol.

Implements the eight matched pure-KEM variants per directive 5882184688:
  1. Q_ADAPT
  2. FULL_ADAPT
  3. CENTER_ADAPT
  4. CENTROID_ADAPT
  5. Q_RELEASED
  6. MLEM_FEATURE_MATCHED
  7. Q_SHUFFLED_MATCHED  (corrected BW shuffle per directive)
  8. UNIFORM_FEATURE_MATCHED  (all local RBF edges weight 1)

All variants share the same archived observation, operator, gain/background,
initialization, support, metrics, and frozen Q_ADAPT SAM-KEM tuple. The
final image is the pure KEM reconstruction (alpha=0, no post-fusion).
"""
from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


# ============================================================================
# Variant specifications
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


# ============================================================================
# Qwen box retrieval
# ============================================================================

def qwen_box_lookup(qwen_box_csv: Path, ds: str, subj: str, slc: str, count: int):
    df = pd.read_csv(qwen_box_csv, low_memory=False)
    sub = df[(df["dataset"]==ds) & (df["subject_id"]==subj) & (df["slice_id"]==slc)
             & (df["count_level"]==count)]
    if len(sub) == 0:
        return None
    for _, row in sub.iterrows():
        try:
            xmin = int(row["xmin"]); ymin = int(row["ymin"])
            xmax = int(row["xmax"]); ymax = int(row["ymax"])
            if xmax > xmin and ymax > ymin:
                return (xmin, ymin, xmax, ymax)
        except (ValueError, TypeError):
            continue
    return None


# ============================================================================
# Box builders
# ============================================================================

def build_full_image_box(image_shape):
    h, w = image_shape
    return (0, 0, w, h)


def build_center_box(image_shape, qwen_box):
    h, w = image_shape
    qx0, qy0, qx1, qy1 = qwen_box
    bw = qx1 - qx0
    bh = qy1 - qy0
    cx, cy = (qx0 + qx1) // 2, (qy0 + qy1) // 2
    x0 = max(0, cx - bw // 2)
    y0 = max(0, cy - bh // 2)
    x1 = min(w, x0 + bw)
    y1 = min(h, y0 + bh)
    return (x0, y0, x1, y1)


def build_centroid_box(mlem10: np.ndarray, image_shape, qwen_box):
    h, w = image_shape
    qx0, qy0, qx1, qy1 = qwen_box
    bw = qx1 - qx0
    bh = qy1 - qy0
    arr = np.asarray(mlem10, dtype=np.float64)
    total = float(arr.sum())
    if total <= 0:
        return (max(0, w//2 - bw//2), max(0, h//2 - bh//2), min(w, w//2 + bw//2), min(h, h//2 + bh//2))
    yy, xx = np.indices(arr.shape)
    cx = float((xx * arr).sum() / total)
    cy = float((yy * arr).sum() / total)
    x0 = int(max(0, round(cx - bw/2)))
    y0 = int(max(0, round(cy - bh/2)))
    x1 = int(min(w, x0 + bw))
    y1 = int(min(h, y0 + bh))
    return (x0, y0, x1, y1)


def decode_prior_with_predictor(predictor, pet: np.ndarray, box: tuple, image_shape) -> np.ndarray:
    """Run the decoder with the given box, return 128x128 float32 logits."""
    from .medsam import predict_continuous_prior, Box
    bx = Box(xmin=int(box[0]), ymin=int(box[1]), xmax=int(box[2]), ymax=int(box[3]))
    return predict_continuous_prior(predictor, pet, bx, output_shape=image_shape).astype(np.float32)


# ============================================================================
# Prior loading dispatcher
# ============================================================================

def load_prior_for_variant(
    variant: str, *,
    repo_root: Path, zip_path: Path,
    ds: str, subj: str, slc: str, count: int, real: int,
    pet_image: np.ndarray,
    qwen_box_csv: Path, predictors: dict,
    shuffle_assignment: Optional[dict] = None,
):
    """Return (prior: np.ndarray, source: str)."""
    qwen_box = qwen_box_lookup(qwen_box_csv, ds, subj, slc, count)
    image_shape = pet_image.shape

    if variant == "Q_ADAPT":
        prior = _load_qwen_prior(zip_path, ds, subj, slc, count, real)
        return prior, "cached"

    if variant in ("FULL_ADAPT", "CENTER_ADAPT", "CENTROID_ADAPT"):
        adapted_key = f"pet_adapted_{ds}"
        if qwen_box is None:
            box = build_full_image_box(image_shape)
        elif variant == "FULL_ADAPT":
            box = build_full_image_box(image_shape)
        elif variant == "CENTER_ADAPT":
            box = build_center_box(image_shape, qwen_box)
        else:  # CENTROID_ADAPT
            mlem10_path = repo_root / f"arrays/dev_trainmax_v1/{ds}/{subj}/{slc}/c{count}/r{real}/mlem10.npy"
            if mlem10_path.exists():
                mlem10 = np.load(mlem10_path).astype(np.float32)
            else:
                mlem10 = pet_image
            box = build_centroid_box(mlem10, image_shape, qwen_box)
        prior = decode_prior_with_predictor(predictors[adapted_key], pet_image, box, image_shape)
        return prior, f"decoded/{variant}"

    if variant == "Q_RELEASED":
        if "released" not in predictors:
            raise RuntimeError("Released MedSAM2 predictor not loaded")
        if qwen_box is None:
            box = build_full_image_box(image_shape)
        else:
            box = qwen_box
        prior = decode_prior_with_predictor(predictors["released"], pet_image, box, image_shape)
        return prior, "decoded/Q_RELEASED"

    if variant == "MLEM_FEATURE_MATCHED":
        mlem10_path = repo_root / f"arrays/dev_trainmax_v1/{ds}/{subj}/{slc}/c{count}/r{real}/mlem10.npy"
        if mlem10_path.exists():
            prior = np.load(mlem10_path).astype(np.float32)
        else:
            prior = pet_image.astype(np.float32)
        return prior, "mlem10_cached"

    if variant == "Q_SHUFFLED_MATCHED":
        if shuffle_assignment is None:
            raise ValueError("Q_SHUFFLED_MATCHED requires shuffle_assignment")
        src = shuffle_assignment.get((ds, subj, slc, int(count), int(real)))
        if src is None:
            raise KeyError(f"No shuffle source for {(ds, subj, slc, count, real)}")
        src_ds, src_subj, src_slc, src_count, src_real = src
        prior = _load_qwen_prior(zip_path, src_ds, src_subj, src_slc, src_count, src_real)
        return prior, f"shuffled_from={src_ds}/{src_subj}/{src_slc}/c{src_count}/r{src_real}"

    if variant == "UNIFORM_FEATURE_MATCHED":
        # Constant 128x128 unit feature; all allowed local RBF edges weight 1
        # Per-row normalization happens in rbf_kernel itself (this just sets the source)
        h, w = 128, 128
        return np.ones((h, w), dtype=np.float32), "constant_unit_128"

    raise ValueError(f"Unknown variant: {variant}")


def _load_qwen_prior(zip_path: Path, ds: str, subj: str, slc: str, count: int, real: int) -> np.ndarray:
    cstr = "c" + str(count)
    inner_path = f"arrays/dev_trainmax_v1/{ds}/{subj}/{slc}/{cstr}/r{real}/sam_prior.npy"
    with zipfile.ZipFile(zip_path, "r") as zf:
        with zf.open(inner_path) as f:
            return np.load(f, allow_pickle=False).astype(np.float32)


# ============================================================================
# Corrected Q_SHUFFLED_MATCHED derangement
#
# Per directive 5882184688:
#   BrainWeb: 694/696 same-slice + 2 cross-slice repairs per (count, real) cell.
#     Repair #1: target (brainweb, subject50, z210) <- source (brainweb, subject44, z213)
#     Repair #2: target (brainweb, subject44, z213) <- source (brainweb, subject04, z210)
#     All other BW targets within a slice get a same-slice subject rotation (cyclic).
#   HECKTOR: 45/45 same-slice (z072) derangement per (count, real) cell.
#
# Implementation: build the assignment by walking the case list. For each
# (ds, count, real) group, sort by (slice, subject); apply per-slice
# rotation (cyclic successor within the same slice), then explicitly
# override the two specified repairs, and for HE apply cyclic successor
# within the (ds, count, real) group (one slice, so equivalent to slice-
# level rotation).
# ============================================================================

def compute_corrected_shuffle_assignment(zip_path: Path, shuffle_seed: int = 20260915) -> tuple:
    """Return (shuffle_assignment: dict, derangement_sha: str, summary: dict).

    shuffle_assignment maps (ds, subj, slc, count, real) -> (src_ds, src_subj, src_slc, src_count, src_real).
    """
    # Discover all cases per (ds, count, real)
    with zipfile.ZipFile(zip_path, "r") as zf:
        paths = [n for n in zf.namelist() if n.endswith("sam_prior.npy")]
    cases_by_key = {}  # (ds, count, real) -> [(subj, slc)]
    for p in paths:
        parts = p.split("/")
        ds_p, subj_p, slc_p, cstr, rstr = parts[2], parts[3], parts[4], parts[5], parts[6]
        if not cstr.startswith("c") or not rstr.startswith("r"):
            continue
        count_p = int(cstr[1:])
        real_p = int(rstr[1:])
        cases_by_key.setdefault((ds_p, count_p, real_p), []).append((subj_p, slc_p))

    shuffle = {}
    summary = {"bw_groups": [], "he_groups": [], "bw_cross_slice_repairs": []}

    for (ds, count, real), cs in cases_by_key.items():
        cs_sorted = sorted(cs)
        n = len(cs_sorted)
        if n < 2:
            continue

        # Group by slice
        by_slice = {}
        for subj, slc in cs_sorted:
            by_slice.setdefault(slc, []).append(subj)

        if ds == "brainweb":
            # For each shared slice (>=2 subjects), apply cyclic successor within slice.
            # Singleton targets (e.g. z213 has 1 subject) are then overridden by the
            # explicit cross-slice repairs below.
            intra_slice_assignments = {}
            cross_slice_repairs = {}
            for slc, subjects in by_slice.items():
                subjects_sorted = sorted(subjects)
                k = len(subjects_sorted)
                if k >= 2:
                    for i, t_subj in enumerate(subjects_sorted):
                        s_subj = subjects_sorted[(i + 1) % k]
                        intra_slice_assignments[(slc, t_subj)] = s_subj
            # Small slices:
            # z210 (3 subjects): subject04, subject42, subject44 (verify in by_slice)
            # z211 (2): ?
            # z212 (2): ?
            # z213 (1): subject04
            # Compute the actual subjects present in z210 and z213 only
            # (z211/z212 are auto-handled by per-slice cyclic successor)
            z210_subs = sorted(by_slice.get("z210", []))
            z213_subs = sorted(by_slice.get("z213", []))

            # Repair #1: target (subject50, z210) <- source (subject44, z213)
            # Requires subject50 present in z210, subject44 present in z213
            if "subject50" in z210_subs and "subject44" in z213_subs:
                cross_slice_repairs[("z210", "subject50")] = ("subject44", "z213")
            # Repair #2: target (subject44, z213) <- source (subject04, z210)
            if "subject44" in z213_subs and "subject04" in z210_subs:
                cross_slice_repairs[("z213", "subject44")] = ("subject04", "z210")

            # Build the (count, real) cell assignment
            intra_count = 0
            cross_count = 0
            same_subj_violations = 0
            for subj, slc in cs_sorted:
                if (slc, subj) in cross_slice_repairs:
                    src_subj, src_slc = cross_slice_repairs[(slc, subj)]
                    src = (ds, src_subj, src_slc, count, real)
                else:
                    src_subj = intra_slice_assignments.get((slc, subj), subj)
                    src = (ds, src_subj, slc, count, real)
                shuffle[(ds, subj, slc, count, real)] = src
                intra_count += int((slc, subj) not in cross_slice_repairs)
                cross_count += int((slc, subj) in cross_slice_repairs)
                if src_subj == subj:
                    same_subj_violations += 1

            summary["bw_groups"].append({
                "count": count, "real": real, "n_targets": n,
                "intra_slice": intra_count, "cross_slice": cross_count,
                "same_subject_violations": same_subj_violations,
            })
            summary["bw_cross_slice_repairs"].append({
                "count": count, "real": real,
                "repairs": list(cross_slice_repairs.items()),
            })
        elif ds == "hecktor":
            # Single slice (z072), 45 subjects: cyclic successor within (count, real)
            for i, (t_subj, t_slc) in enumerate(cs_sorted):
                s_subj, s_slc = cs_sorted[(i + 1) % n]
                shuffle[(ds, t_subj, t_slc, count, real)] = (ds, s_subj, s_slc, count, real)
            same_subj_violations = sum(
                1 for k, v in shuffle.items()
                if k[0] == "hecktor" and k[1] == v[1] and k[3] == count and k[4] == real
            )
            summary["he_groups"].append({
                "count": count, "real": real, "n_targets": n,
                "same_subject_violations": same_subj_violations,
            })
        else:
            raise ValueError(f"Unknown dataset: {ds}")

    # Verify no fixed points anywhere
    for k, v in shuffle.items():
        ds_k, subj_k, slc_k, count_k, real_k = k
        if (v[0], v[1], v[2]) == (ds_k, subj_k, slc_k):
            raise RuntimeError(f"Fixed point detected: {k}")

    # SHA the assignment
    sorted_records = sorted([f"{k}->{v}" for k, v in shuffle.items()])
    sha = hashlib.sha256(json.dumps(sorted_records).encode()).hexdigest()

    return shuffle, sha, summary


# ============================================================================
# UNIFORM_FEATURE_MATCHED kernel
# ============================================================================

def build_uniform_kernel(feature: np.ndarray, sigma: float, radius: int,
                         support: np.ndarray | None = None):
    """Build a row-normalized RBF kernel where every allowed local edge has
    weight 1 before the row-normalization step. Pass feature=ones (constant)
    so the Gaussian-decay collapses to 1 for all (i,j) within Manhattan
    distance r.

    Returns a CSR kernel identical in shape to ``rbf_kernel(constant_feature, ...)``
    but with weight=1 instead of ``exp(-(diff**2)/(2*sigma**2))``. Uses the
    same Manhattan-radius neighborhood as ``rbf_kernel``.
    """
    import scipy.sparse as sp
    values = np.asarray(feature, dtype=np.float64)
    if values.ndim != 2 or sigma <= 0 or radius < 0:
        raise ValueError("feature must be 2-D, sigma > 0 and radius >= 0")
    active = np.ones_like(values, dtype=bool)
    if support is not None:
        active &= support.astype(bool)
    height, width = values.shape
    rows, cols, weights = [], [], []
    for r in range(height):
        for c in range(width):
            i = r * width + c
            if not active[r, c]:
                continue
            for dr in range(-radius, radius + 1):
                for dc in range(-radius, radius + 1):
                    if abs(dr) + abs(dc) > radius:
                        continue
                    rr, cc = r + dr, c + dc
                    if 0 <= rr < height and 0 <= cc < width and active[rr, cc]:
                        j = rr * width + cc
                        rows.append(i); cols.append(j); weights.append(1.0)
    K = sp.csr_matrix((weights, (rows, cols)), shape=(height * width, height * width))
    row_sums = np.asarray(K.sum(axis=1)).ravel()
    row_sums = np.maximum(row_sums, 1e-12)
    return K.multiply(1.0 / row_sums[:, None])
