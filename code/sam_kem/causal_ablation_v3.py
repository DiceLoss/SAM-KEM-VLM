"""Step 6BN Stage A — corrected 8-variant causal ablation protocol (v3).

Fixes per directive 5890879059:

1. ``build_center_box`` now places the box at the **image geometric
   center**, preserving the Qwen (or fallback) width/height. It no
   longer aliases Q_ADAPT.
2. A single ``translate_to_bounds`` helper handles edge clamping for
   both CENTER and CENTROID controls without shrinking width or height.
3. ``build_centroid_box`` continues to use the preliminary MLEM10
   intensity centroid; with ``translate_to_bounds`` it never shrinks.
4. Corrected BrainWeb ``Q_SHUFFLED_MATCHED`` derangement (same-slice
   cyclic rotation + two explicit cross-slice repairs).
5. UNIFORM_FEATURE_MATCHED: constant 128x128 unit feature; all
   allowed local RBF edges weight 1 before row normalization.

All eight variants share the same archived observation, operator,
gain/background, initialization, support, metrics, and frozen Q_ADAPT
SAM-KEM tuple. Final image is the pure KEM reconstruction (alpha=0,
no post-fusion).
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
# Box builders — fixed per directive 5890879059
# ============================================================================

def _clamp_size_to_image(width: int, height: int, image_shape) -> tuple[int, int]:
    """Clip a requested (width, height) to the image's max possible extent.

    If the Qwen box is larger than the image on either axis, the
    resulting box saturates at ``min(width, W)`` and
    ``min(height, H)``. This is the *only* place where a box's
    width/height may shrink.
    """
    h, w = image_shape
    return min(width, w), min(height, h)


def translate_to_bounds(x0: int, y0: int, width: int, height: int,
                        image_shape) -> tuple[int, int, int, int]:
    """Place a box of fixed (width, height) at the requested top-left,
    translating the whole box inward if any side would fall off the
    image. Never shrinks the box.
    """
    h, w = image_shape
    if x0 < 0:
        x0 = 0
    if y0 < 0:
        y0 = 0
    if x0 + width > w:
        x0 = w - width
    if y0 + height > h:
        y0 = h - height
    # If width/height exceed the image (shouldn't happen if
    # _clamp_size_to_image was called), saturate.
    if x0 < 0:
        x0 = 0
    if y0 < 0:
        y0 = 0
    return x0, y0, x0 + width, y0 + height


def build_full_image_box(image_shape):
    h, w = image_shape
    return (0, 0, w, h)


def build_center_box(image_shape, qwen_box) -> tuple[int, int, int, int]:
    """Image geometric-center control: same width/height as the Qwen
    box, placed at the image geometric center. Independent of Qwen
    location.
    """
    h, w = image_shape
    qx0, qy0, qx1, qy1 = qwen_box
    bw, bh = qx1 - qx0, qy1 - qy0
    bw, bh = _clamp_size_to_image(bw, bh, image_shape)
    # Image geometric center: the box is centered at (W/2, H/2).
    # For integer arithmetic we use floor on the top-left and rely on
    # translate_to_bounds to keep it inside the image.
    cx = (w - bw) // 2
    cy = (h - bh) // 2
    return translate_to_bounds(cx, cy, bw, bh, image_shape)


def build_centroid_box(mlem10: np.ndarray, image_shape, qwen_box) -> tuple[int, int, int, int]:
    """Intensity-weighted centroid control: same width/height as the
    Qwen box, centered on the MLEM10 first intensity-moment centroid.
    """
    h, w = image_shape
    qx0, qy0, qx1, qy1 = qwen_box
    bw, bh = qx1 - qx0, qy1 - qy0
    bw, bh = _clamp_size_to_image(bw, bh, image_shape)
    arr = np.asarray(mlem10, dtype=np.float64)
    total = float(arr.sum())
    if total <= 0:
        cx = (w - bw) // 2
        cy = (h - bh) // 2
    else:
        yy, xx = np.indices(arr.shape)
        cxx = float((xx * arr).sum() / total)
        cyy = float((yy * arr).sum() / total)
        # Top-left so that the box center is at (cxx, cyy).
        cx = int(round(cxx - bw / 2.0))
        cy = int(round(cyy - bh / 2.0))
    return translate_to_bounds(cx, cy, bw, bh, image_shape)


# ============================================================================
# Decoder invocation helper
# ============================================================================

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
    bw, bh = _qwen_or_fallback_size(qwen_box, image_shape)

    if variant == "Q_ADAPT":
        prior = _load_qwen_prior(zip_path, ds, subj, slc, count, real)
        return prior, "cached"

    if variant in ("FULL_ADAPT", "CENTER_ADAPT", "CENTROID_ADAPT"):
        adapted_key = f"pet_adapted_{ds}"
        if variant == "FULL_ADAPT":
            box = build_full_image_box(image_shape)
        elif variant == "CENTER_ADAPT":
            box = build_center_box(image_shape, (0, 0, bw, bh))
        else:  # CENTROID_ADAPT
            mlem10_path = repo_root / f"arrays/dev_trainmax_v1/{ds}/{subj}/{slc}/c{count}/r{real}/mlem10.npy"
            if mlem10_path.exists():
                mlem10 = np.load(mlem10_path).astype(np.float32)
            else:
                mlem10 = pet_image
            box = build_centroid_box(mlem10, image_shape, (0, 0, bw, bh))
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


def _qwen_or_fallback_size(qwen_box, image_shape) -> tuple[int, int]:
    """Qwen width/height if available; otherwise full image. Used as
    the canonical reference size for CENTER/CENTROID controls."""
    if qwen_box is None:
        h, w = image_shape
        return w, h
    qx0, qy0, qx1, qy1 = qwen_box
    return qx1 - qx0, qy1 - qy0


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
# ============================================================================

def compute_corrected_shuffle_assignment(zip_path: Path, shuffle_seed: int = 20260915) -> tuple:
    """Return (shuffle_assignment: dict, derangement_sha: str, summary: dict).

    shuffle_assignment maps (ds, subj, slc, count, real) -> (src_ds, src_subj, src_slc, src_count, src_real).
    """
    with zipfile.ZipFile(zip_path, "r") as zf:
        paths = [n for n in zf.namelist() if n.endswith("sam_prior.npy")]
    cases_by_key = {}
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

        by_slice = {}
        for subj, slc in cs_sorted:
            by_slice.setdefault(slc, []).append(subj)

        if ds == "brainweb":
            intra_slice_assignments = {}
            cross_slice_repairs = {}
            for slc, subjects in by_slice.items():
                subjects_sorted = sorted(subjects)
                k = len(subjects_sorted)
                if k >= 2:
                    for i, t_subj in enumerate(subjects_sorted):
                        s_subj = subjects_sorted[(i + 1) % k]
                        intra_slice_assignments[(slc, t_subj)] = s_subj

            z210_subs = sorted(by_slice.get("z210", []))
            z213_subs = sorted(by_slice.get("z213", []))

            if "subject50" in z210_subs and "subject44" in z213_subs:
                cross_slice_repairs[("z210", "subject50")] = ("subject44", "z213")
            if "subject44" in z213_subs and "subject04" in z210_subs:
                cross_slice_repairs[("z213", "subject44")] = ("subject04", "z210")

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

    for k, v in shuffle.items():
        ds_k, subj_k, slc_k, count_k, real_k = k
        if (v[0], v[1], v[2]) == (ds_k, subj_k, slc_k):
            raise RuntimeError(f"Fixed point detected: {k}")

    sorted_records = sorted([f"{k}->{v}" for k, v in shuffle.items()])
    sha = hashlib.sha256(json.dumps(sorted_records).encode()).hexdigest()

    return shuffle, sha, summary


# ============================================================================
# UNIFORM_FEATURE_MATCHED kernel
# ============================================================================

def build_uniform_kernel(feature: np.ndarray, sigma: float, radius: int,
                         support: np.ndarray | None = None):
    """Build a row-normalized RBF kernel where every allowed local edge
    has weight 1 before the row-normalization step. Uses the same
    Manhattan-radius neighborhood as ``rbf_kernel``.
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


# ============================================================================
# Box audit (per directive 5890879059 §2)
# ============================================================================

def audit_box_over_validation_cohort(repo_root: Path, qwen_box_csv: Path,
                                     image_shape=(128, 128),
                                     mlem10_root: Optional[Path] = None) -> dict:
    """Walk every Qwen box in the validation cohort; verify that
    CENTER and CENTROID boxes have width/height exactly equal to the
    Qwen (or fallback) width/height and valid half-open bounds.
    Returns a JSON-friendly audit dict.
    """
    df = pd.read_csv(qwen_box_csv, low_memory=False)
    n_total = 0
    n_zero_width_center = 0
    n_zero_height_center = 0
    n_zero_width_centroid = 0
    n_zero_height_centroid = 0
    n_invalid_center = 0
    n_invalid_centroid = 0
    mismatch_width_center = []
    mismatch_height_center = []
    mismatch_width_centroid = []
    mismatch_height_centroid = []

    for _, row in df.iterrows():
        try:
            x0, y0 = int(row["xmin"]), int(row["ymin"])
            x1, y1 = int(row["xmax"]), int(row["ymax"])
        except (ValueError, TypeError):
            continue
        if not (x1 > x0 and y1 > y0):
            continue
        bw, bh = x1 - x0, y1 - y0
        n_total += 1
        # CENTER box
        cb = build_center_box(image_shape, (x0, y0, x1, y1))
        if cb[2] - cb[0] != bw:
            n_zero_width_center += 1
            if len(mismatch_width_center) < 5:
                mismatch_width_center.append({"qwen": (x0, y0, x1, y1), "center": cb})
        if cb[3] - cb[1] != bh:
            n_zero_height_center += 1
            if len(mismatch_height_center) < 5:
                mismatch_height_center.append({"qwen": (x0, y0, x1, y1), "center": cb})
        if not (0 <= cb[0] < cb[2] <= image_shape[1] and 0 <= cb[1] < cb[3] <= image_shape[0]):
            n_invalid_center += 1
        # CENTROID box (fallback to image center if no mlem10)
        cc = build_centroid_box(np.zeros(image_shape, dtype=np.float32), image_shape,
                                (x0, y0, x1, y1))
        if cc[2] - cc[0] != bw:
            n_zero_width_centroid += 1
            if len(mismatch_width_centroid) < 5:
                mismatch_width_centroid.append({"qwen": (x0, y0, x1, y1), "centroid": cc})
        if cc[3] - cc[1] != bh:
            n_zero_height_centroid += 1
            if len(mismatch_height_centroid) < 5:
                mismatch_height_centroid.append({"qwen": (x0, y0, x1, y1), "centroid": cc})
        if not (0 <= cc[0] < cc[2] <= image_shape[1] and 0 <= cc[1] < cc[3] <= image_shape[0]):
            n_invalid_centroid += 1

    return {
        "n_qwen_boxes_audited": n_total,
        "center": {
            "n_width_mismatch": n_zero_width_center,
            "n_height_mismatch": n_zero_height_center,
            "n_invalid_bounds": n_invalid_center,
            "mismatches_first5": mismatch_width_center + mismatch_height_center,
        },
        "centroid": {
            "n_width_mismatch": n_zero_width_centroid,
            "n_height_mismatch": n_zero_height_centroid,
            "n_invalid_bounds": n_invalid_centroid,
            "mismatches_first5": mismatch_width_centroid + mismatch_height_centroid,
        },
    }