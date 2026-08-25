#!/usr/bin/env python3
"""Preprocessing and evaluation-mask pipeline for SAM-KEM experiments.

This module is the single source of truth for the per-dataset preprocessing
contract documented in `provenance/model_provenance.md`. It contains:

1. PET2Sinogram - noise-free Radon projection of a 2D PET image.
2. apply_count_scaling - convert (noise-free projection, expected total
   counts) into a Poisson-sampled sinogram. The simulator is the one in
   `scripts/poisson_simulate.py`; this helper just routes through it.
3. normalize01 - clip+min-max normalization to [0,1].
4. build_evaluation_mask - body mask from MR (BrainWeb) or CT (HECKTOR).
5. load_dataset_case - extract a single canonical 2D central-axial slice
   per subject, with the paired anatomy channel.
6. write_case_arrays - persist the canonical per-case array layout to disk.

Image sizes:
- Simulation grid: 128x128 (PET), 128x128 (anatomy)
- Model input: 256x256 (SAM input, resized via nearest-neighbour from
  128x128)
- Kernel construction: 128x128
- Evaluation: 128x128, with body mask applied

HECKTOR tumor masks are NEVER read.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

try:
    from skimage.transform import radon, iradon
except ImportError:  # pragma: no cover - handled by environment dependency
    radon = None
    iradon = None

from poisson_simulate import simulate, POISSON_SEED_BASE


SIM_SIZE = 128
SAM_INPUT_SIZE = 256
ANGLES = np.linspace(0.0, np.pi, SIM_SIZE, endpoint=False)
CANONICAL_COUNT_LEVELS = (100_000, 200_000, 500_000, 1_000_000)


@dataclass
class DatasetMeta:
    dataset: str
    subject_id: str
    pet_2d: np.ndarray
    anatomy_2d: np.ndarray
    evaluation_mask: np.ndarray
    anatomy_name: str           # "MR" for brainweb, "CT" for hecktor
    scale_C: float             # fixed physical-scale constant for x_norm


def normalize01(arr: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    a = np.asarray(arr, dtype=np.float32)
    lo, hi = float(a.min()), float(a.max())
    if hi <= lo:
        return np.zeros_like(a, dtype=np.float32)
    return ((a - lo) / (hi - lo + eps)).clip(0.0, 1.0)


def apply_evaluation_mask(arr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Zero out background voxels; metrics are computed inside the mask."""
    out = np.asarray(arr, dtype=np.float32).copy()
    out[mask <= 0] = 0.0
    return out


def build_body_mask_from_anatomy(anatomy_2d: np.ndarray, threshold: float | None = None) -> np.ndarray:
    """Body mask = pixels above Otsu-style threshold of the anatomy channel.

    For BrainWeb MR and HECKTOR CT (after HU clipping), a simple
    percentile-based threshold is sufficient. The threshold defaults to
    the 5th percentile of the non-zero anatomy voxels (background-aware).
    """
    a = np.asarray(anatomy_2d, dtype=np.float32)
    nz = a[a > 0]
    if nz.size == 0:
        return np.ones_like(a, dtype=np.float32)
    if threshold is None:
        threshold = float(np.percentile(nz, 5.0))
    mask = (a >= threshold).astype(np.float32)
    # morphological closing via simple 3x3 max filter to fill small holes
    from scipy.ndimage import binary_closing
    mask = binary_closing(mask > 0, structure=np.ones((3, 3))).astype(np.float32)
    return mask


def pet2sinogram(pet_2d: np.ndarray) -> np.ndarray:
    """Forward Radon projection via scikit-image."""
    if radon is None:
        raise ImportError("skimage.transform.radon unavailable")
    img = np.asarray(pet_2d, dtype=np.float64)
    if img.shape != (SIM_SIZE, SIM_SIZE):
        # nearest-neighbour resize to SIM_SIZE
        from scipy.ndimage import zoom
        zoom_factors = (SIM_SIZE / img.shape[0], SIM_SIZE / img.shape[1])
        img = zoom(img, zoom_factors, order=0)
    sinogram = radon(img, theta=ANGLES, circle=False)
    return sinogram.astype(np.float64)


def apply_count_scaling(p: np.ndarray, n_true: int, *,
                        realization: int = 0) -> dict[str, Any]:
    """Scale the noise-free projection to N_true counts and sample Poisson."""
    seed = POISSON_SEED_BASE + int(realization)
    return simulate(p, n_true, seed=seed)


def resize_for_sam(img_2d: np.ndarray) -> np.ndarray:
    """Nearest-neighbour resize 128x128 -> 256x256 for SAM input."""
    from scipy.ndimage import zoom
    return zoom(np.asarray(img_2d, dtype=np.float32), 2.0, order=0).astype(np.float32)


# ---------------- Per-dataset loading -----------------------------------


def _central_axial_slice(volume: np.ndarray) -> np.ndarray:
    """Pick the central axial slice from a 3D volume (D, H, W) or (H, W, D).

    For some HECKTOR patients the geometric center is empty (zero PET); we
    fall back to the slice with the largest mean intensity in a +-5-slice
    window around the center.
    """
    v = np.asarray(volume)
    axial = int(np.argmin(v.shape))
    center = v.shape[axial] // 2
    slicer_center = [slice(None)] * v.ndim
    slicer_center[axial] = center
    central = v[tuple(slicer_center)]
    if float(np.asarray(central).max()) > 0:
        return np.asarray(central, dtype=np.float32)
    # Search a wider +-20-slice window around the center for the largest-mean slice
    best_idx, best_sum = center, -1.0
    for delta in range(-20, 21):
        idx = center + delta
        if idx < 0 or idx >= v.shape[axial]:
            continue
        s = [slice(None)] * v.ndim
        s[axial] = idx
        slc = v[tuple(s)]
        m = float(np.asarray(slc).sum())
        if m > best_sum:
            best_sum = m
            best_idx = idx
    slicer_best = [slice(None)] * v.ndim
    slicer_best[axial] = best_idx
    return np.asarray(v[tuple(slicer_best)], dtype=np.float32)


def _resize_2d(img: np.ndarray, size: int = SIM_SIZE) -> np.ndarray:
    from scipy.ndimage import zoom
    s = np.asarray(img, dtype=np.float32)
    if s.shape == (size, size):
        return s
    zoom_factors = (size / s.shape[0], size / s.shape[1])
    return zoom(s, zoom_factors, order=1).astype(np.float32)


def load_brainweb_case(brainweb_npy: Path, subject_id: str) -> DatasetMeta:
    """Load a BrainWeb source model from the raw 3D phantom volumes.

    The directory `D:/MIAProject/brainweb/数据/brainweb_petmr/<subject_id>/`
    contains `t1.nii.gz` (MR) and `sim_0/true_pet.nii.gz` (reference PET).
    We pick the central axial slice of the 3D volume and resize to 128x128.
    """
    try:
        import nibabel as nib  # type: ignore
    except ImportError as exc:
        raise ImportError("nibabel required to load BrainWeb nii.gz volumes") from exc
    petmr_root = brainweb_npy.parent / "数据" / "brainweb_petmr"
    subj_dir = petmr_root / subject_id
    if not subj_dir.is_dir():
        raise KeyError(f"BrainWeb subject {subject_id} not found at {subj_dir}")
    pet_path = subj_dir / "sim_0" / "true_pet.nii.gz"
    t1_path = subj_dir / "t1.nii.gz"
    pet_vol = nib.load(str(pet_path)).get_fdata().astype(np.float32)
    mr_vol = nib.load(str(t1_path)).get_fdata().astype(np.float32)
    # central axial slice along the smallest axis
    pet_2d = _resize_2d(_central_axial_slice(pet_vol), SIM_SIZE)
    mr_2d = _resize_2d(_central_axial_slice(mr_vol), SIM_SIZE)
    pet_2d = normalize01(pet_2d)
    mr_2d = normalize01(mr_2d)
    mask = build_body_mask_from_anatomy(mr_2d)
    return DatasetMeta(
        dataset="brainweb",
        subject_id=subject_id,
        pet_2d=pet_2d,
        anatomy_2d=mr_2d,
        evaluation_mask=mask,
        anatomy_name="MR",
        scale_C=1.0,
    )


def load_hecktor_case(hecktor_npz: Path, subject_id: str) -> DatasetMeta:
    """Load a HECKTOR patient from the npz.

    The npz contains:
            input.npy  shape (224, 144, 144, 144, 2)
                    channel 0 = PET in HU-like range (NOT used as reference)
                    channel 1 = normalized CT [0, 1]-ish
            target.npy shape (224, 144, 144, 144)
                    the normalized [0, 1] reference PET (full-count clinical reconstruction)

    Patient indices are 1-based; subject_id is the canonical original
    challenge ID. We use target.npy as the reference PET and channel 1 as
    the anatomy (CT) channel for the CT-KEM and CT-Bowsher baselines.
    """
    pid = int(subject_id) - 1
    if pid < 0 or pid >= 224:
        raise KeyError(f"HECKTOR subject_id {subject_id} out of [1, 224]")
    arr = np.load(hecktor_npz, mmap_mode="r")
    ct = arr["input"][pid, ..., 1]       # normalized CT
    target = arr["target"][pid]          # reference PET in [0, 1]
    ct_2d = _resize_2d(_central_axial_slice(ct), SIM_SIZE)
    target_2d = _resize_2d(_central_axial_slice(target), SIM_SIZE)
    ct_2d = np.clip(ct_2d, 0.0, 1.0).astype(np.float32)
    target_2d = np.clip(target_2d, 0.0, 1.0).astype(np.float32)
    mask = build_body_mask_from_anatomy(ct_2d)
    return DatasetMeta(
        dataset="hecktor",
        subject_id=subject_id,
        pet_2d=target_2d,
        anatomy_2d=ct_2d,
        evaluation_mask=mask,
        anatomy_name="CT",
        scale_C=1.0,
    )


def write_case_arrays(out_root: Path, meta: DatasetMeta, *,
                      count: int, realization: int,
                      noise_free_projection: np.ndarray,
                      sinogram: np.ndarray) -> dict[str, Path]:
    """Write the canonical per-case numpy layout.

    Returns a dict of relative POSIX paths suitable for embedding in
    `manifests/case_manifest.csv`.
    """
    case_dir = out_root / "arrays" / meta.dataset / meta.subject_id / "axial_center" / f"r{realization}"
    case_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    arrays = {
        "target.npy": meta.pet_2d.astype(np.float32),
        "noise_free_projection.npy": noise_free_projection.astype(np.float32),
        "sinogram.npy": sinogram.astype(np.int64),
        "anatomy.npy": meta.anatomy_2d.astype(np.float32),
        "evaluation_mask.npy": meta.evaluation_mask.astype(np.float32),
        "qwen_input.npy": normalize01(meta.pet_2d).astype(np.float32),
    }
    for name, arr in arrays.items():
        p = case_dir / name
        np.save(p, arr)
        paths[name] = p.relative_to(out_root)
    return paths