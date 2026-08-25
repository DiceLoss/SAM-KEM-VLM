#!/usr/bin/env python3
"""CPU-only iterative reconstruction algorithms for the SAM-KEM pipeline.

Provides:
- MLEM (Maximum Likelihood Expectation Maximization)
- OSEM (Ordered Subsets EM; default 10 subsets so 180/10=18 is integer)
- KEM (Kernel EM) with RBF kernel + Manhattan neighborhood radius
- Bowsher (penalized EM with nearest-neighbour weights)
- SAM-KEM (KEM with a SAM-prior derived kernel)

All algorithms operate on numpy float32/float64 arrays and use a
sparse-matrix forward operator built from the noise-free projection.

CANONICAL CONFIGURATION (Phase 5):
- Forward operator: `discrete_radon.build_pair(size=128, n_angles=180)`.
- MLEM iterations: 10.
- OSEM subsets: 10 (180/10=18 is integer; legacy 8 was tied to the
  128-angle BW system matrix and is forbidden by the validator).
- The `skimage.transform.radon / iradon` fallback has been removed; the
  precise discrete Radon transform is the only operator used for
  simulation, MLEM, KEM, and Poisson deviance.

This module deliberately avoids torch_radon (which has platform-specific
build issues) and runs end-to-end on CPU.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
from scipy.ndimage import zoom
from scipy.sparse import csr_matrix


SIM_SIZE = 128
N_ANGLES = 180  # canonical Phase 5 default
OSEM_SUBSETS = 10  # canonical Phase 5 default; 180/10=18 is integer
ANGLES = np.linspace(0.0, np.pi, N_ANGLES, endpoint=False)


def build_radon_pair(img_shape: tuple[int, int] = (SIM_SIZE, SIM_SIZE),
                     n_angles: int = N_ANGLES) -> tuple[Callable, Callable, Callable]:
    """Return (forward, backward, fbp) functions for the canonical system matrix.

    The forward function takes a (H, W) image and returns the sinogram of
    shape (n_angles, H). The backward function takes a sinogram and
    returns the exact adjoint (A^T y) back-projection. The fbp function
    returns the filtered back-projection (Shepp-Logan ramp filter) for
    initialization.

    Uses the precise discrete Radon transform from `discrete_radon.py`
    (sparse line-intersection matrix, exact adjoint) so that the system
    matrix is consistent with the analytical parallel-beam geometry.
    The legacy `skimage.transform.radon / iradon` fallback has been
    removed (Phase 5); see validate_v3.py.
    """
    import sys as _sys
    _sys.path.insert(0, str(Path(__file__).parent))
    from discrete_radon import build_pair as _build_discrete
    discrete_forward, discrete_backward, discrete_fbp = _build_discrete(
        size=img_shape[0], n_angles=n_angles
    )

    def forward(img: np.ndarray) -> np.ndarray:
        img = np.asarray(img, dtype=np.float64)
        if img.shape != img_shape:
            zy = img_shape[0] / img.shape[0]
            zx = img_shape[1] / img.shape[1]
            img = zoom(img, (zy, zx), order=1)
        return discrete_forward(img)

    def backward(sino: np.ndarray) -> np.ndarray:
        return discrete_backward(sino)

    def fbp(sino: np.ndarray) -> np.ndarray:
        return discrete_fbp(sino)

    return forward, backward, fbp


def normalize01(arr: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    a = np.asarray(arr, dtype=np.float32)
    lo, hi = float(a.min()), float(a.max())
    if hi <= lo:
        return np.zeros_like(a)
    return ((a - lo) / (hi - lo + eps)).clip(0.0, 1.0).astype(np.float32)


# ---------------------------------------------------------------- MLEM


def mlem(sinogram: np.ndarray, *, iterations: int,
         init: np.ndarray | None = None,
         save_iters: tuple[int, ...] = (),
         forward: Callable | None = None,
         backward: Callable | None = None,
         normalize_to_unit_max: bool = True,
         mul_factor: np.ndarray | None = None,
         pet_noise: np.ndarray | None = None,
         tv_beta: float = 0.0,
         tv_iter: int = 5,
         damping: float = 1.0) -> dict[int, np.ndarray]:
    """Vanilla MLEM with non-negativity and optional unit-max rescaling.

    The legacy `EM.py::MLEMorKEM` uses the noise model

        sinogram_eval = mul_factor * proj + pet_noise
        ratio = sinogram / sinogram_eval
        backproj  = K^T @ proj_backward(mul_factor * ratio, radon)
        img = img / sensitivity * backproj

    so MLEM *divides* the update by the gain `mul_factor` (scanner-gain
    correction) and adds back the `pet_noise` (scatter/random background)
    to the expected counts. When both are None, this reduces to the
    standard Poisson MLEM (clean sinogram, no gain / scatter correction).

    Returns a dict mapping each saved iteration to its reconstruction.
    Iteration 0 is the initial image (uniform ones if init is None).
    """
    if forward is None or backward is None:
        _forward, _backward, _fbp = build_radon_pair(); forward, backward = _forward, _backward
    img = np.ones((SIM_SIZE, SIM_SIZE), dtype=np.float64) if init is None else np.asarray(init, dtype=np.float64)
    img = np.clip(img, 1e-6, None)
    if mul_factor is None:
        mul_factor = np.ones_like(sinogram)
    if pet_noise is None:
        pet_noise = np.zeros_like(sinogram)
    sens = backward(mul_factor)  # gain-weighted sensitivity
    sens = np.clip(sens, 1e-6, None)
    out: dict[int, np.ndarray] = {0: img.astype(np.float32)}
    eps = 1e-6
    for it in range(1, iterations + 1):
        proj = forward(img)
        # Predicted counts: gain * projection + scatter
        sinogram_eval = mul_factor * proj + pet_noise
        ratio = sinogram / (sinogram_eval + eps * float(sinogram.mean()))
        # Back-projection with gain correction
        update = backward(mul_factor * ratio)
        # Apply damped multiplicative update
        img = img * (update / sens) ** damping
        img = np.clip(img, eps, None)
        if normalize_to_unit_max:
            m = float(img.max())
            if m > 1e-6:
                img = img / m
        if it in save_iters:
            out[it] = img.astype(np.float32)
    return out


# ----------------------------------------------------------------- OSEM


def osem(sinogram: np.ndarray, *, iterations: int, subsets: int = OSEM_SUBSETS,
         save_iters: tuple[int, ...] = (),
         init: np.ndarray | None = None,
         forward: Callable | None = None,
         backward: Callable | None = None,
         normalize_to_unit_max: bool = True,
         mul_factor: np.ndarray | None = None,
         pet_noise: np.ndarray | None = None) -> dict[int, np.ndarray]:
    """Ordered-Subsets EM with `subsets` angular subsets per iteration.

    `normalize_to_unit_max=True` rescales the image after each iteration so
    the reconstruction stays in a sane [0, 1]-ish range for low-count
    Poisson data.
    """
    if forward is None or backward is None:
        _forward, _backward, _fbp = build_radon_pair(); forward, backward = _forward, _backward
    img = np.ones((SIM_SIZE, SIM_SIZE), dtype=np.float64) if init is None else np.asarray(init, dtype=np.float64)
    img = np.clip(img, 1e-6, None)
    n_angles = sinogram.shape[0]
    angles_per_subset = n_angles // subsets
    if mul_factor is None:
        mul_factor = np.ones_like(sinogram)
    if pet_noise is None:
        pet_noise = np.zeros_like(sinogram)
    sens_full = backward(mul_factor)
    sens_full = np.clip(sens_full, 1e-6, None)
    sens_subset = sens_full / subsets
    out: dict[int, np.ndarray] = {0: img.astype(np.float32)}
    eps = 1e-6
    for it in range(1, iterations + 1):
        order = np.arange(subsets)
        rng = np.random.default_rng(0xCAFE0000 + it)
        rng.shuffle(order)
        for s in order:
            i0 = s * angles_per_subset
            i1 = (s + 1) * angles_per_subset
            sub_sino = sinogram[i0:i1]
            sub_mul = mul_factor[i0:i1]
            sub_noise = pet_noise[i0:i1]
            sub_proj = forward(img)[i0:i1]
            sinogram_eval = sub_mul * sub_proj + sub_noise
            ratio = sub_sino / (sinogram_eval + eps * float(sub_sino.mean() + 1e-9))
            full_ratio = np.zeros_like(sinogram)
            full_ratio[i0:i1] = sub_mul * ratio
            update = backward(full_ratio) / sens_subset
            img = img * update
            img = np.clip(img, eps, None)
            if normalize_to_unit_max:
                m = float(img.max())
                if m > 1e-6:
                    img = img / m
        if it in save_iters:
            out[it] = img.astype(np.float32)
    return out


# ------------------------------------------------------------------- KEM


def _build_rbf_kernel(feature: np.ndarray, sigma: float = 0.05, J: int = 2) -> csr_matrix:
    """RBF kernel restricted to a Manhattan-neighborhood radius `J`.

    `feature` is a (H, W) feature image in [0, 1] (e.g. min-max of the
    SAM prior). Each pixel is connected only to pixels within `J` in
    Manhattan distance; the affinity is exp(-|f_i - f_j|^2 / (2 sigma^2)).
    The kernel is row-normalized.
    """
    h, w = feature.shape
    n = h * w
    feat_flat = feature.reshape(-1).astype(np.float64)
    two_sigma_sq = 2.0 * sigma * sigma
    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []
    for r in range(h):
        for c in range(w):
            i = r * w + c
            for dr in range(-J, J + 1):
                for dc in range(-J, J + 1):
                    if abs(dr) + abs(dc) > J:
                        continue
                    rr = r + dr
                    cc = c + dc
                    if rr < 0 or rr >= h or cc < 0 or cc >= w:
                        continue
                    j = rr * w + cc
                    diff = feat_flat[i] - feat_flat[j]
                    w_ij = math.exp(-(diff * diff) / (two_sigma_sq + 1e-12))
                    rows.append(i)
                    cols.append(j)
                    data.append(w_ij)
    K = csr_matrix((data, (rows, cols)), shape=(n, n))
    # Row-normalize (sinkhorn-like, single iteration)
    row_sum = np.asarray(K.sum(axis=1)).reshape(-1)
    row_sum[row_sum <= 0] = 1.0
    inv = 1.0 / row_sum
    D = csr_matrix((inv, (np.arange(n), np.arange(n))), shape=(n, n))
    K = D @ K
    return K


def kem(sinogram: np.ndarray, feature: np.ndarray, *,
        iterations: int = 60, sigma: float = 0.05, J: int = 2,
        init: np.ndarray | None = None,
        save_iters: tuple[int, ...] = (),
        forward: Callable | None = None,
        backward: Callable | None = None,
        normalize_to_unit_max: bool = True,
        mul_factor: np.ndarray | None = None,
        pet_noise: np.ndarray | None = None) -> dict[int, np.ndarray]:
    """Kernel EM with a sparse RBF kernel matrix.

    Same gain / scatter correction as `mlem`; in addition the image is
    smoothed by a precomputed RBF kernel matrix after each EM step.
    """
    if forward is None or backward is None:
        _forward, _backward, _fbp = build_radon_pair(); forward, backward = _forward, _backward
    img = np.ones((SIM_SIZE, SIM_SIZE), dtype=np.float64) if init is None else np.asarray(init, dtype=np.float64)
    img = np.clip(img, 1e-6, None)
    if mul_factor is None:
        mul_factor = np.ones_like(sinogram)
    if pet_noise is None:
        pet_noise = np.zeros_like(sinogram)
    sens = backward(mul_factor)
    sens = np.clip(sens, 1e-6, None)
    K = _build_rbf_kernel(feature, sigma=sigma, J=J)
    out: dict[int, np.ndarray] = {0: img.astype(np.float32)}
    eps = 1e-6
    for it in range(1, iterations + 1):
        proj = forward(img)
        sinogram_eval = mul_factor * proj + pet_noise
        ratio = sinogram / (sinogram_eval + eps * float(sinogram.mean()))
        update = backward(mul_factor * ratio)
        img = img * (update / sens)
        img = np.clip(img, eps, None)
        # Kernel smoothing
        img = (K @ img.reshape(-1)).reshape(SIM_SIZE, SIM_SIZE)
        img = np.clip(img, eps, None)
        if normalize_to_unit_max:
            m = float(img.max())
            if m > 1e-6:
                img = img / m
        if it in save_iters:
            out[it] = img.astype(np.float32)
    return out


# ---------------------------------------------------------------- Bowsher


def _bowsher_weights(feature: np.ndarray, n_neighbours: int = 8,
                     J: int = 1) -> csr_matrix:
    """Bowsher weights: each pixel connects to the n_neighbours nearest
    pixels in feature space within a Manhattan-neighborhood radius `J`.
    """
    h, w = feature.shape
    n = h * w
    feat_flat = feature.reshape(-1).astype(np.float64)
    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []
    for r in range(h):
        for c in range(w):
            i = r * w + c
            cand = []
            for dr in range(-J, J + 1):
                for dc in range(-J, J + 1):
                    if dr == 0 and dc == 0:
                        continue
                    if abs(dr) + abs(dc) > J:
                        continue
                    rr = r + dr
                    cc = c + dc
                    if rr < 0 or rr >= h or cc < 0 or cc >= w:
                        continue
                    j = rr * w + cc
                    diff = abs(feat_flat[i] - feat_flat[j])
                    cand.append((diff, j))
            cand.sort(key=lambda t: t[0])
            for _, j in cand[:n_neighbours]:
                rows.append(i)
                cols.append(j)
                data.append(1.0 / n_neighbours)
    W = csr_matrix((data, (rows, cols)), shape=(n, n))
    return W


def bowsher(sinogram: np.ndarray, feature: np.ndarray, *,
            iterations: int = 60, n_neighbours: int = 8, J: int = 1,
            init: np.ndarray | None = None,
            save_iters: tuple[int, ...] = (),
            forward: Callable | None = None,
            backward: Callable | None = None,
            normalize_to_unit_max: bool = True) -> dict[int, np.ndarray]:
    """Bowsher-penalized EM with optional unit-max rescaling."""
    if forward is None or backward is None:
        _forward, _backward, _fbp = build_radon_pair(); forward, backward = _forward, _backward
    img = np.ones((SIM_SIZE, SIM_SIZE), dtype=np.float64) if init is None else np.asarray(init, dtype=np.float64)
    img = np.clip(img, 1e-6, None)
    sens = backward(np.ones_like(sinogram))
    sens = np.clip(sens, 1e-6, None)
    W = _bowsher_weights(feature, n_neighbours=n_neighbours, J=J)
    out: dict[int, np.ndarray] = {0: img.astype(np.float32)}
    eps = 1e-6
    for it in range(1, iterations + 1):
        proj = forward(img)
        ratio = sinogram / (proj + eps)
        update = backward(ratio) / sens
        img = img * update
        img = np.clip(img, eps, None)
        img = (W @ img.reshape(-1)).reshape(SIM_SIZE, SIM_SIZE)
        img = np.clip(img, eps, None)
        if normalize_to_unit_max:
            m = float(img.max())
            if m > 1e-6:
                img = img / m
        if it in save_iters:
            out[it] = img.astype(np.float32)
    return out


# ----------------------------------------------------- SAM-KEM (no post)


def sam_kem_no_post(sinogram: np.ndarray, sam_prior: np.ndarray, *,
                    iterations: int = 60, sigma: float = 0.05, J: int = 2,
                    init: np.ndarray | None = None,
                    save_iters: tuple[int, ...] = (),
                    forward: Callable | None = None,
                    backward: Callable | None = None,
                    normalize_to_unit_max: bool = True,
                    mul_factor: np.ndarray | None = None,
                    pet_noise: np.ndarray | None = None) -> dict[int, np.ndarray]:
    """SAM-derived KEM (no post-fusion).

    The SAM prior is min-max-normalized per image to form the kernel
    feature `phi`. The kernel is then used in the standard KEM loop.
    """
    phi = normalize01(sam_prior)
    return kem(sinogram, phi, iterations=iterations, sigma=sigma, J=J,
               init=init, save_iters=save_iters, forward=forward, backward=backward)


def post_fusion(sam_prior: np.ndarray, kem_image: np.ndarray, alpha: float = 0.3) -> np.ndarray:
    """Convex post-fusion: x_hat = alpha * z + (1 - alpha) * x_KEM.

    `sam_prior` is the raw decoder output `z` in normalized [0, 1] scale.
    `kem_image` is the KEM reconstruction in the same scale.
    """
    z = np.clip(sam_prior, 0.0, 1.0).astype(np.float32)
    x = np.clip(kem_image, 0.0, 1.0).astype(np.float32)
    return (alpha * z + (1.0 - alpha) * x).astype(np.float32)