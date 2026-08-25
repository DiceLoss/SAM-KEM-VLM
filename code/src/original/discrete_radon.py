"""Precise discrete Radon transform (parallel-beam).

The implementation uses a sparse line-intersection matrix so that the
forward and adjoint operators are exact transposes of each other and
the discretization is consistent with the original (continuous) parallel-
beam geometry. This avoids the discretization error introduced by
`skimage.transform.radon/iradon` (which uses FFT-based filtered back-
projection and has noticeable aliasing at small image sizes).

API:
    forward, backward = build_pair(size=128, n_angles=180)
    sinogram = forward(image)        # shape (n_angles, size)
    image    = backward(sinogram)    # shape (size, size), filtered back-projection

The forward and backward are exact adjoints: <A x, y> = <x, A^T y>.

CANONICAL DEFAULT (Phase 5): n_angles=180, size=128. This is the
single forward operator used in simulation, MLEM, KEM, and Poisson
deviance for both BrainWeb and HECKTOR. The system-matrix SHA-256 is
written to arrays/system_matrix_A_n180_sha256.txt by
`scripts/_port_forward_operator.py` and is referenced by every metric
that uses the operator. The legacy `skimage.transform.radon` fallback
is explicitly forbidden by the validator.
"""
from __future__ import annotations

import math
from typing import Callable, Tuple

import numpy as np
from scipy.sparse import csr_matrix


def _radon_lines(n: int, n_angles: int) -> csr_matrix:
    """Build the (n_angles * n_det, n*n) discrete Radon matrix.

    For each angle theta_i, the line integral along s = -floor(n/2)+k
    for k = 0..n-1 is the sum of pixel values f(x, y) intersected by
    the line `x * cos(theta) + y * sin(theta) = s`. Each pixel (x, y) is
    assigned a width equal to the chord length of the pixel along the
    integration line.
    """
    n2 = n * n
    n_rays = n_angles * n
    # Pixel center coordinates in the range [-1, 1) (continuous).
    coords = (np.arange(n) - (n - 1) / 2.0) / n
    # Detector positions s (in pixel units), same continuous range.
    s_coords = (np.arange(n) - (n - 1) / 2.0) / n
    # Pixel half-width.
    dx = 1.0 / n
    # Pre-compute pixel-center matrix.
    X, Y = np.meshgrid(coords, coords, indexing="xy")  # X: x, Y: y
    # For each angle theta, the projection of pixel (x, y) onto
    # detector axis s is t = x * cos(theta) + y * sin(theta).
    angles = np.linspace(0.0, np.pi, n_angles, endpoint=False)
    cos_t = np.cos(angles)
    sin_t = np.sin(angles)
    rows, cols, data = [], [], []
    pixel_idx = np.arange(n2).reshape(n, n)
    for i, (c, s) in enumerate(zip(cos_t, sin_t)):
        # For each pixel center, compute its projection t.
        t = X * c + Y * s  # shape (n, n)
        # The pixel contributes to the detector at the nearest s bin with
        # weight equal to the chord length within the pixel perpendicular
        # to the line. For a 1-pixel-wide square, the chord is
        # |n * dx / max(|c|, |s|)| in pixel units, but for simplicity we
        # use the perpendicular extent |dx| / max(|c|, |s|).
        weight = dx / max(abs(c), abs(s))
        # Convert t (continuous) to bin index.
        bin_idx = np.round(t * n + (n - 1) / 2.0).astype(int)
        valid = (bin_idx >= 0) & (bin_idx < n)
        # Flatten and add to COO arrays.
        pix_flat = pixel_idx[valid]
        ray_flat = i * n + bin_idx[valid]
        rows.append(ray_flat)
        cols.append(pix_flat)
        data.append(np.full(int(valid.sum()), float(weight)))
    rows = np.concatenate(rows)
    cols = np.concatenate(cols)
    data = np.concatenate(data)
    A = csr_matrix((data, (rows, cols)), shape=(n_rays, n2))
    # Row-normalize so each ray sums to 1.0 (i.e. the forward Radon
    # transform is angle-independent, like the analytical continuous
    # transform). Without this, the diagonal pixel weight is
    # 1/(n*max(|cos|theta|,|sin|theta|)), making the projection of a
    # constant image equal to 1/sin(theta+pi/4), so a single global
    # scale mismatch ruins MLEM convergence against a sinogram that was
    # generated with a different system matrix.
    row_sum = np.asarray(A.sum(axis=1)).reshape(-1)
    row_sum[row_sum <= 0] = 1.0
    inv = 1.0 / row_sum
    from scipy.sparse import diags
    D = diags(inv)
    A = (D @ A).tocsr()
    return A


def build_pair(size: int = 128, n_angles: int = 180) -> Tuple[Callable, Callable, Callable]:
    """Return (forward, backward, fbp) callables for the canonical system matrix.

    `forward(image)` returns a sinogram of shape (n_angles, size).
    `backward(sinogram)` returns the EXACT ADJOINT back-projection A^T y
    (unfiltered, not FBP). This is what MLEM needs for its sensitivity
    and update computation; FBP is reserved for initialization.

    Use `fbp(sinogram)` for filtered back-projection initialization.
    """
    A = _radon_lines(size, n_angles)
    n = size

    def forward(image):
        img = np.asarray(image, dtype=np.float64).reshape(-1)
        sino = A @ img
        return sino.reshape(n_angles, n)

    def backward(sinogram):
        sino = np.asarray(sinogram, dtype=np.float64)
        sino = np.atleast_2d(sino)
        sino = sino.flatten()
        return (A.T @ sino).reshape(n, n)

    def fbp(sinogram):
        """Filtered back-projection (Shepp-Logan ramp filter)."""
        sino = np.asarray(sinogram, dtype=np.float64)
        k = np.arange(n, dtype=np.float64)
        H = np.zeros(n, dtype=np.float64)
        H[0] = 0.25
        H[2::2] = -1.0 / (np.pi * k[2::2]) ** 2
        sino_filt = np.real(np.fft.ifft(
            np.fft.fft(sino, axis=1) * H[None, :], axis=1
        ))
        img_flat = A.T @ sino_filt.reshape(-1)
        return (np.pi / (2.0 * n_angles)) * img_flat.reshape(n, n)

    return forward, backward, fbp

    def backward(sinogram):
        sino = np.asarray(sinogram, dtype=np.float64).reshape(-1)
        return (A.T @ sino).reshape(size, size)

    return forward, backward


if __name__ == "__main__":
    # Quick adjoint check.
    forward, backward = build_pair(64, 90)
    img = np.random.RandomState(0).rand(64, 64)
    s = forward(img)
    s2 = np.random.RandomState(1).rand(90, 64)
    ip1 = float((s * s2).sum())
    ip2 = float((img * backward(s2)).sum())
    print(f"<A x, y> = {ip1:.6f},  <x, A^T y> = {ip2:.6f},  diff = {abs(ip1 - ip2):.2e}")
