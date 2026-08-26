"""Canonical idealized parallel-beam PET projection operator.

This is a pixel-driven discrete projector on an ``n x n`` circular field of
view.  At every angle, each pixel is linearly distributed between the two
nearest detector bins.  Consequently, projection mass is conserved for an
image supported inside the inscribed circle.  The backprojector is the exact
sparse-matrix transpose; it is not filtered backprojection.

This operator deliberately does not model attenuation, scatter, randoms,
detector normalization, PSF, TOF, dead time, or a scanner-specific geometry.
It therefore supports an idealized low-count study, not a claim of raw-scanner
or injected-dose simulation.
"""
from __future__ import annotations

from typing import Callable, Tuple

import numpy as np
from scipy.sparse import coo_matrix, csr_matrix


def _radon_lines(n: int, n_angles: int) -> csr_matrix:
    """Return a mass-preserving pixel-driven system matrix.

    Pixel centres and detector centres use pixel units.  A projected pixel
    coordinate is split by linear interpolation between adjacent bins.  No
    per-ray normalization is applied: such normalization would amplify short
    edge rays and change the physical meaning of the system matrix.
    """
    if n < 2 or n_angles < 1:
        raise ValueError("n must be >=2 and n_angles must be >=1")
    centre = (n - 1) / 2.0
    coords = np.arange(n, dtype=np.float64) - centre
    xx, yy = np.meshgrid(coords, coords, indexing="xy")
    pixel = np.arange(n * n, dtype=np.int64).reshape(n, n)
    circular = (xx * xx + yy * yy) <= centre * centre

    rows: list[np.ndarray] = []
    cols: list[np.ndarray] = []
    values: list[np.ndarray] = []
    angles = np.linspace(0.0, np.pi, n_angles, endpoint=False)
    for angle_index, theta in enumerate(angles):
        detector_coordinate = xx * np.cos(theta) + yy * np.sin(theta) + centre
        lower = np.floor(detector_coordinate).astype(np.int64)
        upper_weight = detector_coordinate - lower
        lower_weight = 1.0 - upper_weight

        for detector_bin, weight in ((lower, lower_weight), (lower + 1, upper_weight)):
            valid = circular & (detector_bin >= 0) & (detector_bin < n) & (weight > 0)
            rows.append(angle_index * n + detector_bin[valid])
            cols.append(pixel[valid])
            values.append(weight[valid])

    matrix = coo_matrix(
        (np.concatenate(values), (np.concatenate(rows), np.concatenate(cols))),
        shape=(n_angles * n, n * n),
        dtype=np.float64,
    ).tocsr()
    matrix.sum_duplicates()
    return matrix


def build_pair(size: int = 128, n_angles: int = 180) -> Tuple[Callable, Callable, Callable]:
    """Return the canonical ``(forward, adjoint, fbp)`` callables."""
    matrix = _radon_lines(size, n_angles)
    n = size

    def forward(image):
        image_array = np.asarray(image, dtype=np.float64)
        if image_array.shape != (n, n):
            raise ValueError(f"expected image shape {(n, n)}, got {image_array.shape}")
        return (matrix @ image_array.reshape(-1)).reshape(n_angles, n)

    def backward(sinogram):
        sinogram_array = np.asarray(sinogram, dtype=np.float64)
        if sinogram_array.shape != (n_angles, n):
            raise ValueError(
                f"expected sinogram shape {(n_angles, n)}, got {sinogram_array.shape}"
            )
        return (matrix.T @ sinogram_array.reshape(-1)).reshape(n, n)

    def fbp(sinogram):
        """Approximate ramp-filtered initialization using the matched adjoint."""
        sinogram_array = np.asarray(sinogram, dtype=np.float64)
        if sinogram_array.shape != (n_angles, n):
            raise ValueError(
                f"expected sinogram shape {(n_angles, n)}, got {sinogram_array.shape}"
            )
        frequencies = np.fft.rfftfreq(n)
        filtered = np.fft.irfft(
            np.fft.rfft(sinogram_array, axis=1) * np.abs(frequencies)[None, :],
            n=n,
            axis=1,
        )
        return (np.pi / n_angles) * (matrix.T @ filtered.reshape(-1)).reshape(n, n)

    return forward, backward, fbp


if __name__ == "__main__":
    forward, backward, _ = build_pair(64, 90)
    rng = np.random.default_rng(0)
    image = rng.random((64, 64))
    projection_test = rng.random((90, 64))
    left = float(np.sum(forward(image) * projection_test))
    right = float(np.sum(image * backward(projection_test)))
    print(f"adjoint absolute error: {abs(left - right):.3e}")
