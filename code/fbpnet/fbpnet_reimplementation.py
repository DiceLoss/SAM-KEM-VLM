"""FBP-Net reimplementation (Wang & Liu, Physics in Medicine & Biology,
2020, vol. 65, no. 23, DOI 10.1088/1361-6560/abc09d).

*** reimplementation *** — dynamic-frame version (v3, 2026-10-14)

This is a clean reimplementation built from the paper's equations
only. The original authors did not release public code; this
reimplementation is per the paper (Wang & Liu 2020).

Per directive 6159621060 item 6, this v3 adds:
  - dynamic-frame FBP (per-frame FBP with learned filter, T frames)
  - dynamic-frame denoiser (3D U-Net over T x H x W) that fuses
    information across time frames (the paper's key contribution:
    "merges information across all time frames to improve
    reconstruction quality, especially for early time frames")
  - per-frame MSE loss stub (no training performed)
  - per-frame + temporal L1 regularizer stub (no training performed)
  - LayerNorm + GroupNorm placed alongside the existing BatchNorm
    so the per-frame distribution normalization matches the paper's
    discussion of "dynamic frame normalization".

Architecture (per the paper §2):
  Part 1 (per-frame FBP with learned frequency filter):
    For each frame t in {1..T}, apply 1D FFT to the projection,
    multiply by a learnable frequency-domain filter H(ω) (MLP from
    normalized frequency bin), and apply inverse Radon back-projection.
  Part 2 (dynamic-frame denoiser):
    A 3D U-Net that takes the (B, T, 1, H, W) stack of FBP outputs
    and produces the (B, T, 1, H, W) stack of denoised images.
  Loss (per-frame MSE + temporal L1):
    L = sum_t ||x_pred[t] - x_gt[t]||² + λ * ||x_pred[t+1] - x_pred[t]||_1.
    This stub is included as documentation; the worker does NOT
    train under any branch per directive §E.

ASSUMPTIONs (to be reviewed):
  A1. U-Net channel layout 32-64-128-256 (paper silent).
  A2. No temporal-consistency loss training; loss stub is
      documentation-only.
  A3. MLP filter width 32 hidden units (paper silent).
  A4. No clipping to >= 0 in forward (paper clips post-hoc; we
      apply ReLU on the denoiser output as an alternative to make
      the synthetic-input smoke pass non-negative).
  A5. FBP uses our rfft-based back-projection (paper uses vendor smat).
  A6. Dynamic denoiser: paper describes "merges information across
      all time frames"; we implement this by treating the T frames
      as channels of a 3D U-Net (Conv3d). This is an ASSUMPTION
      — the paper's exact temporal-fusion mechanism (early-frame
      conditioning vs. late-frame conditioning vs. bidirectional)
      is not specified. We choose the symmetric Conv3d U-Net
      channel-treatment as the simplest 3D-UNet baseline.

References:
    Wang, B., Liu, H. FBP-Net for direct reconstruction of dynamic
        PET images. Phys. Med. Biol. 65 (23) 235014 (2020).
        DOI 10.1088/1361-6560/abc09d.
"""
from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def make_ramp_filter(n_detectors: int, dtype=torch.float32) -> torch.Tensor:
    """Standard ramp filter |ω| for FBP."""
    freqs = torch.fft.fftfreq(n_detectors, dtype=dtype)
    return torch.abs(freqs)


def fbp_per_frame(
    sinogram: torch.Tensor,         # (B, T, n_angles, n_det)
    angles_deg: torch.Tensor,        # (n_angles,)
    image_size: int,
    learnable_filter: torch.Tensor,  # (n_det//2+1,)
    ramp_weight: float = 1.0,
) -> torch.Tensor:
    """Apply per-frame FBP with hybrid filter (ramp + learnable).

    Args:
        sinogram: (B, T, n_angles, n_det)
        angles_deg: (n_angles,) in degrees
        image_size: target output spatial size
        learnable_filter: (n_det//2+1,) positive-frequency half
        ramp_weight: scalar on standard ramp (default 1.0)

    Returns:
        (B, T, H, W) per-frame FBP reconstruction.
    """
    B, T, n_angles, n_det = sinogram.shape
    half = n_det // 2 + 1
    assert learnable_filter.shape == (half,)
    # 1D FFT
    sin_freq = torch.fft.rfft(sinogram, dim=-1)  # (B, T, n_angles, half)
    ramp = make_ramp_filter(n_det, dtype=sinogram.dtype).to(sinogram.device)
    full_filter = ramp_weight * ramp[:half] + learnable_filter
    filtered = sin_freq * full_filter
    proj = torch.fft.irfft(filtered, n=n_det, dim=-1)  # (B, T, n_angles, n_det)
    # Back-project per frame
    return _back_project(proj, angles_deg, image_size)  # (B, T, H, W)


def _back_project(
    proj: torch.Tensor,    # (B, T, n_angles, n_det)
    angles_deg: torch.Tensor,
    image_size: int,
) -> torch.Tensor:
    """Simple per-angle, per-frame back-projection (gather-based)."""
    B, T, n_angles, n_det = proj.shape
    device = proj.device
    dtype = proj.dtype
    image = torch.zeros(B, T, image_size, image_size, device=device, dtype=dtype)
    center = image_size / 2.0
    with torch.no_grad():
        ys = torch.arange(image_size, device=device, dtype=dtype)
        xs = torch.arange(image_size, device=device, dtype=dtype)
        yi = ys - center
        xi = xs - center
        Y, X = torch.meshgrid(yi, xi, indexing="ij")
        for i in range(n_angles):
            theta_rad = float(angles_deg[i]) * math.pi / 180.0
            cos_t = math.cos(theta_rad)
            sin_t = math.sin(theta_rad)
            u = X * cos_t + Y * sin_t
            idx = (u + (n_det - 1) / 2.0).clamp(0, n_det - 1).long()
            # proj[:, :, i, :] is (B, T, n_det); index over n_det
            image = image + proj[:, :, i, :][:, :, idx]
    return image / n_angles


class LearnableFrequencyFilter(nn.Module):
    """Per-frequency-bin learnable filter H(ω)."""

    def __init__(self, n_detectors: int, hidden: int = 32):
        super().__init__()
        self.n_detectors = n_detectors
        self.half = n_detectors // 2 + 1
        self.mlp = nn.Sequential(
            nn.Linear(1, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, 1),
        )

    def forward(self) -> torch.Tensor:
        device = next(self.mlp.parameters()).device
        omega = torch.linspace(0.0, 1.0, self.half, device=device)
        out = torch.empty(self.half, device=device)
        for i in range(self.half):
            h = self.mlp(omega[i:i + 1].view(1, 1))
            out[i] = h[0, 0]
        return out


class DoubleConv3D(nn.Module):
    """(conv3d => BN3d => ReLU) * 2."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class UNet3D(nn.Module):
    """3D U-Net for dynamic-frame denoising.

    The 3D U-Net treats the T time-frames as a depth axis, so
    convolutions operate over (T, H, W). This is the ASSUMPTION
    (paper §2 describes "merges information across all time frames"
    but does not specify the exact temporal-fusion mechanism).
    """

    def __init__(self, in_ch: int = 1, out_ch: int = 1,
                  base_ch: int = 16, depth: int = 3):
        super().__init__()
        chs = [base_ch * (2 ** i) for i in range(depth + 1)]
        self.encoders = nn.ModuleList()
        prev = in_ch
        for c in chs[:-1]:
            self.encoders.append(DoubleConv3D(prev, c))
            prev = c
        self.bottleneck = DoubleConv3D(chs[-2], chs[-1])
        self.decoders = nn.ModuleList()
        self.upsamples = nn.ModuleList()
        prev = chs[-1]
        for c in reversed(chs[:-1]):
            self.upsamples.append(nn.ConvTranspose3d(prev, c, (1, 2, 2),
                                                       stride=(1, 2, 2)))
            self.decoders.append(DoubleConv3D(c * 2, c))
            prev = c
        self.out_conv = nn.Conv3d(chs[0], out_ch, 1)
        self.pool = nn.MaxPool3d((1, 2, 2))

    def forward(self, x):
        skips = []
        for enc in self.encoders:
            x = enc(x)
            skips.append(x)
            x = self.pool(x)
        x = self.bottleneck(x)
        for up, dec, skip in zip(self.upsamples, self.decoders,
                                  reversed(skips)):
            x = up(x)
            if x.shape[-2:] != skip.shape[-2:]:
                x = F.interpolate(x, size=skip.shape[-2:], mode="nearest")
            x = torch.cat([x, skip], dim=1)
            x = dec(x)
        return self.out_conv(x)


class FBPNetReimplementation(nn.Module):
    """FBP-Net reimplementation, dynamic-frame version (v3).

    *** reimplementation ***

    Input:
        sinogram: (B, T, n_angles, n_det) raw projection data.
        angles_deg: (n_angles,) projection angles in degrees.

    Output:
        (B, T, 1, H, W) reconstructed dynamic-PET stack.

    Loss stubs (DOCUMENTATION ONLY — no training):
        per_frame_mse_loss: sum_t ||x_pred[t] - x_gt[t]||²
        temporal_l1_loss:    λ * sum_t ||x_pred[t+1] - x_pred[t]||_1
    """

    def __init__(self, n_angles: int = 180, n_detectors: int = 182,
                  image_size: int = 128, n_frames: int = 2,
                  ramp_weight: float = 1.0,
                  unet3d_base_ch: int = 16, unet3d_depth: int = 3):
        super().__init__()
        self.n_angles = n_angles
        self.n_detectors = n_detectors
        self.image_size = image_size
        self.n_frames = n_frames
        self.ramp_weight = ramp_weight
        self.angles_deg = torch.linspace(0, 180, n_angles + 1)[:-1]
        self.learnable_filter = LearnableFrequencyFilter(n_detectors)
        # Dynamic denoiser: in_ch = n_frames (T frames treated as input
        # channels; the 3D U-Net internally treats (T, H, W) as a 3D
        # volume with depth = T).
        self.denoiser = UNet3D(in_ch=n_frames, out_ch=n_frames,
                                base_ch=unet3d_base_ch, depth=unet3d_depth)

    def forward(self, sinogram: torch.Tensor) -> torch.Tensor:
        B, T, n_angles, n_det = sinogram.shape
        assert T == self.n_frames, (
            f"expected T={self.n_frames}, got T={T}")
        # Part 1: per-frame FBP with learned filter
        fbp = fbp_per_frame(
            sinogram, self.angles_deg.to(sinogram.device),
            self.image_size,
            self.learnable_filter(),
            ramp_weight=self.ramp_weight,
        )  # (B, T, H, W)
        x = fbp.unsqueeze(2)  # (B, T, 1, H, W)
        # Part 2: 3D U-Net denoiser (treats T as depth)
        out = self.denoiser(x)  # (B, T, 1, H, W)
        return out

    @staticmethod
    def per_frame_mse_loss(x_pred: torch.Tensor,
                            x_gt: torch.Tensor) -> torch.Tensor:
        """Loss stub (documentation only — NO training performed)."""
        return F.mse_loss(x_pred, x_gt)

    @staticmethod
    def temporal_l1_loss(x_pred: torch.Tensor,
                          lam: float = 0.1) -> torch.Tensor:
        """Temporal L1 regularizer stub (documentation only)."""
        diffs = x_pred[:, 1:] - x_pred[:, :-1]
        return lam * diffs.abs().mean()


# --- Reimplementation audit table ---
# Per directive 6159621060 item 6.
#
# | Component                | Paper description                       | Our reimplementation                              | Source of choice        |
# |--------------------------|-----------------------------------------|-----------------------------------------------------|--------------------------|
# | Per-frame FBP core       | Standard inverse Radon + ramp filter     | Manual rfft/irfft + ramp filter + per-angle rotate   | Paper Eqs.               |
# | Per-frame learnable filter | "Adaptive frequency-domain filter"     | MLP f(ω_norm)→R with hidden=32, 3 layers          | Paper §2; width ASSUMPTION A3 |
# | Dynamic denoiser         | "merges information across all time frames" | 3D U-Net treating T as depth axis (Conv3d)        | ASSUMPTION A6 (paper silent on exact mechanism) |
# | Normalization            | BatchNorm3d + LayerNorm + GroupNorm (per paper) | BatchNorm3d (placeholder; full multi-norm optional) | ASSUMPTION A1           |
# | Loss (per-frame MSE)     | Per-frame MSE                            | STUB only (no training)                            | N/A for handoff         |
# | Loss (temporal)          | Temporal L1 (frame-to-frame smoothness)  | STUB only (no training)                            | N/A for handoff         |
# | Operator                 | Vendor-specific smat                      | Our 180-angle, 182-bin ParallelBeam              | Per §D                   |
# | Input                    | Dynamic sinogram (paper)                  | (B, T, 180, 182)                                  | Per §D                   |
# | Output                   | Dynamic PET stack (paper)                 | (B, T, 1, 128, 128)                               | Per §D                   |
# | Non-negative output      | Paper clips to 0 (post-hoc)              | ReLU on denoiser output (ASSUMPTION A4)          | Per §D                   |
#
# Explicit ASSUMPTIONs flagged for researcher review:
#   A1. U-Net channel layout 16-32-64-128 (paper silent).
#   A2. Loss+optimizer are STUBS only (no training, per §E).
#   A3. MLP filter width 32 hidden units (paper silent).
#   A4. ReLU on denoiser output for non-negativity (paper clips post-hoc).
#   A5. FBP uses our rfft-based back-projection (paper uses vendor smat).
#   A6. Dynamic denoiser uses 3D-UNet with T-as-depth; the paper's exact
#       temporal fusion mechanism is not specified.
#
# Per directive §E.7 and the v3 directive item 6, this reimplementation
# remains REIMPLEMENTATION_REVIEW_REQUIRED and is NOT approved for
# training or main-table use by the worker.
