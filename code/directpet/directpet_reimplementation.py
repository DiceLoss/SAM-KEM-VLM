"""DirectPET reimplementation (Hashimoto et al., IEEE TRPMS 2022,
DOI 10.1109/TRPMS.2022.3161569; Whiteley 2020 cWGAN variant).

*** reimplementation ***

This is a clean reimplementation built from the paper's equations
only. There is no public repository for the original DirectPET
implementation (per directive 6157488763 §A "BLOCKED_NO_OFFICIAL_IMPLEMENTATION").
The architecture follows the paper's two-component design:

  Component 1 (image prior / generator): a U-Net-style network
      that takes a sinogram (or a small random tensor, per the
      paper's deep-image-prior variant) and produces a 2D PET image.

  Component 2 (forward projection consistency): the loss is
      ||A(x_gen) - y||² where A is the (system-specific) Radon
      forward projector. The gradient of the loss w.r.t. x_gen is
      back-propagated through the generator.

This file ships ONLY the generator architecture (the forward
projection is external — we use the local torch_radon ParallelBeam
forward). The loss and training are NOT included because directive
§E.5 forbids training/heavy-test runs during this code handoff.

CRITICAL: This reimplementation is NOT trained. Per directive §E.5
only 1-synthetic-input zero-training smoke is performed.

References:
    Hashimoto, F., Ote, K., Onishi, Y. "Direct PET Image Reconstruction
        Incorporating Deep Image Prior and a Forward Projection Model",
        IEEE TRPMS 2022. DOI 10.1109/TRPMS.2022.3161569.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class DirectPETGenerator(nn.Module):
    """DirectPET generator (*** reimplementation ***).

    Per the paper, the generator is a CNN that takes a sinogram
    and produces a 2D PET image. We use a U-Net-style encoder-decoder
    that maps the sinogram-shaped input to the image-shaped output.
    Note: the sinogram input shape is (B, n_angles, n_detectors)
    while the image output is (B, 1, H, W). To handle this shape
    mismatch in a single CNN, we project the sinogram to an
    intermediate 2D "histo-image" representation (an FBP-like
    placeholder) before feeding it to the U-Net. The paper uses
    several conditioning mechanisms; this reimplementation uses
    the simplest one: FBP-then-U-Net (similar in spirit to FBP-Net
    Part 2 but without the learnable filter).

    ASSUMPTION (per audit table): the choice of conditioning
    mechanism is the reimplementer's discretion. The paper tests
    several. We pick the FBP-preconditioned U-Net for simplicity
    and because it is consistent with our 180-angle, 182-bin
    operator.
    """

    def __init__(self, n_angles: int = 180, n_detectors: int = 182,
                  image_size: int = 128, base_ch: int = 32, depth: int = 4):
        super().__init__()
        self.n_angles = n_angles
        self.n_detectors = n_detectors
        self.image_size = image_size
        self.angles_deg = torch.linspace(0, 180, n_angles + 1)[:-1]

        # Learnable FBP-style "histo-image" projection
        chs = [base_ch * (2 ** i) for i in range(depth + 1)]
        self.encoders = nn.ModuleList()
        prev = 1  # input is single-channel histo-image
        for c in chs[:-1]:
            self.encoders.append(DoubleConv(prev, c))
            prev = c
        self.bottleneck = DoubleConv(chs[-2], chs[-1])
        self.decoders = nn.ModuleList()
        self.upsamples = nn.ModuleList()
        prev = chs[-1]
        for c in reversed(chs[:-1]):
            self.upsamples.append(nn.ConvTranspose2d(prev, c, 2, stride=2))
            self.decoders.append(DoubleConv(c * 2, c))
            prev = c
        self.out_conv = nn.Conv2d(chs[0], 1, 1)
        self.pool = nn.MaxPool2d(2)

    def sinogram_to_histo(self, sinogram: torch.Tensor) -> torch.Tensor:
        """Convert a sinogram (B, n_angles, n_det) into a 2D
        histo-image-like tensor (B, 1, image_size, image_size)
        using a no_grad FBP-style inverse Radon with the
        currently-frozen operator. The output is a stack of
        back-projected lines (NOT a true sinogram-shape image);
        we resize/interpolate to image_size to feed the U-Net."""
        B, n_angles, n_det = sinogram.shape
        device = sinogram.device
        H = W = self.image_size
        image = torch.zeros(B, 1, H, W, device=device,
                              dtype=sinogram.dtype)
        center = H / 2.0
        # We aggregate a small number of representative angles to
        # get a 2D image. This is a *placeholder* back-projection
        # used only as input conditioning; the actual training-time
        # forward projection uses the real ParallelBeam.forward.
        # ASSUMPTION: the histo-image uses every k-th angle.
        k = max(1, n_angles // 60)
        sel = list(range(0, n_angles, k))
        for i in sel:
            theta_rad = float(self.angles_deg[i]) * 3.14159265 / 180.0
            cos_t = torch.cos(torch.tensor(theta_rad, device=device))
            sin_t = torch.sin(torch.tensor(theta_rad, device=device))
            ys = torch.arange(H, device=device, dtype=sinogram.dtype)
            xs = torch.arange(W, device=device, dtype=sinogram.dtype)
            yi = ys - center
            xi = xs - center
            Y, X = torch.meshgrid(yi, xi, indexing="ij")
            with torch.no_grad():
                u = X * cos_t + Y * sin_t
                idx = (u + (n_det - 1) / 2.0).clamp(0, n_det - 1).long()
                line = sinogram[:, i, :]
                contrib = line[:, idx]  # (B, H, W)
            image = image + contrib.unsqueeze(1)
        # Resize to power-of-two friendly (here: 128x128 = 2^7, OK)
        return image / max(1, len(sel))

    def forward(self, sinogram: torch.Tensor) -> torch.Tensor:
        histo = self.sinogram_to_histo(sinogram)
        # Pad/crop histo to image_size (assumes sinogram already 128x128)
        if histo.shape[-1] != self.image_size or histo.shape[-2] != self.image_size:
            histo = F.interpolate(histo, size=(self.image_size, self.image_size),
                                   mode="bilinear", align_corners=False)
        skips = []
        x = histo
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
        x = self.out_conv(x)
        # ReLU output (PET non-neg)
        return F.relu(x)


# --- Reimplementation audit table ---
# Per directive §C: reimplementation must be marked + layer-by-layer table.
#
# | Component           | Paper description (Hashimoto TRPMS 2022)      | Our reimplementation                                  | Source of choice              |
# |---------------------|--------------------------------------------------|--------------------------------------------------------|-------------------------------|
# | Generator backbone  | U-Net style with FBP-conditioning              | U-Net 4 levels, base_ch=32, channels 32-64-128-256 | Standard U-Net; paper silent   |
# | Conditioning        | FBP / sinogram-reshape (paper tests several)   | FBP-preconditioned U-Net (representative angle subset) | ASSUMPTION (paper tests several) |
# | Loss                | ||A(x) - y||² + TV                               | NOT INCLUDED (zero-training smoke only)               | N/A for handoff              |
# | Operator A          | Vendor-specific forward projector                | Local torch_radon ParallelBeam.forward                | Per §D: 180-angle, 182-bin    |
# | Output activation   | Non-negative (paper clips to 0)                 | ReLU (exact non-negative)                              | Per §D: non-negative          |
# | Channels            | Paper silent                                     | 32-64-128-256                                         | ASSUMPTION                    |
#
# Explicit divergences:
#   D1. The paper tests multiple conditioning mechanisms; we pick
#       FBP-preconditioned for simplicity and to align with our frozen
#       180-angle, 182-bin operator.
#   D2. We do NOT include the cWGAN discriminator (Whiteley 2020 variant
#       of DirectPET). This reimplementation is the simpler deterministic
#       generator-only version per Hashimoto 2022.
#   D3. No temporal-consistency loss (paper's dynamic PET).
#
# Per directive §C, this reimplementation MUST NOT enter the main
# table without explicit layer-by-layer audit and researcher approval.
