"""FastPET reimplementation (Whiteley, Pan, et al., arXiv 2002.04665,
2020; later IEEE TRPMS publication).

*** reimplementation ***

This is a clean reimplementation built from the paper's equations
only. There is no public repository (per directive 6157488763 §A
"BLOCKED_NO_OFFICIAL_IMPLEMENTATION" finding).

The paper's FastPET approach:
  1. Construct "histo-images" from the sinogram: a fixed
     preprocessing that bins sinogram columns into a 2D
     representation suitable for direct CNN input.
  2. Apply a CNN (the paper uses an encoder-decoder U-Net with
     histo-image conditioning) to produce a 2D PET image.

The histo-image construction depends on the geometry; for our
180-angle, 182-detector operator we follow the paper's spirit:
each histo-image is a 2D tile where one axis is the detector
index and the other axis is the angle index (this is essentially
just a reshape of the sinogram with some optional blur). This
reimplementation treats the histo-image as a 2-channel "image"
(2 channels = real + sin-of-angle or two different orderings).

CRITICAL: zero-training, no paper numbers copied.

References:
    Whiteley, W., et al. "FastPET: Near Real-Time PET Reconstruction
        from Histo-Images Using a Neural Network", arXiv 2002.04665,
        2020.
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


class FastPETReimplementation(nn.Module):
    """FastPET reimplementation (*** reimplementation ***).

    Input: sinogram (B, n_angles, n_detectors).
    Output: 2D PET image (B, 1, image_size, image_size).

    We construct a "histo-image" by reshaping the sinogram to
    (B, 1, n_angles, n_detectors) (treating sinogram as a 2D
    image) and then resize/interpolate to (image_size, image_size).
    A 2D U-Net processes this histo-image and produces the output.

    ASSUMPTION: the histo-image construction used here is the
    simplest reshape + resize. The original paper uses a more
    sophisticated histo-image projection. The choice is the
    reimplementer's discretion; the model architecture (U-Net
    encoder-decoder) is standard.
    """

    def __init__(self, n_angles: int = 180, n_detectors: int = 182,
                  image_size: int = 128, base_ch: int = 32, depth: int = 4):
        super().__init__()
        self.n_angles = n_angles
        self.n_detectors = n_detectors
        self.image_size = image_size

        chs = [base_ch * (2 ** i) for i in range(depth + 1)]
        # Input: 1-channel histo-image (B, 1, H, W)
        self.encoders = nn.ModuleList()
        prev = 1
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
        """Treat sinogram as a 2D image: (B, 1, n_angles, n_detectors)
        and resize to (image_size, image_size). This is the
        simplest "histo-image" interpretation; the original
        paper uses a more sophisticated construction.
        """
        # (B, n_angles, n_det) -> (B, 1, n_angles, n_det)
        x = sinogram.unsqueeze(1)
        x = F.interpolate(x, size=(self.image_size, self.image_size),
                          mode="bilinear", align_corners=False)
        return x

    def forward(self, sinogram: torch.Tensor) -> torch.Tensor:
        x = self.sinogram_to_histo(sinogram)
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
        x = self.out_conv(x)
        return F.relu(x)


# --- Reimplementation audit table ---
# Per directive §C.
#
# | Component          | Paper description (Whiteley arXiv 2002.04665)  | Our reimplementation                                       | Source of choice              |
# |--------------------|--------------------------------------------------|----------------------------------------------------------|-------------------------------|
# | Histo-image        | Vendor / paper-specific projection             | Reshape sinogram to (n_angles, n_det) and resize         | ASSUMPTION (simplest)         |
# | CNN backbone       | Encoder-decoder                                  | 4-level U-Net, base_ch=32, channels 32-64-128-256      | Standard U-Net                |
# | Loss               | MSE on sinogram-domain ||A(x)-y||²              | NOT INCLUDED                                              | N/A for handoff              |
# | Output activation  | Non-negative (paper clips to 0)                 | ReLU (exact)                                              | Per §D                        |
# | Channels           | Paper silent                                     | 32-64-128-256                                             | ASSUMPTION                    |
#
# Explicit divergences:
#   F1. Histo-image construction is the simplest reshape+resize; the
#       original paper uses a more sophisticated projection.
#   F2. We do NOT include the histo-projection analytic forward model
#       from the paper's updated 2025 work ("Improved quantification in
#       end-to-end deep learning FastPET reconstruction using multi-view
#       histo-images of attenuation correction factors", DOI
#       10.1109/trpms.2025.3569198); that follow-up uses ACF histo-images
#       which require μ-maps we do not bundle here.
#   F3. Loss not included (zero-training smoke only).
#
# Per directive §C, this reimplementation MUST NOT enter the main
# table without explicit layer-by-layer audit and researcher approval.
