"""MedSAM2 / SAM 2 (Hiera) model builder.

This mirrors the role of ``build_sam.py`` (SAM 1 / ViT) but targets the SAM 2
Hiera architecture used by MedSAM2 (bowang-lab, arXiv:2411.09446).

The official medical-finetuned MedSAM2 weights are hosted on HuggingFace
(``wanglab/MedSAM2``).  When those weights are available locally they are used
directly; otherwise we fall back to the general SAM 2 Hiera-base-plus checkpoint
that ships with this project (``sam2-hiera-base-plus/sam2_hiera_base_plus.pt``).
Because our pipeline fine-tunes the SAM mask decoder on PET data, the medical
adaptation is learned during training regardless of the starting weights.

Key difference vs SAM 1: the SAM 2 prompt encoder / dense positional encoding
are hard-wired to the native 1024 input (64x64 embedding grid).  When we run at
a smaller ``image_size`` (256, matching the SAM-Med2D pipeline) we reconfigure
the prompt encoder grid to ``image_size // 16`` so that the dense PE matches the
Hiera feature map.
"""
import os
import torch

from sam2.build_sam import build_sam2

# ---------------------------------------------------------------------------
# Checkpoint / config resolution
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))

# Config files shipped inside the installed ``sam2`` package.
# sam2.1 configs match the checkpoint format used by the bowang-lab MedSAM2
# release (Nov 2024). sam2 configs match the older Meta SAM 2 release.
_CFG_HIERA_T = "configs/sam2.1/sam2.1_hiera_t.yaml"
_CFG_HIERA_B_PLUS = "configs/sam2/sam2_hiera_b+.yaml"
_CFG_HIERA_L = "configs/sam2/sam2_hiera_l.yaml"

# Candidate locations for the Hiera-base-plus weights (in priority order):
#   1. A downloaded medical MedSAM2 checkpoint (preferred, when present)
#   2. The local general SAM 2 Hiera-base-plus checkpoint (documented fallback)
# Candidate locations for MedSAM2 weights (Hiera-Tiny, the bowang-lab default
# backbone, sam2.1_hiera_t.yaml).  ``MedSAM2.pt`` is the official release.
_CANDIDATE_T = [
    os.path.join(_HERE, "MedSAM2.pt"),
    os.path.join(_HERE, "MedSAM2_latest.pt"),
    os.path.join(_HERE, "MedSAM2_2411.pt"),
    os.path.join(_HERE, "medsam2_best.pt"),
]

_CANDIDATE_B_PLUS = [
    "/mnt/d/MIAProject/sam2-hiera-base-plus/sam2_hiera_base_plus.pt",
    os.path.join(os.path.dirname(_HERE), "sam2-hiera-base-plus", "sam2_hiera_base_plus.pt"),
]

_CANDIDATE_L = [
    os.path.join(_HERE, "MedSAM2_large.pt"),
    os.path.join(_HERE, "sam2_hiera_large.pt"),
]


def _resolve_checkpoint(preferred, candidates):
    """Return the first existing checkpoint path, preferring ``preferred``."""
    search = []
    if preferred is not None:
        search.append(preferred)
    search.extend(candidates)
    for path in search:
        if path is not None and os.path.exists(path):
            return path
    raise FileNotFoundError(
        "No MedSAM2 / SAM2 checkpoint found. Searched: " + ", ".join(str(s) for s in search)
    )


def _reconfigure_prompt_encoder(model, image_size):
    """Adapt the SAM 2 prompt encoder grid to a non-native ``image_size``.

    The prompt encoder positional encoding is generated on the fly, so it works
    for any grid size once ``image_embedding_size`` / ``input_image_size`` are
    set consistently with the Hiera feature map (image_size // 16).
    """
    emb = image_size // 16
    pe = model.sam_prompt_encoder
    pe.input_image_size = (image_size, image_size)
    pe.image_embedding_size = (emb, emb)
    return model


def build_medsam2_hiera_t(
    image_size=256,
    checkpoint=None,
    device="cuda",
    mode="eval",
    reconfigure=True,
    encoder_adapter=True,  # accepted for API compatibility with build_sam.py; no-op for SAM2
):
    """Build the actual bowang-lab MedSAM2 (Hiera-Tiny backbone, sam2.1)."""
    ckpt = _resolve_checkpoint(checkpoint, _CANDIDATE_T)
    model = build_sam2(_CFG_HIERA_T, ckpt, device=device, mode=mode)
    if reconfigure and image_size != 1024:
        _reconfigure_prompt_encoder(model, image_size)
    print(f"*******load MedSAM2 (Hiera-T, real) from {ckpt}  (image_size={image_size})")
    return model


def build_medsam2_hiera_b_plus(
    image_size=256,
    checkpoint=None,
    device="cuda",
    mode="eval",
    reconfigure=True,
    encoder_adapter=True,
):
    """Build a SAM 2 Hiera-base-plus model (general SAM 2, NOT medical-finetuned)."""
    ckpt = _resolve_checkpoint(checkpoint, _CANDIDATE_B_PLUS)
    model = build_sam2(_CFG_HIERA_B_PLUS, ckpt, device=device, mode=mode)
    if reconfigure and image_size != 1024:
        _reconfigure_prompt_encoder(model, image_size)
    print(f"*******load MedSAM2 (Hiera-b+, general SAM2) from {ckpt}  (image_size={image_size})")
    return model


def build_medsam2_hiera_l(
    image_size=256,
    checkpoint=None,
    device="cuda",
    mode="eval",
    reconfigure=True,
    encoder_adapter=True,
):
    """Build a SAM 2 Hiera-large model (requires an l-variant checkpoint)."""
    ckpt = _resolve_checkpoint(checkpoint, _CANDIDATE_L)
    model = build_sam2(_CFG_HIERA_L, ckpt, device=device, mode=mode)
    if reconfigure and image_size != 1024:
        _reconfigure_prompt_encoder(model, image_size)
    print(f"*******load MedSAM2 (Hiera-l) from {ckpt}  (image_size={image_size})")
    return model


medsam2_registry = {
    "medsam2": build_medsam2_hiera_l,
    "medsam2_b": build_medsam2_hiera_b_plus,
    "medsam2_t": build_medsam2_hiera_t,  # the actual bowang-lab MedSAM2 (default)
    "default": build_medsam2_hiera_t,
}
