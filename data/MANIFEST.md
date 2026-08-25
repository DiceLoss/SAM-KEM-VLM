# Data Manifest

This repo does **NOT** contain any large binary data. All large
files (checkpoints, raw nii.gz, per-slice sinograms/priors) live
outside this repo and are referenced by path and SHA.

## 1. Checkpoints (NOT in git)

| File | SHA-256 | Location |
|---|---|---|
| `medsam2_decoder_brainweb_v4.pt` | `2ee2b4c334c3fe4583092238742f9bd6d214c8e4af2d8221e94d005806f79ef7` | `D:/MIAProject/SAM_KEM_v3_checkpoints/` |
| `medsam2_decoder_hecktor.pt` | `ab6fbbb89fbc7d15e1ceee88936522a35dc46a3879aea8d4047acd653bb32d18` | `D:/MIAProject/SAM_KEM_v3_checkpoints/` |
| `MedSAM2.pt` (pretrained, upstream) | `c92743b99f00d078bf32a3afcc38aaa9faf1c1692dffe3eaa7a90938c1991060` | `D:/MIAProject/EM-reconstruction/` |

Each checkpoint file is ~156 MB. They are NOT git-tracked.

## 2. Raw data (NOT in git)

| File | Source | Location |
|---|---|---|
| BrainWeb nii.gz (20 subjects × 3 sim dirs × 4 files) | Original BrainWeb | `D:/MIAProject/brainweb/数据/brainweb_petmr/` |
| HECKTOR npz (~15 GB) | HECKTOR challenge | `D:/MIAProject/HECKTOR.npz` |
| Pretrained MedSAM2 (156 MB) | `EM-reconstruction/` | `D:/MIAProject/EM-reconstruction/MedSAM2.pt` |

## 3. Processed data (NOT in git)

The v3 pipeline produces per-slice sinograms, lambda files, and SAM
priors under `D:/MIAProject/SAM_KEM_QWEN3_MEDSAM2_DELIVERY/arrays/`.
Total ~30 GB across 3,511 slices × 12 combos = 42,132 sinograms
plus 12,612 priors. These are NOT in this repo.

The per-case metrics CSVs that summarize these arrays ARE in this
repo under `results/metrics/`.

## 4. AUDIT_ONLY archive (byte-unchanged)

| File | SHA-256 | Location |
|---|---|---|
| `SAM_KEM_QWEN3_AUDIT_ONLY_20260823.tar.gz` (603 MB) | `95dbb1a470fe1ddcf3e4f17b8801da6649e829b7fcca67253e193da12a7e52c4` | `D:/MIAProject/` |
| `SAM_KEM_QWEN3_AUDIT_ONLY_20260823.sha256` | (above) | `D:/MIAProject/` |

This archive is byte-unchanged from the original v1/v2 delivery. It
is NOT in this repo (it predates the v3 fix).

## 5. v3 FINAL archive

| File | SHA-256 | Location |
|---|---|---|
| `SAM_KEM_QWEN3_FINAL_v3.tar.gz` (~280 MB) | `c413227b944d1df81e9405353970ca68d36bcd81c42bc2d07d8a135be82c2d79` | `D:/MIAProject/` |
| `SAM_KEM_QWEN3_FINAL_v3.sha256` | (above) | `D:/MIAProject/` |

Sealed by `python code/src/v3_canonical.py --phase 10`.

## How to fetch the data store

```bash
# Checkpoints (must be present for the validator and any retraining)
mkdir -p D:/MIAProject/SAM_KEM_v3_checkpoints/
# Copy from the original delivery location:
cp D:/MIAProject/SAM_KEM_QWEN3_MEDSAM2_DELIVERY/checkpoints/medsam2_decoder_brainweb_v4.pt D:/MIAProject/SAM_KEM_v3_checkpoints/
cp D:/MIAProject/SAM_KEM_QWEN3_MEDSAM2_DELIVERY/checkpoints/medsam2_decoder_hecktor.pt D:/MIAProject/SAM_KEM_v3_checkpoints/

# Verify SHAs match
sha256sum D:/MIAProject/SAM_KEM_v3_checkpoints/*.pt
# Expected:
# 2ee2b4c334c3fe4583092238742f9bd6d214c8e4af2d8221e94d005806f79ef7  medsam2_decoder_brainweb_v4.pt
# ab6fbbb89fbc7d15e1ceee88936522a35dc46a3879aea8d4047acd653bb32d18  medsam2_decoder_hecktor.pt
```