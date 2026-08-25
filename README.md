# SAM-KEM v3 Canonical Pipeline

This repository contains the **v3 canonical rerun** of the SAM-KEM PET
reconstruction pipeline (Mode B audit). All reconstruction algorithms
(MLEM, OSEM, Bowsher, SelfKEM, MR-KEM, CT-KEM, SAM-KEM) are reused
from the original codebase at `D:/MIAProject/EM-reconstruction/` —
no algorithm code was duplicated.

## Repository layout

```
SAM_KEM_QWEN3_V3_REPO/
├── code/                         # ALL v3-specific code
│   ├── src/                      # Pipeline drivers (call into original codebase)
│   ├── configs/                  # YAML configs (split, splits, frozen hyperparameters)
│   └── scripts/                  # Thin shell scripts (audit, validator, finalize)
├── data/
│   ├── raw/                      # Raw BrainWeb nii.gz + HECKTOR npz
│   ├── interim/                  # Per-slice 128×128 targets, anatomy
│   └── processed/                # Sinograms, lambda, SAM priors (per slice/count/realization)
├── checkpoints/                  # Trained MedSAM2 decoders (sha-tracked)
├── results/
│   ├── metrics/                  # per_case_metrics_v3.csv, frozen_hyperparameters.csv, etc.
│   ├── convergence/              # convergence_v3.csv
│   ├── figures/                  # PDFs
│   └── reports/                  # EXPERIMENT_REPORT_v3.md
├── logs/                         # Training logs, validation logs
├── docs/                         # Audit notes, decision records
└── README.md
```

## What this repo adds vs the original codebase

The original codebase (`D:/MIAProject/EM-reconstruction/`) is the source
of truth for all algorithms:
- `EM.py` — MLEM, KEM, SAM-KEM, Bowsher (with `radon=ParallelBeam` for GPU)
- `EM-reconstruction/recon.py` — CPU discrete_radon (sparse matrix)
- `EM-reconstruction/medsam2_decoder_train.py` — MedSAM2 training driver
- `EM-reconstruction/medsam2_decoder_infer.py` — MedSAM2 inference
- `EM-reconstruction/discrete_radon.py` — exact-adjoint sparse Radon
- `EM-reconstruction/poisson_simulate.py` — Poisson noise generator
- `EM-reconstruction/metrics.py` — PSNR/SSIM/NRMSE/MAE

This repo provides:
1. **A v3 split** (`code/configs/subject_split_v4.csv`) that uses ALL
   usable slices per BrainWeb subject (172-181 slices × 14 train
   subjects = 2460 train slices), instead of the v3 1-slice-per-subject
   choice (14 train slices). HE keeps the v3 single-slice split.
2. **A 5-point checkpoint audit** (`code/scripts/audit_checkpoints.py`)
   that verifies hash, encoder-freeze, decoder-differs, training-log
   cleanliness, and inference-SHA match for each MedSAM2 decoder.
3. **A 9-rule validator** (`code/scripts/validate_v3.py`) that enforces
   the v3 canonical contract (split counts, config SHA match, anchor
   cache, no skimage Radon, AUDIT_ONLY markers preserved, no leakage,
   unique keys, decoder SHA match per-row).
4. **A canonical pipeline driver** (`code/src/v3_canonical.py`) that
   orchestrates the 10 Phase 0–10 steps using the original algorithms.
5. **Final artifact package** (`code/scripts/finalize.py`) that seals
   the v3 tree into `D:/MIAProject/SAM_KEM_QWEN3_FINAL_v3.tar.gz`
   without touching the AUDIT_ONLY archive.

## How to reproduce v3 results

```bash
# From WSL kernelSAM venv (Python 3.8, torch 1.10.2+cu111, C++ torch_radon)
/home/miku/kernelSAM/bin/python code/scripts/audit_checkpoints.py
/home/miku/kernelSAM/bin/python code/scripts/validate_v3.py .
/home/miku/kernelSAM/bin/python code/scripts/finalize.py
```

All Phase 0-10 steps are documented in `docs/v3_workflow.md`.

## Key SHAs (DO NOT change)

| Artifact | SHA-256 (prefix) |
|---|---|
| AUDIT_ONLY archive | `95dbb1a470fe1ddcf3e4f17b8801da6649e829b7fcca67253e193da12a7e52c4` |
| v3 BW MedSAM2 decoder | `2ee2b4c334c3fe4583092238742f9bd6d214c8e4af2d8221e94d005806f79ef7` |
| v3 HE MedSAM2 decoder | `ab6fbbb89fbc7d15e1ceee88936522a35dc46a3879aea8d4047acd653bb32d18` |
| Forward operator A (180 angles, 128×128) | `cac8a60455083217b0f51e6698bf934bd8f551cfca81abca0a318359c9d7b60a` |

## Headline results (200k count, 180 angles, C++ torch_radon for BW,
   discrete_radon for HE)

| Method | BW | HE |
|---|---|---|
| MLEM | 20.31 | **31.34** |
| SAM-prior | 14.41 | 16.13 |
| SAM-KEM-no-post | 14.36 | 25.04 |
| SAM-KEM-full (α=0) | 14.36 | 25.04 |

Full results in `results/metrics/` and `results/reports/EXPERIMENT_REPORT_v3.md`.

## Known limitations (documented for review)

1. **BW SAM-KEM < MR-KEM**: BW SAM prior (14.41 dB) is too noisy to
   beat the MR-anatomy kernel feature (MR-KEM 18.59 dB at 200k). The
   alpha sweep correctly selected α≈0 to prevent the prior from
   degrading the result.
2. **HE SAM-KEM < SelfKEM**: same root cause — SAM prior (16.13 dB)
   is worse than MLEM init (31.34 dB) as a kernel feature.
3. **BW data still smaller than v1**: v3 uses 2460 train slices × 12
   combos = 29,520 samples (vs v1's 612 slices × 12 combos = 7,344
   samples for the IF paper's MRIBowsher benchmark). The v4 split is
   the largest possible given the source BrainWeb nii.gz volumes.
4. **HE decoder retrained only on 60 steps** (n_train=1872 samples).
   BW decoder was retrained on 29,520 steps (loss 0.185→0.005). HE
   retraining is a follow-up item — see `docs/decisions.md`.