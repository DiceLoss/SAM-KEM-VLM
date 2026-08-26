# Retired v3 workflow (forensic only)

> **Do not execute this workflow.** It is preserved only to explain the origin
> of historical v3 artifacts. It uses exposed test data, mixed operators and
> superseded checkpoints/normalization. The active workflow is
> [`EXECUTION.md`](EXECUTION.md).

This document describes the 10-step pipeline that produced the v3
canonical results. **All algorithm code lives in
`D:/MIAProject/EM-reconstruction/`** — this repo only adds
audit, split, validation, and orchestration code on top.

## Environments

| Env | Path | Python | Torch | CUDA | Used for |
|---|---|---|---|---|---|
| Windows native | `python` (system) | 3.13 | 2.10.0+cu130 | 13.1 | MedSAM2 inference/training, data extraction |
| WSL kernelSAM | `/home/miku/kernelSAM/bin/python` | 3.8 | 1.10.2+cu111 | 11.x | C++ torch_radon forward/backward at 180 angles |

Files share via `/mnt/d/` in WSL.

## Phase 0 — AUDIT_ONLY preservation (Windows)

```bash
python code/scripts/mark_audit_only.py   # adds audit_only_marker column to legacy CSVs
```

- Renames any old `SAM_KEM_QWEN3_FINAL.tar.gz` → `SAM_KEM_QWEN3_AUDIT_ONLY_<date>.tar.gz`
- Records SHA-256 in sidecar file
- Writes `AUDIT_ONLY_README.md` enumerating 12 voided issues
- Adds `AUDIT_ONLY_DO_NOT_USE_FOR_MANUSCRIPT` row-1 marker to every CSV in `results/metrics/` and `results/convergence/`

## Phase 1 — Checkpoint audit (Windows)

```bash
python code/scripts/audit_checkpoints.py
```

Verifies for both decoders (`medsam2_decoder_brainweb_v4.pt` and
`medsam2_decoder_hecktor.pt`):
1. SHA-256 matches expected value
2. `image_encoder.*` params are near-identical to pretrained MedSAM2 (`frozen=True`)
3. `sam_mask_decoder.*` params differ from pretrained AND between BW/HE
4. Training logs mention only train subjects (no val/test_locked leakage)
5. Inference loader prints `LOADED sha=… path=…` and the SHA matches expected

Writes `results/metrics/audit_report.csv`.

## Phase 2 — Split swap (Windows)

```bash
python code/scripts/swap_splits.py
```

Backs up v3 split to `splits/subject_split_v3_AUDIT.csv.csv`.
Swaps val ↔ test_locked for BW and HE.
Writes `splits/swap_log.csv`.

## Phase 3 — Config freeze (Windows)

Edit `code/configs/headline_qwen3_medsam2.yaml` and
`code/configs/canonical_experiment_config.yaml`:
- `data.system_matrix_id: torch_radon_parallelbeam_n180`
- `data.normalization_id: dataset_minmax_clip01`
- `method.mlem_iterations: 10`
- `method.forward_operator: discrete_radon_n180`
- `sam.checkpoint_sha256.brainweb: 2ee2b4c334c3fe4583092238742f9bd6d214c8e4af2d8221e94d005806f79ef7`
- `sam.checkpoint_sha256.hecktor: ab6fbbb89fbc7d15e1ceee88936522a35dc46a3879aea8d4047acd653bb32d18`
- `vlm.anchor_realization_enforced: true`
- `prompts.anchor_box_cache: prompts/anchor_box_lookup.csv`
- `inputs.scale_constant_C: 1.0`

## Phase 4 — Anchor box cache (Windows)

```bash
python code/scripts/build_anchor_box_cache.py
```

Reads `metrics/prompt_records.csv` (444 rows), filters to r=0, writes
`prompts/anchor_box_lookup.csv` (148 rows for the v3 dataset).
Records `prompts/anchor_box_audit.json` summary.

## Phase 5 — Forward operator port (Windows)

```bash
python code/scripts/port_forward_operator.py
```

Generates `arrays/system_matrix_A_n180_sha256.txt` (=`cac8a604…`) and
`arrays/adjoint_check_n180.json` (rel_diff=0).

## Phase 6 — Hyperparameter sweep on validation (Windows)

```bash
python code/scripts/sweep_hyperparams_v3.py
```

Reads `splits/subject_split.csv` filtered to `split=validation` (=3 BW +34 HE).
For each (dataset, count): runs MLEM × 10 → mlem_init; runs SelfKEM +
MR-KEM (BW) / CT-KEM (HE) on grid σ ∈ {0.02, 0.05, 0.10}, J ∈ {1,2,3},
N_KEM ∈ {10,20,40,60,80,100,120}. Boundary extension if N_KEM* ∈ {10,120}.
Writes `results/metrics/hyperparameter_sweep_v3.csv` and `frozen_hyperparameters.csv`.

```bash
python code/scripts/alpha_sweep_v3.py
```

For each (dataset, count), generates SAM priors on validation set and
sweeps α ∈ {0,0.1,…,0.5}. Updates `frozen_hyperparameters.csv` with
α*. For HE, α=0.0 was selected (prior too noisy to help). For BW,
α=0.1 at 100k, α=0.0 elsewhere.

## Phase 6 (BW extension) — Full slice coverage (Windows + WSL)

```bash
# Windows: extract all BW targets from raw nii.gz → 128×128 .npy
python code/scripts/extract_bw_targets.py

# WSL: regenerate sinograms for ALL (subject, slice, count, realization)
wsl -d Ubuntu -- bash -c "/home/miku/kernelSAM/bin/python code/scripts/wsl_regen_bw_all_slices.py"

# Windows: regenerate SAM priors for test+val with the v4 decoder
python code/scripts/gen_bw_sam_priors_v4.py
```

This expanded the BW training set from 168 samples (v3, 1 slice/subject)
to 29,520 samples (v4, 175 slices × 14 subjects × 12 combos).

## Phase 6 (decoder retraining) — BW MedSAM2 (Windows)

```bash
python code/scripts/retrain_bw_decoder_v4.py
```

Continues from the v3 decoder, trains for 3 epochs (29,520 steps) on
the v4 split. Final loss=0.005. Saves
`checkpoints/medsam2_decoder_brainweb_v4.pt` (sha `2ee2b4c3…`).

## Phase 7 — Canonical rerun on test_locked (WSL + Windows)

```bash
wsl -d Ubuntu -- bash -c "/home/miku/kernelSAM/bin/python code/scripts/wsl_bw_v4_all.py"
python code/scripts/rerun_convergence_v3.py   # HE convergence (CPU discrete_radon)
```

Writes `results/metrics/per_case_metrics_v3_bw.csv` (12,576 BW rows)
and `results/metrics/per_case_metrics_v3.csv` (merged with HE).

## Phase 8 — Statistics, figures, report (Windows)

```bash
python code/scripts/build_statistics_v3.py
python code/scripts/regen_figures_v3.py
python code/scripts/regen_report_v3.py
```

Writes `paired_subject_statistics_v3.csv`, `cross_count_results_v3.csv`,
`ablation_per_case_v3.csv`, `figures/*.pdf`, `EXPERIMENT_REPORT_v3.md`.

## Phase 9 — Validation (Windows + WSL)

```bash
wsl -d Ubuntu -- bash -c "/home/miku/kernelSAM/bin/python code/scripts/validate_v3.py ."
python code/scripts/validate_v3.py .       # cross-platform
```

Enforces 9 rules (see `validate_v3.py`). Must exit 0.

## Phase 10 — Final tarball (Windows)

```bash
python code/scripts/finalize.py
```

Tars this repo into `D:/MIAProject/SAM_KEM_QWEN3_FINAL_v3.tar.gz`.
Records sha in `D:/MIAProject/SAM_KEM_QWEN3_FINAL_v3.sha256`.
**Does NOT touch the AUDIT_ONLY archive.**

## Key file map

| What | Where |
|---|---|
| Original algorithms | `D:/MIAProject/EM-reconstruction/{EM,recon,discrete_radon,medsam2_decoder_*,poisson_simulate,metrics}.py` |
| v3 audit, split, validator | `code/scripts/` |
| v3 config snapshots | `code/configs/` |
| Final per-case metrics | `results/metrics/per_case_metrics_v3.csv` |
| Convergence | `results/convergence/convergence_v3.csv` |
| Headline report | `results/reports/EXPERIMENT_REPORT_v3.md` |
| Audit report | `results/metrics/audit_report.csv` |
| BW decoder | `checkpoints/medsam2_decoder_brainweb_v4.pt` (sha 2ee2b4c3…) |
| HE decoder | `checkpoints/medsam2_decoder_hecktor.pt` (sha ab6fbbb…) |
