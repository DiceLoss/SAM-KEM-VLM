# SAM-KEM v3 Canonical Pipeline

This repository contains the v3 canonical rerun of the SAM-KEM PET
reconstruction pipeline (Mode B audit). The v3 work was reviewed and
**8 fatal issues + several data/validation issues were identified**.
This commit **uploads all 18 previously-missing scripts plus the
required data and configuration files** for reviewer access. The
**code is NOT yet corrected** — see "Open issues" below.

## Repository layout

```
SAM_KEM_QWEN3_V3_REPO/
├── code/
│   ├── src/
│   │   ├── v3_canonical.py            # 10-stage orchestrator
│   │   ├── audit_checkpoints.py       # 5-point checkpoint audit
│   │   └── original/                  # 12 algorithm files copied verbatim from EM-reconstruction
│   │       ├── EM.py, recon.py, discrete_radon.py
│   │       ├── poisson_simulate.py, metrics.py
│   │       ├── build_medsam2.py, medsam2_decoder_{train,infer,inference}.py
│   │       └── ...
│   ├── scripts/                       # 18 pipeline scripts (NEWLY UPLOADED)
│   │   ├── _audit_checkpoints.py
│   │   ├── _build_anchor_box_cache.py
│   │   ├── _sweep_hyperparams_v3.py
│   │   ├── _alpha_sweep_v3.py
│   │   ├── _extract_bw_targets.py
│   │   ├── _redesign_bw_split_v4.py
│   │   ├── _wsl_regen_bw_all_slices.py
│   │   ├── _gen_bw_sam_priors_v4.py
│   │   ├── _retrain_bw_decoder_v4.py
│   │   ├── _wsl_bw_v4_all.py
│   │   ├── _rerun_convergence_v3.py
│   │   ├── _build_statistics_v3.py
│   │   ├── _regen_figures_v3.py
│   │   ├── _regen_report_v3.py
│   │   ├── _port_forward_operator.py
│   │   ├── _swap_splits.py
│   │   ├── _phase0_mark_audit_only.py
│   │   ├── _phase10_finalize.py
│   │   ├── _build_case_manifest.py    # builds results/metrics/case_manifest.csv
│   │   └── validate_v3.py
│   └── configs/                       # YAML + split CSVs + audit reports
├── data/MANIFEST.md                   # External data store index
├── prompts/
│   ├── anchor_box_lookup.csv          # NEW: Qwen3 box cache
│   ├── anchor_box_audit.json          # NEW: cache audit
│   ├── brainweb_qwen3_samples/        # NEW: 5 sample Qwen JSON responses
│   └── hecktor_qwen3_samples/         # NEW: 5 sample Qwen JSON responses
├── logs/
│   ├── medsam2_train_brainweb_stdout.log   # NEW
│   └── medsam2_train_hecktor_stdout.log    # NEW
├── arrays/
│   └── system_matrix_A_n180_sha256.txt    # NEW: forward operator SHA
├── docs/
│   ├── v3_workflow.md
│   └── decisions.md
├── results/                           # Metrics, figures, reports
│   ├── metrics/
│   │   ├── per_case_metrics_v3.csv
│   │   ├── per_case_metrics_v3_bw.csv
│   │   ├── cross_count_results_v3.csv
│   │   ├── paired_subject_statistics_v3.csv
│   │   ├── ablation_per_case_v3.csv
│   │   ├── frozen_hyperparameters.csv
│   │   ├── hyperparameter_sweep_v3.csv    # NEW: full sweep raw results
│   │   ├── prompt_records.csv
│   │   ├── system_matrix_A_n180_sha256.txt
│   │   └── case_manifest.csv             # NEW: per-case repro manifest
│   ├── convergence/convergence_v3.csv
│   ├── figures/                          # PDFs
│   └── reports/                          # EXPERIMENT_REPORT_v3.md + PSNR_INVESTIGATION_REPORT.md
├── requirements.txt                    # NEW: Windows env
├── environment.yml                     # NEW: Conda YAML
├── Dockerfile.windows                  # NEW: Docker image
└── README.md
```

## Key SHAs (DO NOT change)

| Artifact | SHA-256 (prefix) |
|---|---|
| AUDIT_ONLY archive | `95dbb1a470fe1ddcf3e4f17b8801da6649e829b7fcca67253e193da12a7e52c4` |
| v3 BW MedSAM2 decoder | `2ee2b4c334c3fe4583092238742f9bd6d214c8e4af2d8221e94d005806f79ef7` |
| v3 HE MedSAM2 decoder | `ab6fbbb89fbc7d15e1ceee88936522a35dc46a3879aea8d4047acd653bb32d18` |
| Forward operator A (180 angles, 128×128) | `cac8a60455083217b0f51e6698bf934bd8f551cfca81abca0a318359c9d7b60a` |
| Pretrained MedSAM2 | `c92743b99f00d078...` |

## Headline results (200k count, C++ torch_radon for BW, discrete_radon for HE)

| Method | BW | HE |
|---|---|---|
| MLEM | **20.31** | **31.34** |
| SAM-prior | 14.41 | 16.13 |
| SAM-KEM-no-post | 14.36 | 25.04 |
| SAM-KEM-full (α=0) | 14.36 | 25.04 |

## OPEN ISSUES (from peer review — NOT YET FIXED)

The v3 results in `results/` are **not yet trustworthy** as paper
evidence. The following issues were identified in peer review and
**remain uncorrected** in this commit:

### Fatal issues (8)

1. **Tests didn't actually use Qwen3 box.** Anchor cache covers 0/37
   NEW test_locked subjects; all used the fallback centroid box.
   The "Qwen3" suffix on the SAM-KEM methods is therefore misleading.
2. **Box cache dimensions reversed.** Same image_sha256 was used
   across counts but different per realization, producing 444
   independent Qwen API calls instead of the planned 148 cached
   anchors.
3. **HECKTOR "KEM" is not coefficient-space KEM.** `recon.py::kem()`
   applies image-space smoothing via K@img; it does not update α
   in coefficient space as described in the paper. `EM.py::kem()`
   is closer to the paper but is not what HECKTOR used.
4. **MedSAM2 inference returns binary mask, not continuous prior.**
   `predict(..., multimask_output=False)` without
   `return_logits=True` returns thresholded binary mask. The paper
   claims a continuous prior.
5. **Checkpoint cache bug.** `medsam2_decoder_infer.py::_ensure_model`
   returns immediately if `_MODEL is not None`, ignoring any new
   `ckpt_path`. Process-global cache means a BW-trained model stays
   loaded for HE inference if BW ran first.
6. **Forward operator not unified.** BW uses C++ torch_radon (via WSL),
   HE uses discrete_radon. The headline config claims
   `forward_operator: discrete_radon_n180` everywhere. BW
   poisson_deviance = 0 for all 12,576 rows (Poisson noise not modeled
   on the BW side).
7. **Per-image min-max normalization violates C=1.0.** `recon.py::mlem`,
   `osem`, `kem`, `bowsher` all call `normalize_to_unit_max=True`,
   dividing by `img.max()` after every iteration. The paper says
   input/target use a fixed dataset-wide C; only the SAM prior
   kernel feature may be per-prior min-max.
8. **Poisson seed not per-case independent.** `preprocessing.py:113`
   uses `seed = POISSON_SEED_BASE + realization`, so all cases share
   the same Poisson random stream conditioned on r alone.

### Data issues

- BW test subject-split coverage is ~131/528 (24.8%) without declared
  sampling rule. HE test coverage is 18/34 subjects.
- `subject_split_v4.csv` contains only BrainWeb, not HECKTOR.
- v4 split source-data SHAs and split script SHAs are TBD.
- Hyperparameter sweep is degenerate: σ=0.05, J=2, α=0.0 only.
  `frozen_hyperparameters.csv` lists `alpha_star=0.1` for BW 100k
  with no sweep row that supports it.
- `N_KEM*` was chosen by minimizing mean( SelfKEM, MR-KEM ) NRMSE,
  not by validating SAM-KEM.
- `cross_count_results_v3.csv` has 672 rows from 64 unique keys
  (per-key 3 or 18 duplicates).
- `ablation_per_case_v3.csv` is identical to `per_case_metrics_v3.csv`
  — not a real ablation.
- `validate_v3.py` PASSES but checks are weak: anchor cache missing
  is only a warning, paths are wrong (`root/scripts` vs `root/code/scripts`),
  per-case completeness is not verified, no check on actual loaded
  checkpoint, no continuous-prior check, no fixed-C check, no
  deviance-zero check, no test-coverage check, no hyperparam-sweep
  presence check.

## Reproduction

```bash
# Windows native (Python 3.13, torch 2.10, MedSAM2)
pip install -r requirements.txt

# WSL kernelSAM venv (Python 3.8, torch 1.10.2+cu111, C++ torch_radon)
/home/miku/kernelSAM/bin/python code/scripts/audit_checkpoints.py
/home/miku/kernelSAM/bin/python code/scripts/validate_v3.py .

# Re-run canonical pipeline (long)
/home/miku/kernelSAM/bin/python code/scripts/_wsl_regen_bw_all_slices.py
/home/miku/kernelSAM/bin/python code/scripts/_wsl_bw_v4_all.py
python code/scripts/_gen_bw_sam_priors_v4.py
python code/scripts/_retrain_bw_decoder_v4.py
python code/scripts/_build_case_manifest.py
python code/scripts/_build_statistics_v3.py
python code/scripts/_regen_figures_v3.py
python code/scripts/_regen_report_v3.py
python code/scripts/_phase10_finalize.py
```

## Next steps (post-review)

Tasks #32–#35 are open in the issue tracker:
1. Fix MedSAM2 inference (`return_logits=True`, return logit array
   not binary mask).
2. Fix checkpoint cache (always rebuild when ckpt_path changes).
3. Fix HECKTOR KEM to use `EM.py` coefficient-space updates.
4. Remove per-image `normalize_to_unit_max`; switch to fixed C.

After these fixes, the v3 results must be re-generated from scratch
with: regenerated Qwen3 anchors covering all NEW test_locked subjects,
fixed-C normalization throughout, continuous-logit SAM prior, and
proper α/σ/J sweep on validation.