# SAM-KEM Experiment Report (v3 — canonical rerun)

_Build UTC: 2026-08-24T23:24:00Z_

This report is generated from `metrics/per_case_metrics_v3.csv` by `scripts/_regen_report_v3.py`. The report is byte-deterministic for a given input CSV set (the only varying field is the build timestamp above). Do not edit by hand.

## 1. Execution Summary

- Delivery version: v3 (Mode B canonical; Phase 0..10)
- Build UTC: 2026-08-24T23:24:00Z
- Audit archive: `SAM_KEM_QWEN3_AUDIT_ONLY_20260823.tar.gz` (preserved byte-for-byte; see AUDIT_ONLY_README.md)
- Per-case metrics rows: 14304
- Cross-count rows: 672
- Paired bootstrap rows: 8
- Frozen hyperparameter rows: 8

## 2. Protocol Deviations (cumulative)

| deviation_id | severity | status | description (truncated) |
|---|---|---|---|
| D01 | medium | open | HECKTOR locked-test split mandates 34 subjects per the authoritative work order; |
| D02 | low | resolved | BrainWeb locked-test split mandates 3 subjects (subject41, subject42, subject43) |
| D03 | high | resolved | Historical BrainWeb aggregate n=4080 is excluded from the canonical rerun. |
| D04 | high | resolved | Historical HECKTOR aggregate n=1763 is excluded from the canonical rerun. |
| D05 | high | resolved | Historical HECKTOR 179/37/8 split is retired. |
| D06 | low | resolved | Code-archive freeze-time SHA-256 computed: archive D:/MIAProject/EM-reconstructi |
| D07 | medium | resolved | QWEN3_VL_PLUS_API_KEY WAS provisioned on 2026-08-19 via the user-supplied DashSc |
| D08 | medium | partial | MedSAM2 Hiera-T decoder WAS fine-tuned per-dataset with the PET-adapted L1 objec |
| D09 | low | resolved | Real MedSAM2 Hiera-T inference (released pretrained weights, image_size=1024) wa |
| D10 | medium | resolved | System matrix was replaced with a precise discrete Radon transform (sparse line- |
| D11 | medium | resolved | BrainWeb 20e4 PSNRs: subject41 29.0-29.3 dB, subject42 28.9-29.1 dB, subject43 2 |
| D12 | medium | resolved | BW 20e4 per-method PSNRs (final, original torch_radon via WSL kernelSAM env): ML |
| D13 | medium | resolved | KEM iteration sweep expanded to all 3 BW test subjects × 3 realizations × {SelfK |
| D14 | low | resolved | Per-dataset fine-tuned MedSAM2 decoder checkpoints registered in checkpoints/dat |
| D15 | low | partial | Prompt record cache policy gap: per the plan, paired realizations at the same (i |
| D16 | medium | resolved | Poisson deviance and SSIM previously missing for BW (and Poisson deviance missin |
| D17 | high | open | Current SAM_KEM_QWEN3_FINAL package is preserved byte-for-byte as D:/MIAProject/ |
| D18 | high | open | Stale numbers in reports/EXPERIMENT_REPORT.md. Report quotes BW SAM-KEM-full - M |
| D19 | high | open | Test-set leakage: convergence/convergence.csv was generated against the original |
| D20 | low | open | v3 canonical rerun package sealed at D:/MIAProject/SAM_KEM_QWEN3_FINAL_v3.tar.gz |
| D21 | low | open | Provenance SHA-256 sidecar refreshed at provenance/files.sha256 via scripts/_pha |

## 3. Checkpoint Audit (Phase 1)

| checkpoint_id | encoder_frozen | decoder_differs_pretrained | decoder_differs_across_datasets | decision |
|---|---|---|---|---|
| brainweb | true | true | true | PASS |
| hecktor | true | true | true | PASS |

## 4. Anchor Box Cache (Phase 4)

- Total anchors: 148
- BrainWeb anchors: 12
- HECKTOR anchors: 136
- NEW test_locked coverage: 0/37

Note: prompts were generated only for the OLD test_locked subjects. After Phase 2's split swap, those subjects are in NEW validation, not NEW test_locked. The v3 canonical rerun uses the fallback centroid box for subjects not in this lookup.

## 5. Frozen Hyperparameters (Phase 6)

| dataset | count_level | sigma* | J* | N_KEM* | val_mean_NRMSE | val_mean_SSIM |
|---|---|---|---|---|---|
| brainweb | 100000 | 0.0500 | 2 | 100 | 1.0286 | 0.4272 |
| brainweb | 200000 | 0.0500 | 2 | 100 | 1.0578 | 0.4301 |
| brainweb | 500000 | 0.0500 | 2 | 100 | 1.1291 | 0.4331 |
| brainweb | 1000000 | 0.0500 | 2 | 100 | 1.3532 | 0.4296 |
| hecktor | 100000 | 0.0500 | 2 | 10 | 0.3159 | 0.9620 |
| hecktor | 200000 | 0.0500 | 2 | 20 | 0.2816 | 0.9729 |
| hecktor | 500000 | 0.0500 | 2 | 20 | 0.2501 | 0.9747 |
| hecktor | 1000000 | 0.0500 | 2 | 20 | 0.2396 | 0.9753 |

## 6. Headline PSNR by (dataset, count, method)

All numbers below are mean per-subject mean PSNR across 3 realizations, pulled from `metrics/per_case_metrics_v3.csv`.

### brainweb

| method | 100k | 200k | 500k | 1000k |
|---|---|---|---|---|
| Bowsher | 19.794 | 19.812 | 19.825 | 19.806 |
| CT-KEM | n/a | n/a | n/a | n/a |
| MLEM | 19.175 | 20.312 | 21.197 | 21.530 |
| MR-KEM | 17.077 | 18.588 | 19.791 | 20.087 |
| OSEM | 13.878 | 14.848 | 16.587 | 18.339 |
| SAM-KEM-MedSAM2-Qwen3-full | 14.602 | 14.360 | 14.715 | 14.926 |
| SAM-KEM-MedSAM2-Qwen3-no-post | 14.102 | 14.360 | 14.715 | 14.926 |
| SAM-prior-MedSAM2-Qwen3 | 14.185 | 14.408 | 14.496 | 14.504 |
| SelfKEM | 14.726 | 17.101 | 22.019 | 23.116 |

### hecktor

| method | 100k | 200k | 500k | 1000k |
|---|---|---|---|---|
| Bowsher | 29.320 | 29.446 | 29.576 | 29.626 |
| CT-KEM | 28.266 | 28.413 | 28.468 | 28.490 |
| MLEM | 29.968 | 31.344 | 32.531 | 33.237 |
| MR-KEM | n/a | n/a | n/a | n/a |
| OSEM | 27.441 | 29.227 | 32.291 | 34.223 |
| SAM-KEM-MedSAM2-Qwen3-full | 26.304 | 25.041 | 25.453 | 25.975 |
| SAM-KEM-MedSAM2-Qwen3-no-post | 26.304 | 25.041 | 25.453 | 25.975 |
| SAM-prior-MedSAM2-Qwen3 | 15.999 | 16.133 | 16.253 | 16.275 |
| SelfKEM | 32.406 | 36.124 | 38.720 | 39.815 |

## 7. Paired Bootstrap (Phase 8)

| dataset | count | ref | n_subjects | mean Δ PSNR | CI 95% |
|---|---|---|---|---|---|
| brainweb | 100000 | MR-KEM | 3 | -2.4740 | [-2.9462, -1.8254] |
| brainweb | 200000 | MR-KEM | 3 | -4.2280 | [-4.7271, -3.2585] |
| brainweb | 500000 | MR-KEM | 3 | -5.0764 | [-5.6405, -4.7804] |
| brainweb | 1000000 | MR-KEM | 3 | -5.1622 | [-6.2655, -4.3220] |
| hecktor | 100000 | CT-KEM | 18 | -1.9623 | [-2.3115, -1.5963] |
| hecktor | 200000 | CT-KEM | 18 | -3.3750 | [-4.1843, -2.6566] |
| hecktor | 500000 | CT-KEM | 18 | -3.0182 | [-3.8299, -2.1986] |
| hecktor | 1000000 | CT-KEM | 18 | -2.5159 | [-3.1278, -1.8878] |

## 8. Reproducibility

- Code commit/archive SHA-256: see `provenance/code_archive.sha256`.
- Forward operator SHA-256: see `arrays/system_matrix_A_n180_sha256.txt`.
- Headline config: `configs/headline_qwen3_medsam2.yaml` (system_matrix_id=torch_radon_parallelbeam_n180, MLEM=10, unified 180 angles, anchor cache enforced).
- Validator: `python scripts/validate_v3.py .` (must exit 0).
