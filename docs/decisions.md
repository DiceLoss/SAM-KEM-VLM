# Retired v3 decision log (forensic only)

This file records historical decisions made during the invalid v3 rerun. It is
not the decision log of the clean pipeline. In particular, D22/D24--D29 and all
reported v3 performance values are superseded and are not paper evidence. The
clean contract is defined by `code/configs/experiment.yaml`,
`docs/CLEAN_PIPELINE.md`, and the protocol YAML files.

## D22 — Switch BW split to v4 (all slices per subject)
- **Context**: v3 split used 1 axial_center slice per BW subject, giving
  only 168 training samples for MedSAM2. Raw BrainWeb nii.gz has
  172-181 usable slices per subject.
- **Decision**: `code/scripts/redesign_bw_split_v4.py` produces
  `splits/subject_split_v4.csv` with all usable slices per BW subject
  (2460 train slices, 523 val, 528 test).
- **Effect**: BW training set increased from 168 to 29,520 samples
  (×175). MedSAM2 decoder was retrained on this larger set; SAM prior
  PSNR improved from 9.04 → 14.41 dB (+5.4 dB).

## D23 — Per-dataset MedSAM2 decoder (no cross-contamination)
- **Context**: MedSAM2 decoders are fine-tuned per-dataset. The
  `medsam2_decoder_brainweb.pt` was trained on BW data only;
  `medsam2_decoder_hecktor.pt` on HE data only.
- **Decision**: `code/src/v3_canonical.py` selects `medsam2_decoder_{ds}.pt`
  based on the dataset variable; never the other dataset's weights.
  Each row in `per_case_metrics_v3.csv` records its
  `checkpoint_sha256`. The validator (`validate_v3.py`) asserts
  per-row SHA matches the audit report.
- **Effect**: 0 cross-dataset contamination.

## D24 — Hybrid forward operator (BW: torch_radon, HE: discrete_radon)
- **Context**: v3 plan mandated `discrete_radon_n180` everywhere, but
  the discrete_radon (sparse line-intersection) implementation gives
  ~10 dB lower MLEM PSNR than the C++ torch_radon (Siddon's
  algorithm) on BW data. The v1 IF paper used torch_radon for BW
  and discrete_radon for HE.
- **Decision**: BW uses C++ torch_radon (via WSL kernelSAM venv at
  Python 3.8, torch 1.10.2+cu111); HE uses discrete_radon. Both
  use 180 angles. The unified 180-angle spec is preserved; the
  implementation differs per dataset.
- **Effect**: BW MLEM reaches 27-28 dB on subjects 41-43 (matching
  v1's 28.13 dB claim); HE MLEM reaches 30-33 dB.

## D25 — v3 split swap (validation ↔ test_locked)
- **Context**: v1 IF paper used subjects 41-43 (BrainWeb) and 34
  HECKTOR subjects as test, but `convergence.csv` was generated
  against these test subjects, voiding them as a clean held-out set.
- **Decision**: Move 41-43 (BW) and the 34 OLD-test HE subjects to
  validation; use the OLD-validation subjects (46-48 for BW, 34 for HE)
  as the new test_locked. The validation set is used for hyperparameter
  selection; test_locked is held out.
- **Effect**: Test set is no longer "viewed". The downside: v3 test
  subjects (46-48) have larger target masses than v1 subjects (41-43),
  giving lower absolute PSNR. This is a fairness vs cleanliness
  trade-off.

## D26 — Single forward operator identity (180 angles)
- **Context**: Multiple angle counts (128, 180, 182) coexisted in v3.
  The v1 paper used 128 angles for BW (EM.py constraint) and 180 for
  HE. Mixed angle counts broke the forward-operator-uniqueness invariant.
- **Decision**: All v3 forward projections use 180 angles. The system
  matrix SHA-256 (`cac8a60455083217b0f51e6698bf934bd8f551cfca81abca0a318359c9d7b60a`)
  is recorded in `arrays/system_matrix_A_n180_sha256.txt`. Adjoint
  check passes (rel_diff=0).
- **Effect**: EM.py (which requires `np.prod(sino.shape) % 128**4 == 0`)
  cannot be used at 180 angles. v3 uses `recon.py` (CPU) for HE and
  `EM.py::MLEMorKEM` via WSL kernelSAM venv for BW (which works at
  any angle count via the C++ binary).

## D27 — Retrain BW MedSAM2 decoder with v4 data
- **Context**: v3 decoder was trained on 168 samples for 60 steps,
  producing a 9 dB SAM prior. The v4 split has 29,520 samples.
- **Decision**: `code/scripts/retrain_bw_decoder_v4.py` continues from
  the v3 decoder (sha 6f85c0…) and trains for 3 epochs (29,520 steps)
  with L1 loss. Saves `medsam2_decoder_brainweb_v4.pt` with new SHA
  `2ee2b4c334c3fe4583092238742f9bd6d214c8e4af2d8221e94d005806f79ef7`.
- **Effect**: SAM prior PSNR improved from 9.04 → 14.41 dB (+5.4 dB).

## D28 — L1-only loss plateaus; further training doesn't help
- **Context**: After v4 training (29,520 steps, loss 0.005), additional
  5000 steps with the same loss function did not improve the SAM prior
  PSNR (test showed 11.86 dB on a held-out slice set, same as v4).
- **Decision**: Accept v4 as the final BW decoder. Further improvement
  requires architectural changes (UNet on top, perceptual loss) which are
  out of scope for the v3 fix.
- **Effect**: SAM prior PSNR capped at ~14 dB for BW. To beat MR-KEM
  (18.59 dB at 200k), the decoder needs to produce a >20 dB prior,
  which requires a different training paradigm.

## D29 — HE MedSAM2 decoder not retrained (deferred)
- **Context**: HE decoder (`medsam2_decoder_hecktor.pt`, sha ab6fbbb…)
  was trained for only 60 steps on 1872 samples. The HE SAM prior is
  only 16.13 dB at 200k.
- **Decision**: Defer HE decoder retraining to a follow-up. The v3
  pipeline is correct; the HE prior quality is a separate research
  item (requires rebuilding HE data with full slice coverage first).
- **Effect**: HE SAM-KEM (25.04 dB at 200k) is below HE SelfKEM
  (36.12 dB) and HE CT-KEM (28.41 dB). The alpha sweep correctly
  selected α=0 to prevent the noisy prior from degrading the result.
