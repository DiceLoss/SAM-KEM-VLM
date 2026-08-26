# Retired original algorithm snapshot (forensic only)

> **Do not import or execute these files for a clean run.** They are preserved
> byte-for-byte where possible to support the v3 audit. The active tested
> implementations live in `code/sam_kem/`.

This directory contains the algorithm source files used by the v3
canonical pipeline. **These files are copied verbatim from the
original codebase at `D:/MIAProject/EM-reconstruction/`** (and from
the v3 delivery at `D:/MIAProject/SAM_KEM_QWEN3_MEDSAM2_DELIVERY/`
for files that were re-packaged there).

## Files

| File | Source | Purpose |
|---|---|---|
| `EM.py` | `EM-reconstruction/EM.py` | GPU MLEM, KEM, SAM-KEM, Bowsher (uses `radon=ParallelBeam`) |
| `recon.py` | `delivery/scripts/recon.py` | CPU MLEM, OSEM, KEM, Bowsher, SAM-KEM (uses sparse-matrix `discrete_radon`) |
| `discrete_radon.py` | `delivery/scripts/discrete_radon.py` | Exact-adjoint sparse Radon (180 angles canonical) |
| `poisson_simulate.py` | `delivery/scripts/poisson_simulate.py` | Poisson noise generator for sinograms |
| `metrics.py` | `delivery/scripts/metrics.py` | PSNR, SSIM, NRMSE, MAE |
| `build_medsam2.py` | `EM-reconstruction/build_medsam2.py` | Builds MedSAM2 Hiera-T model |
| `medsam2_decoder_train.py` | `delivery/scripts/medsam2_decoder_train.py` | Training driver (60 steps, L1 loss, encoder frozen) |
| `medsam2_decoder_infer.py` | `delivery/scripts/medsam2_decoder_infer.py` | Inference driver (box-prompted) |
| `medsam2_inference.py` | `delivery/scripts/medsam2_inference.py` | SAM2 ImagePredictor wrapper |
| `preprocessing.py` | `delivery/scripts/preprocessing.py` | Data preprocessing helpers |
| `torch_radon_impl.py` | `EM-reconstruction/torch_radon_impl.py` | Pure-PyTorch ParallelBeam (F.grid_sample; used as fallback) |
| `MydataLoader.py` | `EM-reconstruction/legacy_inactive/MydataLoader.py` | PyTorch Dataset (legacy) |

## How v3 uses these files

The v3 pipeline does **not modify** any of these files. The orchestrator
(`code/src/v3_canonical.py`) imports them by `sys.path.insert(0, str(REPO / "code/src/original"))`
and calls their public functions:

- `EM.MLEMorKEM(...)` — GPU MLEM/KEM via C++ torch_radon
- `recon.kem(...)`, `recon.mlem(...)`, `recon.osem(...)` — CPU versions
- `discrete_radon.build_pair(128, 180)` — forward operator
- `medsam2_decoder_infer.predict_prior(...)` — SAM prior
- `medsam2_decoder_train.train_one(...)` — decoder training
- `poisson_simulate.poisson(...)` — noise generation
- `metrics.psnr_db(...)`, `metrics.ssim_(...)` — metrics

## What was NOT copied

The original EM-reconstruction repo has ~307 .py files including many
build/run/experiment scripts that are not relevant to the v3
canonical pipeline. Only the 12 algorithm files above are included
here. The full original codebase remains at `D:/MIAProject/EM-reconstruction/`.

## Verifying integrity

Each file's SHA-256 was captured at copy time and can be verified
against the original with:

```bash
sha256sum code/src/original/*.py
# Compare with:
sha256sum D:/MIAProject/EM-reconstruction/{EM.py,torch_radon_impl.py}
sha256sum D:/MIAProject/EM-reconstruction/legacy_inactive/MydataLoader.py
sha256sum D:/MIAProject/SAM_KEM_QWEN3_MEDSAM2_DELIVERY/scripts/{recon,discrete_radon,poisson_simulate,metrics,build_medsam2_decoder_train,build_medsam2_decoder_infer,medsam2_decoder_train,medsam2_decoder_infer,medsam2_inference,preprocessing}.py
```

## Version provenance

These are the exact versions used by the v3 FINAL pipeline
(sha `c413227b944d1df81e9405353970ca68d36bcd81c42bc2d07d8a135be82c2d79`).
Any change to these files would require re-running the full pipeline
and re-validating.
