# SAM-KEM: auditable PET reconstruction code

This repository is being scientifically refactored after an audit found that the
previous v3 results were not reproducible paper evidence. The canonical code now
lives in `code/sam_kem/`. Historical `code/src/original/` and
`code/scripts/_*.py` files are preserved only to document how the invalid v3
artifacts were produced; they are not execution entry points.

## What the canonical core enforces

- subject/source-model split isolation;
- one matched 180-angle forward/adjoint operator for simulation,
  reconstruction, and Poisson deviance;
- provenance-bound pseudo-sinogram, Poisson observation, and MLEM-10 caches;
- independent random streams keyed by dataset, subject, slice, count, and
  realization;
- fixed dataset-specific physical scale constants for inputs and targets;
- true coefficient-space KEM with `x = K alpha`;
- Qwen box cache keyed by dataset/subject/slice/count and shared across
  realizations;
- immutable batch prompt/request manifests that preserve failed calls and
  prohibit selective retries;
- dataset-specific MedSAM2 checkpoints cached by verified SHA;
- frozen encoders, trainable mask decoder, L1-only training;
- continuous raw decoder output, with min-max restricted to kernel features;
- validation-only hyperparameter selection and immutable locked-test evaluation;
- PET-operator support for both KEM neighbourhoods and image metrics, with no
  CT/tumor-derived evaluation mask;
- exact per-case primary keys and fail-closed validation.

The PET simulator is an idealized circular-FOV parallel-beam low-count model,
not a scanner-faithful or injected-dose simulator. HECKTOR projections are
pseudo-sinograms generated from reconstructed clinical PET. See the explicit
claim boundary in [SCIENTIFIC_STATUS.md](SCIENTIFIC_STATUS.md).

The current scientific status is recorded in [SCIENTIFIC_STATUS.md](SCIENTIFIC_STATUS.md).
The old CSV results are audit evidence only and must not be cited.
The detailed PET/model audit for the experimenter is
[docs/PET_AND_METHOD_AUDIT.md](docs/PET_AND_METHOD_AUDIT.md).
The black-box audit of the uploaded sample NPZ files is
[docs/UPLOADED_SAMPLE_AUDIT.md](docs/UPLOADED_SAMPLE_AUDIT.md).
Result/checkpoint storage requirements are in
[docs/ARTIFACT_SPEC.md](docs/ARTIFACT_SPEC.md).

## Quick start

```bash
conda env create -f environment.yml
conda activate sam-kem
python -m pip install -e '.[test,evaluation]'
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest
sam-kem validate --root . --config code/configs/experiment.yaml
```

The validator is expected to fail until the experimenter records the exact
BrainWeb and HECKTOR physical scale constants, supplies verified decoder
checkpoints, regenerates complete Qwen caches, and reruns all experiments.

Experiment definitions are separate and reviewable:

- `code/configs/protocols/normalization_calibration.yaml`
- `code/configs/protocols/training.yaml`
- `code/configs/protocols/hyperparameter_sweep.yaml`
- `code/configs/protocols/primary.yaml`
- `code/configs/protocols/comparison.yaml`
- `code/configs/protocols/ablation.yaml`
- `code/configs/protocols/convergence.yaml`
- `code/configs/protocols/cross_count.yaml`

See [docs/EXECUTION.md](docs/EXECUTION.md) for the exact order, required
manifests, and paper-eligibility gate.

## Repository layout

```text
code/sam_kem/                 canonical scientific library and CLI
code/configs/experiment.yaml strict experiment contract
code/configs/protocols/       experiment-specific protocols
tests/                        synthetic mathematical and integrity tests
code/src/original/            historical source snapshot (audit only)
code/scripts/_*.py            invalid v3 drivers (audit only)
results/                      historical v3 artifacts (audit only until rerun)
prompts/                      historical evidence and clean_v1 prompt archives
```

Do not suppress a failed validator, silently fall back to a pretrained
checkpoint, select parameters from `test_locked`, or edit result CSV values by
hand. Scientific corrections require rerunning the affected experiment.
