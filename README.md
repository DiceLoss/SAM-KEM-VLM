# SAM-KEM: auditable PET reconstruction code

This repository contains the clean, fail-closed implementation of the SAM-KEM
PET reconstruction study and a separately identified forensic snapshot of the
invalid v3 workflow.
Only `code/sam_kem/`, `code/experiments/`,
`code/configs/experiment.yaml`, and `code/configs/protocols/` are active
experiment sources.
The base configuration is development-only: its SHA-pinned split has train and
validation subjects but deliberately no locked test. Normalization calibration
and GPU decoder training may proceed; the release validator remains non-zero
until an external untouched test and all final artifacts are supplied.

Start with the [documentation index](docs/README.md). GPU workers must follow
the [Chinese GPU execution guide](docs/GPU_WORKER_START_ZH.md) exactly.

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
- full-square PET-operator support for both KEM neighbourhoods and image metrics, with no
  CT/tumor-derived evaluation mask;
- exact per-case primary keys and fail-closed validation.

The PET simulator is an idealized full-square, 180-angle/182-bin parallel-beam low-count model,
not a scanner-faithful or injected-dose simulator. HECKTOR projections are
pseudo-sinograms generated from reconstructed clinical PET. See the explicit
claim boundary in [SCIENTIFIC_STATUS.md](SCIENTIFIC_STATUS.md).

The current scientific status is recorded in
[SCIENTIFIC_STATUS.md](SCIENTIFIC_STATUS.md).
The old CSV results are audit evidence only and must not be cited.

## Quick start

```bash
conda env create -f environment.yml
conda activate sam-kem
python -m pip install -e '.[test,evaluation]'
make quality
sam-kem validate --root . --config code/configs/experiment.yaml
```

The validator is expected to fail during development. The official MedSAM2
source and released checkpoint identity are already pinned; remaining release
requirements include an external unexposed test, selected fixed-scale
provenance, retrained decoders, complete Qwen caches, validation sweeps and an
immutable artifact archive.

The last command is a scientific release gate, not a smoke test.
Its failure is expected in this checkout; do not suppress it or replace missing
values with guesses.

Experiment definitions are separate and reviewable:

- `code/configs/protocols/normalization_calibration.yaml`
- `code/configs/protocols/training.yaml`
- `code/configs/protocols/hyperparameter_sweep.yaml`
- `code/configs/protocols/primary.yaml`
- `code/configs/protocols/comparison.yaml`
- `code/configs/protocols/ablation.yaml`
- `code/configs/protocols/convergence.yaml`
- `code/configs/protocols/cross_count.yaml`
- `code/configs/protocols/runtime.yaml`

See the [execution guide](docs/EXECUTION.md) for the exact order, required
manifests, and paper-eligibility gate.

## Repository layout

```text
code/sam_kem/                 canonical scientific library and CLI
code/experiments/             thin active experiment entry points
code/configs/experiment.yaml strict experiment contract
code/configs/protocols/       experiment-specific protocols
tests/                        synthetic mathematical and integrity tests
code/src/original/            historical source snapshot (audit only)
code/scripts/_*.py            invalid v3 drivers (audit only)
data/raw_samples/             small traceability exports (diagnostic only)
data/raw_pet/, data/samples/  larger historical uploaded samples (diagnostic only)
results/*_v3.csv              historical v3 artifacts (audit only)
results/clean_v1/             future tracked clean manifests and summaries
runs/clean_v1/                future immutable experiment run directories
```

Do not suppress a failed validator, silently fall back to a pretrained
checkpoint, select parameters from `test_locked`, or edit result CSV values by
hand. Scientific corrections require rerunning the affected experiment.

## Documentation map

- [Chinese start guide](docs/START_HERE_ZH.md): the exact first-stage commands,
  required provenance package, stopping point, and prohibited actions.
- [Repository map](docs/REPOSITORY_MAP.md): which files are active, diagnostic,
  forensic, generated, or externally archived.
- [Data dictionary](docs/DATA_DICTIONARY.md): source manifests, units, geometry,
  case identities, arrays, and sidecars.
- [Experiment matrix](docs/EXPERIMENT_MATRIX.md): datasets, methods, counts,
  splits, selection rules, and output dependencies.
- [Execution guide](docs/EXECUTION.md): complete command order from ingestion to
  release.
- [Operations runbook](docs/OPERATIONS_RUNBOOK.md): safe resume, failure
  quarantine, archival, and handoff checks.
- [Artifact specification](docs/ARTIFACT_SPEC.md): immutable CSV/checkpoint
  schemas and completion rules.
- [PET and method audit](docs/PET_AND_METHOD_AUDIT.md): scientific claim boundary
  and retired implementation problems.
