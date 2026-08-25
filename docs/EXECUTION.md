# Canonical execution guide

All commands are run from the repository root. Do not execute the historical
`code/scripts/_*.py` files: they contain the invalid v3 workflow and are retained
only for audit traceability.

## 1. Create and verify the environment

```bash
conda env create -f environment.yml
conda activate sam-kem
python -m pip install -e '.[test,evaluation]'
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest
```

Install the pinned MedSAM2/SAM2 source and the CUDA-compatible PyTorch build in a
separate GPU environment when training is required. Record `python --version`,
`pip freeze`, CUDA, GPU, Git commit, and checkpoint SHA-256 in every run directory.
Do not use the former Python 3.13 / “torch>=2.10” specification.

## 2. Resolve required provenance before running

Edit `code/configs/experiment.yaml` and fill:

- `normalization.scale_constants.brainweb`;
- `normalization.scale_constants.hecktor` (the scanner-derived value may be near
  30000, but the exact recorded value must be used);
- the exact MedSAM2 source repository and immutable Git commit.
- the released MedSAM2 checkpoint path and SHA-256 (no general-SAM fallback).

Inputs and targets are divided by their dataset constant C. No per-image or
dataset-wide min-max is permitted. Per-prior min-max is used only to build K.

Run the audit immediately to obtain the remaining-blocker list:

```bash
sam-kem validate --root . --config code/configs/experiment.yaml
```

At this preparation stage, missing trained checkpoints/results are expected. Do
not begin locked-test evaluation until every fatal item relevant to inputs,
splits, checkpoints, boxes, manifests and validation sweeps has been resolved.

## 3. Prepare cases and independent Poisson realizations

This is an idealized low-count simulation, not a scanner-faithful dose
simulation. Its observation is `y_i ~ Poisson(s [A x]_i)` with expected total
count `N`; independent count levels are separate draws. The model omits
attenuation, scatter, randoms, detector normalization, PSF, TOF and dead time.
For HECKTOR, `x` comes from a reconstructed clinical PET image and the generated
sinogram is therefore a pseudo-sinogram. Do not describe it as raw acquisition,
noise-free tracer activity, or a calibrated injected-dose reduction.

Use this canonical layout before building the manifests:

```text
arrays/<dataset>/<subject>/<slice>/target.npy
arrays/<dataset>/<subject>/<slice>/noise_free_projection.npy
arrays/<dataset>/<subject>/<slice>/noise_free_projection.json
arrays/<dataset>/<subject>/<slice>/anatomy.npy                 # baseline only
arrays/<dataset>/<subject>/<slice>/c<count>/r<r>/sinogram.npy
arrays/<dataset>/<subject>/<slice>/c<count>/r<r>/sinogram.json
arrays/<dataset>/<subject>/<slice>/c<count>/r<r>/mlem10.npy
arrays/<dataset>/<subject>/<slice>/c<count>/r<r>/mlem10.json
arrays/<dataset>/<subject>/<slice>/c<count>/r<r>/sam_prior.npy
```

The canonical protocol evaluates the full 128x128 field of view, so an
`evaluation_mask.npy` is prohibited. Do not derive the HECKTOR evaluation region
from CT or tumor masks. Anatomy arrays are inputs only to named MR/CT comparison
baselines.

The manifest contains one row per
`(dataset, subject_id, slice_id, count_level, realization_id)` and relative paths
to the target, noise-free projection, sinogram, optional MR/CT baseline image,
evaluation mask, MLEM-10 prompt image, and SAM prior. Use only whole-subject/source
model assignments from `subject_split_v3.csv`.

First divide each already-resampled 128x128 activity image by its fixed dataset C
and create the noise-free projection with the canonical operator:

```bash
sam-kem project-activity --root . --config code/configs/experiment.yaml \
  --dataset brainweb --activity source/subject01_axial_072.npy \
  --target-output arrays/brainweb/subject01/axial_072/target.npy \
  --projection-output arrays/brainweb/subject01/axial_072/noise_free_projection.npy
```

The command rejects non-negligible activity outside the declared inscribed
circular field of view and rejects a C that places the normalized target outside
the declared [0,1] metric range. Preprocessing that crops or masks activity must
be fixed and documented before this step; the command never silently truncates.

Then generate each low-count observation using its complete identity:

```bash
sam-kem simulate \
  --config code/configs/experiment.yaml \
  --projection arrays/.../noise_free_projection.npy \
  --output arrays/.../sinogram.npy \
  --dataset brainweb --subject subject01 --slice axial_072 \
  --count 200000 --realization 0
```

The command exclusively writes both `sinogram.npy` and `sinogram.json`. The JSON
contains the complete case identity, stable seed, projection/sinogram hashes,
expected and realized counts, count scale, Pearson dispersion and Poisson
deviance. The manifest builder verifies these values; do not copy them manually.

Generate MLEM-10 from that exact observation and the `projection_scale` recorded
in `sinogram.json`:

```bash
sam-kem prepare-mlem --root . --config code/configs/experiment.yaml \
  --sinogram arrays/.../sinogram.npy --output arrays/.../mlem10.npy \
  --projection-scale <value-from-sinogram.json>
```

This writes `mlem10.json`, binding the cached input to the sinogram hash, operator
SHA, gain, initialization and ten iterations. The manifest refuses an MLEM cache
without this sidecar, and evaluation also recomputes it numerically.

After target/projection/sinogram/MLEM arrays exist, build an input manifest:

```bash
sam-kem build-manifest --root . --config code/configs/experiment.yaml \
  --arrays arrays --output results/metrics/case_manifest_inputs.csv
```

## 4. Qwen boxes and MedSAM2 training

For every dataset/subject/slice/count, render the r=0 MLEM-10 array as an exact
128x128 PNG (display scaling only), then call Qwen3-VL-Plus once on that image
with the prompt “框选出所有明亮团块”. Cache the result under exactly that
four-field key, then reuse it for r=0/1/2 and every method. Record provider, exact
model, request ID, image SHA-256, prompt SHA-256, coordinates, parse status,
latency, and fallback status. A failed call uses the documented intensity-centroid
box; a zero-mass image uses the geometric center. Fallback cases remain marked as
fallback and must not be described as Qwen-guided.

Example single request and cache construction:

```bash
export DASHSCOPE_API_KEY='...'
sam-kem render-prompt --input arrays/.../mlem10.npy \
  --output prompts/images/brainweb_subject01_axial_072_c200000.png
sam-kem qwen-box --image prompts/images/brainweb_subject01_axial_072_c200000.png \
  --output prompts/responses/brainweb_subject01_axial_072_c200000.json \
  --dataset brainweb --subject subject01 --slice axial_072 --count 200000
sam-kem build-box-cache --manifest results/metrics/case_manifest_inputs.csv \
  --responses prompts/responses --output prompts/anchor_box_lookup_v2.csv \
  --config code/configs/experiment.yaml
```

Train two models: one BrainWeb model and one HECKTOR model. Within each dataset,
jointly sample all four count levels with a balanced sampler. Freeze image, prompt,
and memory encoders; train only the mask decoder with L1 loss. Training consumes
only train subjects and the cached boxes. It must write the actual subject IDs,
counts per epoch, checkpoint SHA, and frozen/trainable parameter inventory.

```bash
python code/experiments/train_decoder.py --dataset brainweb \
  --manifest results/metrics/case_manifest_inputs.csv \
  --pretrained checkpoints/MedSAM2.pt \
  --output checkpoints/medsam2_decoder_brainweb.pt \
  --epochs 20 --steps-per-epoch 400 --learning-rate 1e-4
python code/experiments/train_decoder.py --dataset hecktor \
  --manifest results/metrics/case_manifest_inputs.csv \
  --pretrained checkpoints/MedSAM2.pt \
  --output checkpoints/medsam2_decoder_hecktor.pt \
  --epochs 20 --steps-per-epoch 400 --learning-rate 1e-4
```

`steps-per-epoch` must be divisible by four. Inspect the adjacent JSON provenance
files and retain each `.resume.pt`, `.training.csv`, and CSV manifest. These are
fixed-budget final checkpoints, not validation-selected “best” checkpoints. To
resume an interrupted epoch-boundary run, pass the canonical file, for example
`--resume checkpoints/medsam2_decoder_brainweb.resume.pt`. Then enter both
resulting inference-checkpoint SHA-256 values and immutable archive URIs in
`experiment.yaml`. At that
point set `status: locked_for_execution` and `locked: true`, commit the clean
configuration, and do not change it during prior generation or evaluation.

Generate validation/test priors, then build the final immutable manifest so its
prior paths are populated:

```bash
python code/experiments/generate_priors.py \
  --manifest results/metrics/case_manifest_inputs.csv \
  --splits validation test_locked
sam-kem build-manifest --root . --config code/configs/experiment.yaml \
  --arrays arrays --output results/metrics/case_manifest_v2.csv
```

## 5. Run experiments in the required order

Validate every protocol before starting it:

```bash
for p in code/configs/protocols/*.yaml; do
  sam-kem check-protocol --protocol "$p"
done
```

Execution order is:

1. `training.yaml`: train-set only, one four-count-level model per dataset.
2. `hyperparameter_sweep.yaml`: validation only; select MLEM stopping iterations
   and OSEM complete subset cycles, and separately sweep sigma, radius and KEM
   iterations for Bowsher-KEM, SelfKEM, MR/CT-KEM and SAM-KEM. Sweep alpha only
   for SAM-KEM using real priors. Extend every non-physical grid boundary when
   selected. Freeze one setting per dataset/count/method. The fixed MLEM-10 used
   to create prompts is not reused as the MLEM baseline unless validation selects
   ten iterations independently.
3. `primary.yaml`: locked test, only after all choices are frozen.
4. `comparison.yaml`: same locked cases and realizations for all applicable
   baselines; MR-KEM is BrainWeb-only and CT-KEM is HECKTOR-only.
   `Bowsher-KEM` is specifically the anatomy-selected-neighbour coefficient KEM
   implemented here, not classic MAP-Bowsher. Report OSEM work as both complete
   subset cycles and total subset updates; do not imply equal runtime from equal
   outer-iteration numbers.
5. `ablation.yaml`: real one-factor-at-a-time reruns. Never alias comparison rows
   as ablations.
6. `convergence.yaml`: descriptive locked-test curves only. Never select iteration
   count from these curves.
7. `cross_count.yaml`: aggregate each dataset/count/method key exactly once.
8. `runtime.yaml`: validation cases, cached-prior reconstruction scope, with raw
   warmup/timed repetitions; VLM/network latency is reported separately from logs.

Canonical commands after data/prior preparation are:

```bash
python code/experiments/run_hyperparameter_sweep.py
python code/experiments/run_primary.py
python code/experiments/run_comparison.py
python code/experiments/run_ablation.py
python code/experiments/run_convergence.py
python code/experiments/run_cross_count.py
python code/experiments/run_runtime.py
```

All writers use exclusive creation and refuse to overwrite an existing run.
Create a new run directory/config ID for every rerun; never append mixed runs.
Review and commit each completed run's CSV/JSON provenance before starting the
next protocol so the next run also starts from a clean worktree.
Every result CSV has a schema/SHA sidecar. A run is eligible only after its
`_SUCCESS.json` is written. See `docs/ARTIFACT_SPEC.md` for exact checkpoint and
result primary-key requirements. Before final validation, archive the complete
bundle and fill `outputs.artifact_policy.immutable_archive_uri`.

## 6. Final validation and paper eligibility

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest
sam-kem validate --root . --config code/configs/experiment.yaml
git status --short
```

Archive configuration, strict protocol files, split manifest, box records, raw
arrays, full sweeps, training logs, checkpoints, environment lock, per-case output,
and validation output together. Existing `results/*_v3.csv` files remain
AUDIT_ONLY and must not be copied into the paper.
