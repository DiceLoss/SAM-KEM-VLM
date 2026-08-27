# Canonical execution guide

All commands are run from the repository root. Do not execute the historical
`code/scripts/_*.py` files: they contain the invalid v3 workflow and are retained
only for audit traceability.

## 1. Create and verify the environment

```bash
conda env create -f environment.yml
conda activate sam-kem
python -m pip install -e '.[data,test,evaluation]'
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest
```

Install the pinned MedSAM2/SAM2 source and the CUDA-compatible PyTorch build in a
separate GPU environment when training is required. Record `python --version`,
`pip freeze`, CUDA, GPU, Git commit, and checkpoint SHA-256 in every run directory.
Do not use the former Python 3.13 / “torch>=2.10” specification.

## 2. Resolve required provenance before running

Edit `code/configs/experiment.yaml` and fill:

- a new SHA-pinned subject split whose locked-test rows have never appeared in
  historical training, validation sweeps, tuning, or reported results;
- `normalization.scale_constants.brainweb`;
- `normalization.scale_constants.hecktor`, in the same unit as the ingested PET
  array. A scanner-count constant near 30000 cannot be applied to an SUV array;
  uploaded SUV samples support only provisional train/validation candidates,
  not a final physical constant;
- the exact MedSAM2 source repository and immutable Git commit.
- the released MedSAM2 checkpoint path and SHA-256 (no general-SAM fallback).

The canonical primary arm divides inputs and targets by its dataset constant C.
Per-image min-max is prohibited as a primary scale. Train-only global min-max is
permitted only as an explicitly identified sensitivity arm. Per-prior min-max is
used separately to build K.

If either C is genuinely unknown, do not guess it and do not inspect the locked
test set. Complete the subject split and train/validation ingestion in Section 3
first. Then copy `inputs/normalization_activity_manifest.template.csv` to
`inputs/clean_v1/normalization_activity_manifest.csv`, list each unique
ingested `raw_activity.npy` slice once, retain its native unit, and record its
array SHA-256. Run the train-only scale study from a clean committed worktree:

```bash
sam-kem array-sha --input arrays/clean_v1/brainweb/subject04/z125/raw_activity.npy
sam-kem calibrate-normalization --root . \
  --config code/configs/experiment.yaml \
  --protocol code/configs/protocols/normalization_calibration.yaml
```

It records four declared possibilities:

1. scanner/simulation-derived physical C, when its source is recorded;
2. zero-offset fixed C equal to the training-set global maximum;
3. training-set global min-max, with validation values clipped only for this
   sensitivity analysis;
4. per-image min-max as a diagnostic only.

The command never opens `test_locked` arrays. It writes candidate parameters,
per-validation-case clipping diagnostics, SHA manifests and `_SUCCESS.json`.
The training-maximum candidate is a fixed data-derived scale, not a physical
scale. Global min-max changes the zero point when the training minimum is not
zero. Per-image min-max destroys between-case quantitative comparability and is
never eligible as the primary experiment.

Qwen requests use the separately SHA-pinned
`inputs/clean_v1/qwen_request_policy.json`. Every cache key receives the same
three-attempt budget and fixed backoff schedule for retryable errors. Every
failed and successful attempt is archived. Ten consecutive cache keys that
exhaust the live retry budget trip a circuit breaker, leave the batch incomplete,
and require a later verified resume. Non-retryable HTTP errors such as invalid
authentication abort the entire batch after one archived attempt and can never
be converted into methodological fallback boxes.

Distribution diagnostics do not select a reconstruction method. A normalization
arm needs separate training only when its transformation changes at least one
training input. In this calibration, both training minima are zero, so
train-global min-max and train-maximum fixed C are identical on every training
image. The clipped arm is therefore retained as a validation diagnostic rather
than used to train duplicate decoders. The audited development configuration uses
the non-clipping train-maximum fixed C; this is a data-derived fixed scale, not a
physical constant. Checkpoint and result paths include the normalization ID so
artifacts cannot be mixed.

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

First create the clean subject split. The historical `subject_split_v3.csv`
cannot be reused because its test subjects were opened by the invalid v3 sweep.
Fill `inputs/subject_inventory.template.csv`; use `stratum` for a predeclared
scanner/site or simulation-family balancing variable, not an outcome selected
after looking at performance. Only subjects declared
`never_used_for_training_validation_tuning_or_reporting` are eligible for the
new test set.

```bash
sam-kem create-split --root . --config code/configs/experiment.yaml \
  --inventory inputs/clean_v1/subject_inventory.csv \
  --output inputs/clean_v1/subject_split.csv
sam-kem file-sha --input inputs/clean_v1/subject_split.csv
```

Enter that SHA as `data.split_sha256`, inspect the 14/3/3 BrainWeb and
156/34/34 HECKTOR subject counts, then commit the split before ingestion. If
there are not enough genuinely unexposed subjects, use an external holdout or
state that the study has no untouched test set; do not relabel an exposed
subject.

Next create `inputs/clean_v1/source_activity_manifest.csv` from the template. Source
channel, subject/slice selectors, units, SHA-256 and the fixed spatial transform
must all be explicit. Ingest train and validation without opening the locked test:

```bash
sam-kem ingest-activity --root . \
  --manifest inputs/clean_v1/source_activity_manifest.csv \
  --output arrays/clean_v1 --splits train validation
```

BrainWeb uses the exact `mean_pool_2x2_full_square` transform. HECKTOR uses
`resize_full_square_to_128`, a fixed pixel-centre-aligned bilinear resize of the
complete DICOM-exported 144x144 reconstructed PET slice. It does not infer a
scanner FOV, crop by image content, recenter anatomy, or mask the corners.
DICOM spacing/orientation tags should be retained when available for provenance,
but they do not parameterize this declared pixel-domain operator.

Use this canonical layout before building the manifests:

```text
arrays/clean_v1/<dataset>/<subject>/<slice>/raw_activity.npy
arrays/clean_v1/<dataset>/<subject>/<slice>/raw_activity.json
arrays/clean_v1/<dataset>/<subject>/<slice>/target.npy
arrays/clean_v1/<dataset>/<subject>/<slice>/noise_free_projection.npy
arrays/clean_v1/<dataset>/<subject>/<slice>/noise_free_projection.json
arrays/clean_v1/<dataset>/<subject>/<slice>/anatomy.npy        # baseline only
arrays/clean_v1/<dataset>/<subject>/<slice>/c<count>/r<r>/sinogram.npy
arrays/clean_v1/<dataset>/<subject>/<slice>/c<count>/r<r>/sinogram.json
arrays/clean_v1/<dataset>/<subject>/<slice>/c<count>/r<r>/mlem10.npy
arrays/clean_v1/<dataset>/<subject>/<slice>/c<count>/r<r>/mlem10.json
arrays/clean_v1/<dataset>/<subject>/<slice>/c<count>/r<r>/sam_prior.npy
```

The canonical protocol evaluates the full-square support defined by positive PET
operator sensitivity. This same geometry-only support is used for every method;
an external `evaluation_mask.npy` is prohibited. Do not derive the HECKTOR
evaluation region from CT or tumor masks. Anatomy arrays are inputs only to
named MR/CT comparison baselines.

The manifest contains one row per
`(dataset, subject_id, slice_id, count_level, realization_id)` and relative paths
to the target, noise-free projection, sinogram, optional MR/CT baseline image,
evaluation mask, MLEM-10 prompt image, and SAM prior. Use only whole-subject/source
model assignments from the canonical subject split referenced by the experiment
configuration. Historical `subject_split_v3.csv` and `subject_split_v4.csv` are
forensic inputs only and are not executable clean-pipeline splits.

Normally prepare every ingested train/validation case in one verified batch:

```bash
sam-kem prepare-cases --root . --config code/configs/experiment.yaml \
  --arrays arrays/clean_v1 --splits train validation
```

The batch constructs each fixed-C target, matched noise-free projection,
independent count/realization sinogram, and MLEM-10 input. A rerun refuses all
existing outputs unless `--resume-verified` is supplied; that mode reopens and
hash-verifies every complete artifact group and still rejects partial groups.
The locked test split is refused until the configuration is fully locked.

For single-case diagnosis, divide one already-ingested 128x128 activity image by
its fixed dataset C and create the noise-free projection with:

```bash
sam-kem project-activity --root . --config code/configs/experiment.yaml \
  --dataset brainweb --activity arrays/clean_v1/brainweb/subject01/axial_072/raw_activity.npy \
  --target-output arrays/clean_v1/brainweb/subject01/axial_072/target.npy \
  --projection-output arrays/clean_v1/brainweb/subject01/axial_072/noise_free_projection.npy
```

The command rejects a C that places the normalized target outside the declared
[0,1] metric range. The operator covers every pixel in the complete 128x128
square using 182 detector bins; the command never silently crops or masks activity.

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
  --arrays arrays/clean_v1 --output results/clean_v1/case_manifest_inputs.csv
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

Generate and request every train/validation key as a verified batch. A resumed
batch reuses archived attempts—including failures—and never selectively retries
cases based on their result:

```bash
export DASHSCOPE_API_KEY='...'
sam-kem prepare-prompt-images --root . --config code/configs/experiment.yaml \
  --manifest results/clean_v1/case_manifest_inputs.csv \
  --output prompts/clean_v1/train_validation_images \
  --splits train validation
sam-kem request-boxes --root . --config code/configs/experiment.yaml \
  --manifest prompts/clean_v1/train_validation_images/prompt_manifest.csv \
  --responses prompts/clean_v1/train_validation_responses \
  --policy inputs/clean_v1/qwen_request_policy.json --resume-verified
sam-kem build-box-cache --manifest results/clean_v1/case_manifest_inputs.csv \
  --responses prompts/clean_v1/train_validation_responses \
  --output prompts/clean_v1/train_validation_box_cache.csv \
  --policy inputs/clean_v1/qwen_request_policy.json \
  --config code/configs/experiment.yaml
```

Train two models: one BrainWeb model and one HECKTOR model. Within each dataset,
jointly sample all four count levels with a balanced sampler. Freeze image, prompt,
and memory encoders; train only the mask decoder with L1 loss. Training consumes
only train subjects and the cached boxes. It must write the actual subject IDs,
counts per epoch, checkpoint SHA, and frozen/trainable parameter inventory.

```bash
python code/experiments/train_decoder.py --dataset brainweb \
  --manifest results/clean_v1/case_manifest_inputs.csv \
  --pretrained checkpoints/MedSAM2.pt \
  --output checkpoints/clean_v1/medsam2_decoder_brainweb.pt \
  --epochs 20 --steps-per-epoch 400 --learning-rate 1e-4
python code/experiments/train_decoder.py --dataset hecktor \
  --manifest results/clean_v1/case_manifest_inputs.csv \
  --pretrained checkpoints/MedSAM2.pt \
  --output checkpoints/clean_v1/medsam2_decoder_hecktor.pt \
  --epochs 20 --steps-per-epoch 400 --learning-rate 1e-4
```

`steps-per-epoch` must be divisible by four. Inspect the adjacent JSON provenance
files and retain each `.resume.pt`, `.training.csv`, and CSV manifest. These are
fixed-budget final checkpoints, not validation-selected “best” checkpoints. To
resume an interrupted epoch-boundary run, pass the canonical file, for example
`--resume checkpoints/clean_v1/medsam2_decoder_brainweb.resume.pt`. Then enter both
resulting inference-checkpoint SHA-256 values and immutable archive URIs in
`experiment.yaml`. At that
point set `status: locked_for_execution` and `locked: true`, commit the clean
configuration, and do not change it during prior generation or evaluation.

Generate validation priors only, then build the immutable validation manifest
used by the sweep:

```bash
python code/experiments/generate_priors.py \
  --manifest results/clean_v1/case_manifest_inputs.csv \
  --splits validation
sam-kem build-manifest --root . --config code/configs/experiment.yaml \
  --arrays arrays/clean_v1 --output results/clean_v1/validation_case_manifest.csv
```

Run and freeze the complete validation sweep before any locked-test ingestion.
The code verifies the sweep CSV, frozen-parameter CSV and `_SUCCESS.json` before
allowing `test_locked` access. After release, ingest and prepare the test cases,
archive their Qwen responses, and build an all-split box-input manifest:

```bash
sam-kem ingest-activity --root . --config code/configs/experiment.yaml \
  --manifest inputs/clean_v1/source_activity_manifest.csv \
  --output arrays/clean_v1 --splits test_locked
sam-kem prepare-cases --root . --config code/configs/experiment.yaml \
  --arrays arrays/clean_v1 --splits test_locked --resume-verified
sam-kem build-manifest --root . --config code/configs/experiment.yaml \
  --arrays arrays/clean_v1 --output results/clean_v1/final_box_inputs.csv
sam-kem prepare-prompt-images --root . --config code/configs/experiment.yaml \
  --manifest results/clean_v1/final_box_inputs.csv \
  --output prompts/clean_v1/test_images --splits test_locked
sam-kem request-boxes --root . --config code/configs/experiment.yaml \
  --manifest prompts/clean_v1/test_images/prompt_manifest.csv \
  --responses prompts/clean_v1/test_responses \
  --policy inputs/clean_v1/qwen_request_policy.json --resume-verified
sam-kem build-box-cache --manifest results/clean_v1/final_box_inputs.csv \
  --responses prompts/clean_v1/train_validation_responses \
              prompts/clean_v1/test_responses \
  --output prompts/clean_v1/final_box_cache.csv \
  --policy inputs/clean_v1/qwen_request_policy.json \
  --config code/configs/experiment.yaml
python code/experiments/generate_priors.py \
  --manifest results/clean_v1/final_box_inputs.csv --splits test_locked
sam-kem build-manifest --root . --config code/configs/experiment.yaml \
  --arrays arrays/clean_v1 --output results/clean_v1/case_manifest.csv
```

Do not generate test boxes or priors alongside validation. The two immutable
caches prevent post-test boxes from changing the inputs used for decoder
training or validation selection.

## 5. Run experiments in the required order

Validate every protocol before starting it:

```bash
for p in code/configs/protocols/*.yaml; do
  sam-kem check-protocol --protocol "$p"
done
```

Execution order is:

1. `normalization_calibration.yaml`, when C is unresolved: fit candidates on
   train and examine distributions on validation only. Any performance-based
   choice requires separate retraining and validation reruns per candidate.
2. `training.yaml`: train-set only, one four-count-level model per dataset and
   normalization candidate.
3. `hyperparameter_sweep.yaml`: validation only; select MLEM stopping iterations
   and OSEM complete subset cycles, and separately sweep sigma, radius and KEM
   iterations for Bowsher-KEM, SelfKEM, MR/CT-KEM and SAM-KEM. Sweep alpha only
   for SAM-KEM using real priors. Extend every non-physical grid boundary when
   selected. Freeze one setting per dataset/count/method. The fixed MLEM-10 used
   to create prompts is not reused as the MLEM baseline unless validation selects
   ten iterations independently.
4. `comparison.yaml`: open the locked test only after all choices are frozen;
   compute the unique per-case reconstruction artifact on the same cases and
   realizations for all applicable
   baselines; MR-KEM is BrainWeb-only and CT-KEM is HECKTOR-only.
   `Bowsher-KEM` is specifically the anatomy-selected-neighbour coefficient KEM
   implemented here, not classic MAP-Bowsher. Report OSEM work as both complete
   subset cycles and total subset updates; do not imply equal runtime from equal
   outer-iteration numbers.
5. `primary.yaml`: derive the 200k headline rows from the immutable comparison
   artifact. It performs no second reconstruction or second test-data read.
6. `ablation.yaml`: real one-factor-at-a-time reruns. Never alias comparison rows
   as ablations.
7. `convergence.yaml`: descriptive locked-test curves only. Never select iteration
   count from these curves.
8. `cross_count.yaml`: aggregate each dataset/count/method key exactly once.
9. `runtime.yaml`: validation cases, cached-prior reconstruction scope, with raw
   warmup/timed repetitions; VLM/network latency is reported separately from logs.

Canonical commands after data/prior preparation are:

```bash
python code/experiments/run_hyperparameter_sweep.py
python code/experiments/run_comparison.py
python code/experiments/run_primary.py
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

The repository intentionally ignores bulk `arrays/clean_v1` data, rendered PNG
pixels, and `*.pt` files. Copy them to the declared immutable archive before
continuing; commit their SHA-bound JSON/CSV provenance. In particular, the
worktree must be clean at each of these gates:

1. before BrainWeb training (commit split, input manifests, prompt records/cache);
2. before HECKTOR training (archive the BrainWeb checkpoint and commit its
   training CSV/provenance);
3. before the validation sweep (archive both checkpoints and validation priors,
   then commit all SHA manifests and the locked configuration);
4. before comparison (archive test arrays/PNGs/priors and commit the final box
   cache and case manifest);
5. before every subsequent result protocol (commit the prior completed run).

Do not use `git add -f` to put bulk patient-derived arrays into GitHub. Git
cleanliness proves the code/config/manifests were committed; array integrity is
proved separately by the recorded SHA-256 values and immutable archive URI.

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
