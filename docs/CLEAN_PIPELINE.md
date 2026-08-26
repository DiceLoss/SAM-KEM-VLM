# Clean pipeline boundary

Only `code/sam_kem/`, `code/experiments/`, `code/configs/experiment.yaml`, and
the protocol YAML files are executable sources for the new study.
`code/src/original/`, underscore-prefixed files in `code/scripts/`, v3 result
files, and uploaded historical checkpoints are forensic material only. The
repository audit fails if a clean module imports the retired implementation.

## Source ingestion

Generate and SHA-pin `inputs/clean_v1/subject_split.csv` with `sam-kem
create-split` before ingesting data. The generator assigns whole subjects,
balances the predeclared `stratum` column, and refuses to draw locked-test
subjects from records previously used for development or reporting. The
historical v3 test is exposed and is never accepted as the clean holdout.

Copy `inputs/source_activity_manifest.template.csv` to
`inputs/clean_v1/source_activity_manifest.csv` and fill
one row per unique source slice. Every row pins the source file SHA-256, split,
activity unit, source indices, and one approved spatial transform.

```bash
python -m pip install -e '.[data,test,evaluation]'
sam-kem ingest-activity --root . \
  --config code/configs/experiment.yaml \
  --manifest inputs/clean_v1/source_activity_manifest.csv \
  --output arrays/clean_v1 --splits train validation
```

Approved transforms are deliberately narrow:

- `mean_pool_2x2_full_square`: exact 256-to-128 BrainWeb mean pooling;
- `identity_128_full_square`: already registered 128x128 activity;
- `resize_full_square_to_128`: pixel-centre-aligned bilinear resizing of the
  complete source square without a content crop, recentering, or corner mask.

The ingest command does not normalize intensity, infer a channel, crop a body,
read a segmentation target, or use the locked test split by default. It writes
`raw_activity.npy` plus `raw_activity.json`. The sidecar binds the output to the
source file/array SHA, exact selector, transform, units, and optional source spacing.

The HECKTOR row must keep `source_subject_index = subject_id - 1` and use channel 1
only after the source manifest declares the verified DICOM-export PET unit.
The full 144x144 square is resized to 128x128. Scanner FOV and scanner-axis
coordinates are not inputs to this explicitly pixel-domain simulation.

## Simulation and reconstruction

`sam-kem project-activity` accepts only an ingested `raw_activity.npy` with its
matching sidecar. It applies the configured fixed dataset scale, then calls the
single operator in `code/sam_kem/pet_operator.py`. That same exact
forward/adjoint pair is used by MLEM, OSEM, KEM, simulation, and Poisson
deviance. No clean module imports a Radon or reconstruction function from the
historical source tree.

MedSAM2 is built directly from the configured upstream `sam2` checkout and the
exact declared checkpoint. There is no checkpoint search path and no fallback
to general SAM2 weights. Image, prompt, and memory encoders remain frozen;
training updates only `sam_mask_decoder` with L1 loss, balanced across all four
count levels in one dataset-specific model.

## Required execution order

1. Generate, inspect, SHA-pin, and commit a new subject-level split.
2. Ingest train and validation sources.
3. Fit or verify fixed-scale candidates on train; inspect validation only.
4. Freeze the spatial transform and normalization identity.
5. Generate matched projections, independent Poisson observations, and MLEM-10.
6. Cache one Qwen box per dataset/subject/slice/count from realization 0.
7. Train one decoder per dataset using all dose levels.
8. Run the full validation sweep and expand selected grid boundaries.
9. Freeze all hyperparameters and checkpoints in a clean Git commit.
10. Release the untouched locked test for the predeclared comparison, ablation,
   convergence, and derived-primary suite; never use those outputs for tuning.
11. Aggregate by subject and use paired subject-level inference.

Every scientific run starts from a clean commit and ends with immutable files,
SHA-bound sidecars, unique CSV primary keys, an environment freeze, and a
`_SUCCESS.json`. Partial directories and historical result tables are never
paper-eligible.
