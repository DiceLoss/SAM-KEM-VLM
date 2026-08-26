# Experiment result and checkpoint artifact specification

## Non-negotiable rules

1. A scientific artifact is immutable. Existing final CSV, checkpoint,
   provenance or `_SUCCESS.json` files are never overwritten.
2. CSV and final checkpoint publication is atomic: a temporary file is flushed
   and renamed only after successful completion.
3. Every CSV has an adjacent `<name>.csv.manifest.json` containing its schema,
   columns, primary key, row count, byte size and SHA-256.
4. A run is complete only when `_SUCCESS.json` exists and all files listed in it
   still match their size and SHA-256. A directory without this marker is a
   partial/failed run and is ineligible for the paper.
5. Runs start only from a clean Git worktree and record code/config/protocol,
   Python dependencies, CUDA/cuDNN and GPU identity.
6. A rerun uses a new protocol output directory. Never append a new run to an
   old CSV and never manually edit a generated table.

## Decoder checkpoint bundle

For an output such as `checkpoints/medsam2_decoder_brainweb.pt`, training writes:

| File | Meaning |
|---|---|
| `medsam2_decoder_brainweb.pt` | Immutable inference checkpoint containing only `mask_decoder_state`, dataset and normalization identity, exact base checkpoint SHA and fixed-budget selection statement. |
| `medsam2_decoder_brainweb.resume.pt` | Atomic rolling epoch-boundary recovery state containing decoder, optimizer, Python/NumPy/Torch CPU/CUDA RNG states, completed epoch/global step and sampling counters. |
| `medsam2_decoder_brainweb.training.csv` | One unique row per completed epoch with mean/std/min/max L1 and sample count. |
| `medsam2_decoder_brainweb.training.csv.manifest.json` | Training-curve schema, row count and SHA-256. |
| `medsam2_decoder_brainweb.json` | Final provenance joining all hashes, Git commits, subjects, count balance, parameter inventory and training protocol. |

The inference checkpoint deliberately excludes frozen encoders. At inference,
the verified official MedSAM2 base checkpoint is loaded first and only the
dataset-specific mask decoder is applied. This prevents an HECKTOR decoder from
being silently interpreted as a BrainWeb/full-model checkpoint.

Training currently uses a predeclared fixed budget; therefore the final file is
called a **final fixed-budget checkpoint**, never a “best checkpoint.” A “best”
label is permitted only after a separately specified validation selector and
metric are implemented and archived. Locked-test metrics can never select a
checkpoint.

Resume is supported only from the canonical `.resume.pt` and only when dataset,
base checkpoint, configuration and completed-epoch identity match. The rolling
resume state may be atomically replaced; immutable inference/checksum artifacts
may not.

Both decoder bundles must be copied to an immutable external archive or release.
Record its URI in `medsam2.checkpoint_archive_uri` while retaining the local
SHA-verified file for execution.

## Result schemas and primary keys

| Artifact | Primary key |
|---|---|
| Per-case reconstruction metrics | `(protocol,dataset,subject_id,slice_id,realization_id,count_level,method)` |
| Validation sweep | `(dataset,subject_id,slice_id,realization_id,count_level,method,sigma,radius,neighbours,iterations,alpha)` |
| Frozen hyperparameters | `(dataset,count_level,method)` |
| Ablation | `(dataset,subject_id,slice_id,realization_id,count_level,variant)` |
| Convergence | `(dataset,subject_id,slice_id,realization_id,count_level,method,iteration)` |
| Runtime | `(dataset,subject_id,slice_id,realization_id,count_level,method,repetition)` |
| Cross-count summary | `(dataset,count_level,method)` |
| Paired statistics | `(dataset,count_level,proposed_method,baseline_method)` |

All per-case metrics must be finite, use the declared fixed physical scale and
carry operator/checkpoint/seed/hyperparameter identity. Subject-level statistics
are derived from per-case rows; aggregate rows are never copied back into the
per-case table. Frozen parameters are generated from validation sweep output and
are not entered by hand.

Final run directories, checkpoint bundles, array manifests, prompt records and
locked configuration must be archived together. Record the immutable result
archive URI in `outputs.artifact_policy.immutable_archive_uri` before manuscript
eligibility validation.

Normalization candidates are separate experiment identities. Their ID must be
present in the run directory, checkpoint provenance and regenerated array
metadata. A checkpoint trained under fixed C, global min-max or per-image
min-max may never be evaluated against arrays from another normalization arm.
The calibration CSV and its input activity manifest are archived with the same
SHA-bound rules as reconstruction results.

## Completion checks

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest -q
sam-kem validate --root . --config code/configs/experiment.yaml
```

The validator checks checkpoint payload scope, resume/training hashes, CSV
sidecars, unique primary keys, numeric finiteness, clean-run provenance,
completion marker and immutable archive fields.
