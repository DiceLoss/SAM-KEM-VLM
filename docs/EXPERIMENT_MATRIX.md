# Experiment matrix and dependency graph

All protocol YAMLs are under `code/configs/protocols/`.
The table below defines which data a protocol may see and whether it is allowed
to select a scientific choice.

## Ordered experiment matrix

| Order | Protocol | Split | Counts | Role | May select parameters? | Principal output |
|---:|---|---|---|---|---:|---|
| 0 | normalization calibration | train fit, validation diagnose | native activity | Compare configured physical C, train maximum, train-global min-max, and diagnostic per-image min-max | No automatic winner; a candidate needs complete retraining and validation comparison | `runs/clean_v1/normalization_calibration/` |
| 1 | decoder training | train | 100k, 200k, 500k, 1M balanced | Train one joint four-count decoder per dataset | Fixed budget only; no test or “best” selection | `checkpoints/clean_v1/` and `runs/clean_v1/training/` |
| 2 | hyperparameter sweep | validation | all configured counts | Select baseline/KEM stopping and kernel/fusion parameters | Yes, subject-level NRMSE then SSIM | validation sweep plus `results/clean_v1/frozen_hyperparameters.csv` |
| 3 | comparison | test locked | all configured counts | Paired evaluation of all applicable methods | No | `runs/clean_v1/comparison/per_case_metrics.csv` |
| 4 | primary | test locked, filtered from comparison | 200k, r0/r1/r2 | Produce headline proposed-method rows | No reconstruction and no new test read | `runs/clean_v1/primary/` |
| 5 | ablation | test locked | 200k | One-factor-at-a-time component removal/replacement | No | `runs/clean_v1/ablation/` |
| 6 | convergence | test locked | 200k | Descriptive curves at predeclared iterations | No | `runs/clean_v1/convergence/` |
| 7 | cross-count | test locked, aggregate existing per-case rows | 100k, 200k, 500k, 1M | Count-level robustness summary | No | `runs/clean_v1/cross_count/` |
| 8 | runtime | validation | 200k, r0 | Cached-input reconstruction timing | No | `runs/clean_v1/runtime/` |

The test release occurs only between orders 2 and 3, after the validation sweep,
frozen parameters, manifests, success marker, configuration lock, and clean
worktree are verified.

## Method applicability

| Method | BrainWeb | HECKTOR | Learned/prior input | Anatomical input | Definition |
|---|---:|---:|---|---|---|
| MLEM | Yes | Yes | None | None | Matched Poisson EM baseline; stopping selected on validation |
| OSEM | Yes | Yes | None | None | Ten-subset ordered-subsets baseline; report complete cycles and subset updates |
| Bowsher-KEM | Yes | Yes | None | MR or CT | Coefficient-space KEM with anatomy-selected neighbors; not classic MAP-Bowsher |
| SelfKEM | Yes | Yes | MLEM-10/self feature | None | Coefficient-space KEM with self-image feature |
| MR-KEM | Yes | No | None | MR | BrainWeb anatomical-feature KEM baseline |
| CT-KEM | No | Yes | None | CT | HECKTOR anatomical-feature KEM baseline |
| SAM prior | Yes | Yes | Qwen box + dataset decoder | None | Continuous decoder prior without KEM |
| SAM-KEM no post | Yes | Yes | Qwen box + dataset decoder | None | Coefficient-space KEM using SAM prior feature, before post-fusion |
| SAM-KEM full | Yes | Yes | Qwen box + dataset decoder | None | SAM-KEM plus validation-selected fusion alpha |

The proposed method is PET-only apart from the VLM text/box mechanism.
CT, MR, and tumor masks are not proposed-method inputs.

## Hyperparameter ownership

| Choice | Selected where | Reused where |
|---|---|---|
| Fixed scale C / normalization arm | Physical provenance or complete train/validation sensitivity workflow | Every downstream array, decoder, sweep, and evaluation within that run identity |
| Decoder training budget | Predeclared in config | Dataset-specific final fixed-budget checkpoint |
| Prompt MLEM iterations | Fixed at 10 for VLM/MedSAM input construction | Box generation and decoder input only |
| MLEM baseline iterations | Validation sweep | Test comparison, convergence annotation, cross-count |
| OSEM cycles | Validation sweep | Test comparison and runtime |
| KEM sigma/radius/iterations | Validation sweep per dataset/count/method | All applicable test protocols |
| Bowsher radius/neighbors | Validation sweep | Bowsher-KEM test protocols |
| Post-fusion alpha | Validation sweep for full SAM-KEM | Test comparison/primary/ablation reference |
| Qwen box | One call per dataset/subject/slice/count from r0 | Shared across r0/r1/r2 and applicable variants |

An optimum on a non-physical search-grid boundary requires a predeclared grid
extension and another validation sweep.
The locked test cannot justify an extension or a new choice.

## Required ablations

The reference is `full_qwen_trained_decoder_post_fusion`.
Each row changes exactly one component:

| Variant | Single changed factor |
|---|---|
| `centroid_box_only` | Qwen box replaced by recorded intensity-centroid fallback box |
| `released_decoder_only` | Dataset-trained mask decoder replaced by the verified released decoder |
| `no_post_fusion` | Post-fusion alpha removed |
| `prior_only` | Iterative reconstruction component removed |

These must be actual reruns from the same cases and inputs.
MR-KEM, CT-KEM, or prior-only comparison rows may not be relabeled as missing
ablations.

## Fairness and aggregation

- Applicable methods use identical subjects, slices, counts, and realizations.
- Random streams are keyed independently for every dataset/subject/slice/count/
  realization identity.
- Image metrics use one operator-derived full-square support for every method.
- Per-slice rows are retained, but inferential statistics aggregate at subject
  level because adjacent slices from one subject are not independent subjects.
- Paired statistics compare the proposed method with each applicable baseline
  on identical subject-level units.
- Cross-count keys are unique and aggregated once; duplicate rows are fatal.
- Network/VLM latency is not mixed into cached-prior reconstruction timing.

## Dependency graph

```text
source provenance + untouched subject inventory + declared spatial transform
                |
                v
split -> ingestion -> fixed-C targets -> projections -> Poisson/MLEM-10
                |                                      |
                |                                      v
                |                              Qwen cache (r0 shared)
                |                                      |
                v                                      v
normalization decision ----------------------> decoder training
                                                       |
                                                       v
validation priors -> validation sweep -> frozen parameters -> test release
                                                              |
                                                              v
comparison -> primary / ablation / convergence / cross-count
validation cached inputs ------------------------------------> runtime
```
