# Clean-v1 Phase 1 review package

**Branch:** `clean-v1-data-provenance`
**Head commit:** see `git rev-parse HEAD`
**Generated:** 2026-08-26

This package is the Phase 1 stopping point requested by
[`docs/START_HERE_ZH.md`](../../docs/START_HERE_ZH.md). It contains
provenance only — no model is trained, no Qwen call is made, no test
data is opened.

## Ten review items

| # | Item | Path | Notes |
|---|---|---|---|
| 1 | Current git commit | `git rev-parse HEAD` | see branch head |
| 2 | Subject inventory | `inputs/clean_v1/subject_inventory.csv` | 244 rows; HONEST exposure status (all development_allowed) |
| 3 | Subject split (with sidecar) | `inputs/clean_v1/subject_split.csv` + `.manifest.json` | counts match 14/3/3 + 156/34/34; SHA-256 of inventory is in every row |
| 4 | Source activity manifest | `inputs/clean_v1/source_activity_manifest.csv` | 207 rows (train + validation); test_locked intentionally omitted |
| 5 | Fixed scale C candidates | `inputs/clean_v1/fixed_scale_C.json` | physical source PENDING for both datasets; data-derived global-max candidate pending train-only compute |
| 6 | HECKTOR geometry | `inputs/clean_v1/hecktor_geometry.json` | physical spacing/FOV/scanner-axis MISSING from the NPZ; HE ingestion blocked |
| 7 | MedSAM2 provenance | `inputs/clean_v1/medsam2_provenance.json` | upstream URL pinned; commit/checkout PENDING; released checkpoint SHA recorded; v3 decoders marked forensic-only |
| 8 | Immutable archive URI | (not yet created) | PENDING per OPERATIONS_RUNBOOK |
| 9 | `make quality` output | `logs/quality_clean_v1_phase1.log` | ruff PASS, compileall PASS, 51 tests PASS, 9 protocols PASS |
| 10 | `make audit` blocker output | `logs/audit_clean_v1_phase1.log` | 22 FATAL blockers — expected, see below |

## Quality gate (item 9)

| Step | Result |
|---|---|
| `ruff check --select F code/sam_kem code/experiments tests` | All checks passed |
| `python -m compileall -q code/sam_kem code/experiments tests` | (no output → success) |
| `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest` | 51 passed in 3.78s |
| `sam-kem check-protocol --protocol <each>` | 9/9 OK |

## Audit blockers (item 10)

The 22 FATAL blockers fall into three groups. They are NOT errors to
silence — each represents a Phase 1 prerequisite that the researcher
must explicitly resolve.

### Group A — Provenance records that need real values

* `FATAL SPLIT: data.split_sha256 must pin the clean subject split`
  — the split is currently invalidated (see below); SHA stays null
  until the test-set question is resolved.
* `FATAL SOURCE_MANIFEST: could not convert string to float: 'PENDING_PHYSICAL_GEOMETRY'`
  — the HECKTOR physical geometry is genuinely missing from the NPZ;
  see `hecktor_geometry.json`.
* `FATAL MEDSAM2_CHECKOUT: not a readable Git checkout: ...PENDING_LOCAL_PATH_BEFORE_TRAINING`
  — the official MedSAM2 source checkout must be cloned and the commit
  SHA pinned before training.
* `FATAL MEDSAM2_PRETRAINED: pretrained MedSAM2 checkpoint/path SHA is unresolved`
  — the SHA is recorded in `medsam2_provenance.json` but the audit
  script's path/SHA pair check requires the path to exist on disk
  (or be specified correctly).

### Group B — Resources that Phase 2/3 will produce

* `FATAL CHECKPOINT_ARCHIVE / FATAL CHECKPOINT_SHA` (×4)
  — the clean_v1 dataset-specific decoders do not exist yet.
* `FATAL BOX_CACHE / FATAL TRAINING_BOX_CACHE / FATAL PRETRAINING_MANIFEST / FATAL VALIDATION_MANIFEST / FATAL BOX_INPUT_MANIFEST / FATAL MANIFEST_MISSING` (×6)
  — Qwen box cache, prompt images, validation manifest, and all
  case-manifest artefacts are produced by Phase 2.
* `FATAL RUN_METADATA / FATAL RUN_INCOMPLETE / FATAL RESULTS_MISSING / FATAL SWEEP_MISSING / FATAL FROZEN_HYPERPARAMETERS_MISSING` (×5)
  — these require executing comparison, sweep and frozen-parameter
  protocols in Phase 3.
* `FATAL RESULT_ARCHIVE` — immutable archive URI unresolved.

### Group C — Configuration guards

* `FATAL CONTRACT: clean split_id and 64-character split_sha256 are required`
  — same root cause as Group A `FATAL SPLIT`.
* `FATAL CONFIG_UNLOCKED` — `locked: false` is correct: we are NOT yet
  ready for paper execution. Per START_HERE_ZH.md, the configuration
  must stay unlocked until every Group A and Group B blocker is
  resolved.

## Honest finding: the split is currently invalidated

Per START_HERE_ZH.md §五 (subject-level split), every locked_test row
must have `previous_exposure_status = never_used_for_training_validation_tuning_or_reporting`.
The subject inventory contains ZERO such rows because all 244 v3
subjects were opened during the invalid v3 sweep. The split file
therefore carries `INVALIDATED` status and the manifest sidecar
documents the three legitimate next moves (acquire unexposed subjects,
declare no untouched test, or stop). The split SHA is intentionally
not pinned in `experiment.yaml` until the researcher resolves this.

## What is allowed at the Phase 1 stop point

* Nothing is locked, nothing is trained.
* No Qwen call has been made.
* No test_locked array, prompt image, prior, or metric has been opened.
* The configuration carries only the resources that genuinely exist
  on disk (released MedSAM2 checkpoint SHA, current pet_operator.py
  SHA, source file SHAs) and explicit `PENDING_*` markers where the
  source material is genuinely missing.

## Next actions for the researcher

1. Decide on test-set resolution (acquire unexposed subjects / declare
   no untouched test / stop the study). Update
   `inputs/clean_v1/subject_inventory.csv` accordingly and re-run
   `sam-kem create-split`.
2. Recover HECKTOR physical geometry from a DICOM header / HECKTOR
   challenge documentation; otherwise drop HECKTOR from the study.
3. Clone MedSAM2 at a fixed commit, set `medsam2.source_commit` and
   `medsam2.source_checkout_path` in `experiment.yaml`, and re-run
   `make quality`.
4. Run `sam-kem calibrate-normalization` on the train split to
   populate `inputs/clean_v1/fixed_scale_C.json` with the data-derived
   global-maximum candidate.
5. Provide an immutable archive URI for checkpoints and results.
6. Pin `data.split_sha256` once the test set is resolved.
7. Set `locked: true` and `status: locked_for_execution` only after
   every blocker above is resolved.

Until then, no paper-eligible run is possible.
