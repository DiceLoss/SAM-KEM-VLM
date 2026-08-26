# Clean-v1 Phase 1 review package (v3)

> **Superseded operationally on 2026-08-26.** This file remains the Phase-1
> forensic review record. The active repository now contains a SHA-pinned
> development-only 16/4/0 BrainWeb and 179/45/0 HECKTOR split, a pinned MedSAM2
> source commit, and no HE scanner-geometry blocker. It still has no untouched
> test. GPU execution follows `docs/GPU_WORKER_START_ZH.md`.

**Branch:** `clean-v1-data-provenance`
**Head commit:** see `git rev-parse HEAD`
**Generated:** 2026-08-26 (v3 corrections applied after second review pass)

This package is the Phase 1 stopping point requested by
[`docs/START_HERE_ZH.md`](../../docs/START_HERE_ZH.md). It contains
provenance only — no model is trained, no Qwen call is made, no test
data is opened.

## Revision history

* **v1 (3bfec1f):** initial provenance package. Accepted as a forensic
  snapshot. Rejected for scientific use for five reasons.
* **v2 (21eae87):** corrections applied. Subjective over-correction on
  fixed-scale rules: stated that training-set global-maximum is
  permanently disqualified as PRIMARY, which conflicts with the
  START_HERE_ZH.md guidance that data-derived fixed scales are valid
  candidates pending complete validation-based selection. Also kept
  the MedSAM2 size discrepancy and the "ambiguous file" framing
  because the official HF SHA-256 was not independently retrieved.
* **v3 (this commit):** corrections applied.
  1. **MedSAM2 checkpoint verified.** The official Hugging Face
     `MedSAM2_latest.pt` SHA-256 and size are now recorded:
     - HF revision: `c0b302de5e19211e9f1425c6215f6172eb5eca7f`
     - Official SHA-256: `c92743b99f00d078bf32a3afcc38aaa9faf1c1692dffe3eaa7a90938c1991060`
     - Official size: `156040129` bytes
     - Local file size re-measured: `156040129` bytes (was wrongly
       recorded as `156046933` in v1/v2).
     - Local file is byte-identical to the released
       `MedSAM2_latest.pt`. The "could be SAM2.1 backbone"
       ambiguity is resolved.
  2. **experiment.yaml uses no `PENDING_*` placeholders.** All
     unresolved fields are `null`. The pretrained checkpoint path
     is now `checkpoints/MedSAM2_latest.pt` and the SHA-256 is
     pinned to the verified value.
  3. **fixed_scale_C.json rewritten.** The four candidates are
     named explicitly (domain_physical_fixed_C,
     train_derived_zero_offset_fixed_C,
     train_global_minmax_sensitivity, per_image_minmax_diagnostic_only)
     with correct eligibility rules:
     - No candidate is pre-declared PRIMARY.
     - Each viable candidate must be paired with separate decoder
       training and a complete validation sweep.
     - The validation-based selection protocol is documented.
     - If train_derived_zero_offset_fixed_C wins, it is a
       data-derived fixed scale, NOT a physical constant.

## Ten review items

| # | Item | Path | Notes |
|---|---|---|---|
| 1 | Current git commit | `git rev-parse HEAD` | see branch head |
| 2 | Subject inventory | `inputs/clean_v1/subject_inventory.csv` | 244 rows, ALL `split=development`, ALL `previous_exposure_status=development_allowed` |
| 3 | Subject split | `inputs/clean_v1/drafts/subject_split.csv` (rejected) | 14/3/3 BW + 156/34/34 HE counts but split is INVALIDATED; current honest position is no locked test exists until external unexposed subjects are acquired |
| 4 | Source activity manifest | `inputs/clean_v1/drafts/source_activity_manifest.csv` (rejected) | regenerated after locked test is resolved |
| 5 | Fixed scale C candidates | `inputs/clean_v1/fixed_scale_C.json` | four candidates listed; primary scale determined by the validation-based selection protocol; no candidate pre-declared PRIMARY |
| 6 | HECKTOR geometry | `inputs/clean_v1/hecktor_geometry.json` | resolved for the declared pixel-domain pseudo-sinogram: complete DICOM-exported square, no scanner-FOV inference |
| 7 | MedSAM2 provenance | `inputs/clean_v1/medsam2_provenance.json` | verified byte-identical to official HF `MedSAM2_latest.pt` at revision `c0b302de5e19211e9f1425c6215f6172eb5eca7f` |
| 8 | Immutable archive URI | (not yet created) | null per OPERATIONS_RUNBOOK |
| 9 | `make quality` output | `logs/quality_clean_v1_phase1.log` | ruff PASS, compileall PASS, 51 tests PASS, 9 protocols PASS |
| 10 | `make audit` blocker output | `logs/audit_clean_v1_phase1.log` | 22 FATAL blockers — expected |

## Quality gate (item 9)

| Step | Result |
|---|---|
| `ruff check --select F code/sam_kem code/experiments tests` | All checks passed |
| `python -m compileall -q code/sam_kem code/experiments tests` | (no output → success) |
| `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest` | 51 passed in ~4s |
| `sam-kem check-protocol --protocol <each>` | 9/9 OK |

## Audit blockers (item 10)

22 FATAL blockers. Categorisation is unchanged from v2:

* **Group A:** provenance records that need real values to clear:
  split SHA (null until locked test is resolved), fixed-scale/unit provenance,
  MedSAM2 source commit / checkout path
  (still null — `source_repository`, `pretrained_checkpoint_path`
  and `pretrained_checkpoint_sha256` are now resolved but the
  pinning of the Git commit that ships the verified
  `download.sh` is still pending), and the immutable archive URI.
* **Group B (17):** resources that Phase 2/3 will produce
  (checkpoints, box caches, manifests, sweep results, etc.).
* **Group C (1):** `FATAL CONFIG_UNLOCKED` — `locked: false` is
  correct at the Phase 1 stop point.

The fresh blocker log is at `logs/audit_clean_v1_phase1.log` and is
regenerated each commit.

## Honest position after v3

1. **No locked test.** All 244 subjects are development data. The
   next step is to acquire genuinely unexposed external subjects
   (BrainWeb source models not in the v3 inventory; HECKTOR subjects
   not in HECKTOR.npz or not opened during v3). Until then, the
   study has no honest untouched test set.
2. **No fixed scale C locked.** Four candidates are named. The
   primary scale is determined by the validation-based selection
   protocol: each viable candidate requires separate decoder
   training and a complete validation sweep. Distribution plots
   alone do not select a winner.
3. **HECKTOR scanner geometry is not an ingestion requirement.** The
   researcher confirmed that the array contains reconstructed clinical PET
   slices exported directly from DICOM. The declared experiment is a
   pixel-domain pseudo-sinogram simulation, so it preserves the complete
   square and does not infer scanner FOV or scanner-axis coordinates. DICOM
   unit/rescale provenance remains relevant to normalization-scale selection.
4. **MedSAM2 pretrained checkpoint verified.** Local file is
   byte-identical to the official `MedSAM2_latest.pt`. The
   researcher must still pin `medsam2.source_commit` and
   `medsam2.source_checkout_path` to the Git commit that ships the
   `download.sh` used to verify the SHA.
5. **Immutable archive URI not yet set.** Per
   `OPERATIONS_RUNBOOK.md` a mutable local path is insufficient.

## What is allowed at the Phase 1 stop point

* Nothing is locked, nothing is trained.
* No Qwen call has been made.
* No test array, prompt image, prior, or metric has been opened.
* The configuration carries only the resources that genuinely exist
  on disk (current `pet_operator.py` SHA, source file SHAs,
  verified `MedSAM2_latest.pt` SHA) and `null` for genuinely
  unresolved fields.
