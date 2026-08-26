# Clean-v1 Phase 1 review package (v2)

**Branch:** `clean-v1-data-provenance`
**Head commit:** see `git rev-parse HEAD`
**Generated:** 2026-08-26 (revised after first review pass)

This package is the Phase 1 stopping point requested by
[`docs/START_HERE_ZH.md`](../../docs/START_HERE_ZH.md). It contains
provenance only — no model is trained, no Qwen call is made, no test
data is opened.

## Revision history

* **v1 (3bfec1f):** initial provenance package. Accepted as a forensic
  snapshot. Rejected for scientific use because:
  1. `pet_operator.py` was Windows-CRLF in the working tree and the
     audit was tricked with the CRLF SHA; the LF SHA was the real one
     for the committed blob.
  2. The 14/3/3 + 156/34/34 train/val/test split was kept in
     `inputs/clean_v1/` instead of being moved to `drafts/`.
  3. `fixed_scale_C.json` stated that training-set global-maximum
     would be used as the PRIMARY scale, contradicting the
     START_HERE_ZH.md rule that only a physical source can lock the
     primary scale.
  4. MedSAM2 checkpoint provenance was a placeholder; the exact file
     name and source were unverified.
  5. Subject inventory still used the v3 train/val/test roles.
* **v2 (this commit):** corrections applied.
  1. `.gitattributes` forces LF for every tracked text file; SHA is
     pinned to the LF git blob (see `experiment.yaml`).
  2. Invalid split + invalid source manifest moved to
     `inputs/clean_v1/drafts/` with a `DRAFTS_README.md` documenting
     why they are drafts.
  3. `fixed_scale_C.json` rewritten: global-max and global-min-max are
     explicitly SENSITIVITY ARMS ONLY and forbidden as primary.
  4. `medsam2_provenance.json` rewritten: the official
     `download.sh`, the five named releases on Hugging Face, the
     `sam2.1_hiera_tiny.pt` backbone URL, the observed local file
     ambiguity (could be `MedSAM2_latest.pt` or the SAM2.1 backbone
     since both are ~156 MB) and the SHA verification protocol are
     all recorded.
  5. `subject_inventory.csv` rebuilt: every one of the 244 subjects
     is `split=development` with
     `previous_exposure_status=development_allowed`. The v3 role is
     retained for forensic traceability only.

## Ten review items

| # | Item | Path | Notes |
|---|---|---|---|
| 1 | Current git commit | `git rev-parse HEAD` | see branch head |
| 2 | Subject inventory | `inputs/clean_v1/subject_inventory.csv` | 244 rows, ALL `split=development`, ALL `previous_exposure_status=development_allowed` |
| 3 | Subject split | `inputs/clean_v1/drafts/subject_split.csv` (rejected) | counts match 14/3/3 + 156/34/34 but split is INVALIDATED; current honest position is no locked test exists until external unexposed subjects are acquired |
| 4 | Source activity manifest | `inputs/clean_v1/drafts/source_activity_manifest.csv` (rejected) | 207 rows (train + validation of the invalidated split); regenerated after locked test is resolved |
| 5 | Fixed scale C candidates | `inputs/clean_v1/fixed_scale_C.json` | global-max and global-min-max are SENSITIVITY ARMS ONLY; no scale can be locked as PRIMARY without a physical source |
| 6 | HECKTOR geometry | `inputs/clean_v1/hecktor_geometry.json` | physical spacing/FOV/scanner-axis MISSING from the NPZ; HE ingestion blocked |
| 7 | MedSAM2 provenance | `inputs/clean_v1/medsam2_provenance.json` | official `download.sh`, 5 named releases, `sam2.1_hiera_tiny.pt` backbone URL; local `MedSAM2.pt` SHA cannot be claimed as the released checkpoint SHA without researcher-side verification |
| 8 | Immutable archive URI | (not yet created) | PENDING per OPERATIONS_RUNBOOK |
| 9 | `make quality` output | `logs/quality_clean_v1_phase1.log` | ruff PASS, compileall PASS, 51 tests PASS, 9 protocols PASS |
| 10 | `make audit` blocker output | `logs/audit_clean_v1_phase1.log` | see below |

## Quality gate (item 9)

| Step | Result |
|---|---|
| `ruff check --select F code/sam_kem code/experiments tests` | All checks passed |
| `python -m compileall -q code/sam_kem code/experiments tests` | (no output → success) |
| `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest` | 51 passed in 3.78s |
| `sam-kem check-protocol --protocol <each>` | 9/9 OK |

## Audit blockers (item 10)

Re-run after the v2 corrections. The blocker list will change because:

* `forward_model.implementation_sha256` is now the LF git blob SHA, so
  `FATAL_*` checks that previously passed via the CRLF SHA must still
  pass via the LF SHA.
* `subject_split.csv` and `source_activity_manifest.csv` no longer
  exist at `inputs/clean_v1/` (they are under `drafts/`), so any
  `FATAL_*` referring to their canonical location will fail with
  `missing`. This is the desired Phase 1 behaviour: the audit must
  continue to flag these as blockers until the researcher resolves
  the locked-test question.

The fresh blocker log is at `logs/audit_clean_v1_phase1.log` and is
regenerated each commit.

## Honest position after v2

1. **No locked test.** All 244 subjects are development data. The
   next step is to acquire genuinely unexposed external subjects
   (BrainWeb source models not in the v3 inventory; HECKTOR subjects
   not in HECKTOR.npz or not opened during v3). Until then, the
   study has no honest untouched test set.
2. **No fixed scale C.** Neither BrainWeb nor HECKTOR has a
   defensible physical scale. Training-set global-max and global
   min-max are sensitivity arms only. The primary scale cannot be
   locked without a physical source or peer review.
3. **HECKTOR physical geometry is missing.** `HECKTOR.npz` does not
   embed voxel spacing, scanner FOV, or scanner-axis coordinates.
   18–28% of PET activity lies outside the 144×144 inscribed circle.
   HE ingestion is blocked until DICOM metadata is recovered or the
   dataset is dropped from the study.
4. **MedSAM2 checkpoint provenance is ambiguous.** The local
   `MedSAM2.pt` (156 MB) matches the size of both
   `MedSAM2_latest.pt` and `sam2.1_hiera_tiny.pt`. Neither file has
   a published SHA-256. The researcher must download from the
   official Hugging Face URL, compare SHA-256 locally, and either
   confirm the local file or replace it.
5. **Official MedSAM2 repo commit not yet pinned.** The researcher
   must clone `https://github.com/bowang-lab/MedSAM2`, capture
   `git rev-parse HEAD`, and set `medsam2.source_commit` /
   `medsam2.source_checkout_path` in `experiment.yaml`.
6. **Immutable archive URI not yet set.** Per
   `OPERATIONS_RUNBOOK.md` a mutable local path is insufficient.

## What is allowed at the Phase 1 stop point

* Nothing is locked, nothing is trained.
* No Qwen call has been made.
* No test array, prompt image, prior, or metric has been opened.
* The configuration carries only the resources that genuinely exist
  on disk (current `pet_operator.py` SHA, source file SHAs,
  observed but unverified local `MedSAM2.pt` SHA) and explicit
  `PENDING_*` markers where the source material is genuinely missing
  or unverified.
