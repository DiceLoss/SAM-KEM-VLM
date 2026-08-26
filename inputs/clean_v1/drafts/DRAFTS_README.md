# Drafts (NOT for paper use)

These files were produced during Phase 1 data-provenance work and are
retained only so the chain of reasoning remains auditable. They are
**invalidated** and must not be consumed by `sam-kem` or any Phase 2
script.

## Subject split — `subject_split.csv` + `.manifest.json`

The Phase 1 split attempted to assign subjects to
train/validation/test_locked with the 14/3/3 BW + 156/34/34 HE counts.
This split was rejected because every one of the 244 subjects had
`previous_exposure_status = development_allowed`. START_HERE_ZH.md §五
explicitly forbids relabelling previously exposed subjects as
`never_used_for_training_validation_tuning_or_reporting`. The split
therefore has no honest `test_locked` rows and is kept here only as
forensic material.

The current honest position is: all 244 subjects are development data;
no locked test exists until genuinely unexposed external subjects are
acquired. See `../PHASE1_REVIEW_PACKAGE.md` for the next-step options.

## Source activity manifest — `source_activity_manifest.csv`

This manifest pinned `source_sha256`, `source_slice_axis`, and the
spatial-transform string for each (dataset, subject, slice) in the
invalidated split. Because the split itself is invalidated, this
manifest is invalidated too. Re-generate after the locked test is
resolved.

Several HECKTOR rows carry obsolete `PENDING_PHYSICAL_GEOMETRY` fields from the
retired circular-FOV design. They are retained only because this CSV is an
invalidated forensic artifact. A regenerated manifest must use
`resize_full_square_to_128` and the current template; scanner FOV and axis are
not inputs to the pixel-domain operator. See `../hecktor_geometry.json`.

## `source_sha256.json`

Raw SHAs computed for the BW per-subject `true_pet.nii.gz` files and
for `HECKTOR.npz`. These SHAs are still authoritative for the source
data itself and will be reused when a real split is built.

## `normalization_activity_manifest.csv`

Empty-template copy; not populated because the split it would index is
invalidated.

## How these drafts interact with the rest of the repo

The audit (`sam-kem validate`) references
`inputs/clean_v1/subject_split.csv` and
`inputs/clean_v1/source_activity_manifest.csv`. Because those files
have been moved into `drafts/`, the audit will fail with `missing`
errors instead of parsing invalid content. That is the desired
behaviour at the Phase 1 stop point — every blocker must remain
visible until the researcher explicitly resolves it.
