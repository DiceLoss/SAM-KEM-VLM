# Data asset manifest

This repository does contain committed NumPy samples for traceability and
black-box diagnostics.
At the 2026-08-26 audit, tracked files occupy about 101 MiB; the largest tracked
file is `data/raw_pet/raw_pet_hecktor.npz` at about 15 MiB.
The repository does **not** contain the complete source datasets, clean-run
arrays, rendered prompt images, or model checkpoints.

Committed sample arrays are never substitutes for a complete, SHA-pinned source
manifest and are not clean paper evidence.

## 1. Committed diagnostic samples

| Path | Contents | Intended use | Paper-result eligibility |
|---|---|---|---|
| `data/raw_samples/` | Two BrainWeb train, one BrainWeb validation, four HECKTOR train, and two HECKTOR validation traceability cases | Verify source mapping, units, mean-pooling, and sample provenance | No |
| `data/raw_pet/` | Historical uploaded BrainWeb/HECKTOR PET arrays | Operator/scale diagnosis and comparison with the former pipeline | No |
| `data/samples/` | Historical targets, lambda arrays, and sinograms | Black-box operator-equivalence audit | No |

The authoritative descriptions are the JSON manifests beside these files.
Their findings are summarized in
[`docs/UPLOADED_SAMPLE_AUDIT.md`](../docs/UPLOADED_SAMPLE_AUDIT.md).

Do not copy any committed sample NPZ directly into `arrays/clean_v1`.
Clean ingestion starts from `inputs/clean_v1/source_activity_manifest.csv` and
records the original source file SHA, unit, subject/slice selector, and spatial
geometry.

## 2. Required checkpoints (not in Git)

| File | Status | Clean requirement |
|---|---|---|
| `checkpoints/MedSAM2.pt` | Missing | Record the official source repository, immutable source commit, local path, SHA-256, and external archive URI. |
| `checkpoints/clean_v1/medsam2_decoder_brainweb.pt` | Missing | Retrain with BrainWeb train subjects under the final normalization; record SHA-256 and the complete checkpoint bundle. |
| `checkpoints/clean_v1/medsam2_decoder_hecktor.pt` | Missing | Retrain independently with HECKTOR train patients under the final normalization; record SHA-256 and the complete checkpoint bundle. |

Historical decoder hashes and paths in the v3 reports are forensic evidence,
not accepted clean checkpoints.
The clean configuration deliberately contains `null` checkpoint hashes until
the retraining artifacts exist.

## 3. Complete source datasets (not in Git)

| Data | Declared historical location | Clean action required |
|---|---|---|
| BrainWeb NIfTI volumes | `D:/MIAProject/brainweb/数据/brainweb_petmr/` | Enumerate exact source files, SHA-256, simulation/source-model identity, slice indices, and activity unit. |
| HECKTOR archive | `D:/MIAProject/HECKTOR.npz` | Record archive SHA-256, patient mapping, PET channel, slice selector, SUV unit, pixel spacing, scanner FOV, and scanner axis. |

Machine-specific paths document the former workstation only.
They must be replaced by accessible paths in the researcher's private
`inputs/clean_v1/source_activity_manifest.csv`; do not commit patient-sensitive
absolute paths if institutional policy prohibits them.

## 4. Clean generated data (not in Git)

The clean pipeline writes bulk artifacts under:

```text
arrays/clean_v1/          canonical activities, projections, sinograms, MLEM-10 and priors
prompts/clean_v1/         rendered PNGs, request attempts and box caches
checkpoints/clean_v1/     dataset-specific decoder bundles
runs/clean_v1/            protocol run directories
results/clean_v1/         tracked case manifests and frozen parameters
```

The `.gitignore` excludes bulk arrays, PNGs and Torch files.
Their sidecars and CSV manifests remain tracked, and the bulk files must be
copied to one immutable archive whose URI is recorded in the locked config.
See the [artifact specification](../docs/ARTIFACT_SPEC.md).

## 5. Historical v3 processed data

The v3 pipeline produced per-slice sinograms, lambda files, and SAM
priors under `D:/MIAProject/SAM_KEM_QWEN3_MEDSAM2_DELIVERY/arrays/`.
Total ~30 GB across 3,511 slices × 12 combos = 42,132 sinograms
plus 12,612 priors. These are NOT in this repo.

The per-case metrics CSVs under `results/metrics/` are committed as forensic
evidence only.
They contain known operator, checkpoint, normalization, box-cache, sweep and
coverage violations and must not be transferred into the manuscript.

## 6. Historical external archives

### Audit-only archive

| File | SHA-256 | Location |
|---|---|---|
| `SAM_KEM_QWEN3_AUDIT_ONLY_20260823.tar.gz` (603 MB) | `95dbb1a470fe1ddcf3e4f17b8801da6649e829b7fcca67253e193da12a7e52c4` | `D:/MIAProject/` |
| `SAM_KEM_QWEN3_AUDIT_ONLY_20260823.sha256` | (above) | `D:/MIAProject/` |

This archive is byte-unchanged from the original v1/v2 delivery. It
is NOT in this repo (it predates the v3 fix).

### Former v3 “final” archive

| File | SHA-256 | Location |
|---|---|---|
| `SAM_KEM_QWEN3_FINAL_v3.tar.gz` (~280 MB) | `c413227b944d1df81e9405353970ca68d36bcd81c42bc2d07d8a135be82c2d79` | `D:/MIAProject/` |
| `SAM_KEM_QWEN3_FINAL_v3.sha256` | (above) | `D:/MIAProject/` |

These archives are not clean-run inputs.
The tracked `v3_repo_complete.bundle` is an offline historical Git-transfer
bundle; normal users should clone GitHub and must not treat the bundle as an
experiment artifact.

## 7. Data handling rules

1. Never commit API keys, DICOM headers with identifiers, unrestricted patient
   data, or model checkpoints.
2. Verify source SHA-256 before ingestion and canonical array SHA-256 after
   loading.
3. Keep native units explicit; in particular, do not apply a count-scale
   constant near 30000 to uploaded SUV arrays.
4. Keep train/validation/test assignments at subject or BrainWeb source-model
   level.
5. Do not open `test_locked` until the validation sweep and frozen-parameter
   artifacts pass the release gate.
6. Preserve failed Qwen calls and their fallback records; never selectively
   retry based on an observed result.
