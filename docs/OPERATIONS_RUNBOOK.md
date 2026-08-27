# Clean-run operations runbook

This runbook covers execution hygiene, interruption recovery, failure handling,
archival, and research handoff.
Scientific command order is defined in [`EXECUTION.md`](EXECUTION.md).

## Before any run

Work from a named Git commit and a clean worktree:

```bash
git status --short
git rev-parse HEAD
make quality
```

Record the commit in the run log.
Do not start if active code/config changes are uncommitted, if a configured input
hash does not match, or if the command would write into an existing completed run
directory.

Confirm that secrets and sensitive data are outside Git:

```bash
git status --short --untracked-files=all
git check-ignore -v checkpoints/MedSAM2.pt
```

The DashScope API key belongs in the process environment only.
Never place it in YAML, JSON, shell history committed to Git, logs, or prompt
records.

## Gate checklist by stage

### Before ingestion

- The subject inventory declares prior exposure for every subject.
- The split is subject/source-model level, generated once, inspected, committed,
  and SHA-pinned.
- Every source row has an exact source SHA, unit, selector, and approved spatial
  transform.
- HECKTOR rows contain the verified DICOM-export PET channel/unit, slice mapping,
  source SHA, and the fixed complete-square transform.
- Train and validation are prepared first; locked test remains unopened.

### Before normalization freeze

- Only train/validation raw activities appear in the normalization manifest.
- Physical C and its source are recorded when known.
- Data-derived alternatives have separate normalization identities.
- Before allocating GPU work, prove whether alternatives differ on training
  arrays. Mathematically identical arms are recorded as diagnostics and are not
  assigned duplicate decoder training.
- The audited development choice, evidence hashes, clipping behavior and claim
  boundary are recorded before generating a training configuration.

### Before decoder training

- The official MedSAM2 source checkout and immutable commit are recorded.
- The pretrained checkpoint SHA matches the configuration.
- The training box cache contains one record per dataset/subject/slice/count,
  shared across realizations, with failures/fallbacks preserved.
- The case manifest contains train rows only for training and has no test
  subjects.
- Training steps per epoch are divisible by four count levels.

### Before validation sweep

- Both dataset-specific checkpoint bundles are complete and archived.
- The configuration contains their exact SHAs and archive URIs.
- Validation priors use the correct dataset checkpoint.
- The worktree is clean and the configuration is locked for execution.
- No locked-test array, prompt, prior, or metric has been opened.

### Before locked-test release

- The validation sweep CSV and frozen-parameter CSV pass their schema/hash
  checks.
- The sweep `_SUCCESS.json` binds both files and their sidecars.
- All required dataset/count/method combinations are present.
- Boundary optima have been expanded on validation or explicitly resolved before
  release.
- The validator's pre-test prerequisites pass.

### Before paper use

- Every protocol directory has a valid `_SUCCESS.json`.
- Per-case keys are unique and all required methods/cases are complete.
- Paired subject statistics derive from the immutable comparison output.
- Primary rows are a filter of comparison rows, not a second reconstruction.
- Bulk arrays, PNGs, checkpoints, logs, configs, manifests and results are in one
  immutable archive.
- `outputs.artifact_policy.immutable_archive_uri` is filled.
- `make quality`, `sam-kem validate`, and `git status --short` all pass/are clean.

## Resume policy

`--resume-verified` is safe only for batch commands that explicitly support it.
It verifies all existing artifact groups and reuses only complete, matching
groups.
It must reject partial groups, changed hashes, or changed identities.

For decoder training, resume only from the adjacent canonical `.resume.pt` at an
epoch boundary.
Dataset, base checkpoint, config, completed epoch, optimizer, and RNG states must
match.
The rolling resume state may be atomically replaced; published inference
checkpoints and final provenance may not.

Qwen batch resume reuses every completed fixed-policy request task, including
tasks that exhausted all three attempts. It must not add retries selectively
after inspecting boxes or downstream metrics. A circuit-breaker interruption is
an incomplete batch, not a success marker and not a reason to alter the policy.

## Failure and quarantine procedure

When a command fails:

1. Stop downstream commands; do not write `_SUCCESS.json` manually.
2. Preserve stdout/stderr, run metadata, partial directory, code commit, config
   SHA, and protocol SHA.
3. Record the earliest failing case identity and exception.
4. Determine whether the cause is an environmental interruption, corrupt input,
   missing provenance, or a scientific/configuration change.
5. For an environmental interruption, use only the verified resume mechanism.
6. For corrupt or partial artifacts, rename the whole incomplete run directory
   outside the canonical output namespace for investigation; do not merge it
   with a rerun.
7. For a scientific change, choose a new run ID/output namespace and regenerate
   every downstream dependency.

Never edit a generated CSV, JSON sidecar, checkpoint payload, or success marker
to make validation pass.

## Immutability and archive layout

A recommended external archive layout is:

```text
sam_kem_clean_v1_<git-short-sha>/
  MANIFEST.sha256
  repository/
  environment/
  inputs/
  arrays/
  prompts/
  checkpoints/
  runs/
  results/
  logs/
```

Generate the archive manifest after all files are copied, store it inside the
archive, make the location read-only/versioned, and record its URI in the locked
configuration.
The URI may be an institutional object-store version, DOI-backed deposit, or
other immutable project location; a mutable local path alone is insufficient.

## Minimal handoff record

The outgoing researcher supplies:

- Git commit and branch;
- environment export and hardware/CUDA identity;
- locked configuration and protocol hashes;
- source, normalization, subject inventory, and split manifests;
- exact command invoked and start/finish UTC;
- run metadata and `_SUCCESS.json` for completed stages;
- immutable archive URI and SHA manifest;
- known failures/deviations, including fallback cases;
- confirmation that locked-test data were not used before validation freeze.

The receiving researcher first runs `make quality`, checks all supplied hashes,
and invokes `sam-kem validate`.
A verbal “finished” statement or a folder named `final` is not completion
evidence.

## Expected validator behavior

In the current repository snapshot, validation fails because required clean-run
provenance and artifacts do not exist.
That is the correct behavior.
Unit tests and repository quality checks should pass independently of the
scientific release gate.

Use:

```bash
make quality
make audit
```

`make audit` will remain non-zero until the entire clean experiment has been
executed and archived.
