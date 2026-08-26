# Repository map and authority boundaries

## Status classes

Every repository path belongs to one of five classes.

| Class | Meaning | May execute? | May support paper results? |
|---|---|---:|---:|
| Active | Reviewed clean implementation, configuration, protocol, or entry point | Yes | Yes, after the release gate passes |
| Template | Schema example that must be copied and completed | No | No |
| Diagnostic | Small sample or audit used to understand data/operator behavior | Only with audit entry points | No |
| Forensic | Retained historical source, configuration, log, prompt, or result | No | No |
| Generated | Output of a clean run, normally immutable and SHA-bound | Through the owning command only | Only if complete and validated |

## Top-level inventory

| Path | Class | Purpose |
|---|---|---|
| `README.md` | Active | Repository landing page and safety boundary |
| `SCIENTIFIC_STATUS.md` | Active | Claims that are currently scientifically defensible |
| `pyproject.toml`, `environment.yml`, `requirements.txt` | Active | Reproducible environment definitions |
| `Makefile` | Active | Short quality and audit commands; it does not run scientific experiments |
| `code/sam_kem/` | Active | Canonical library and `sam-kem` CLI |
| `code/experiments/` | Active | Thin protocol-specific entry points |
| `code/configs/experiment.yaml` | Active but blocked | Clean scientific contract; unresolved fields are deliberately `null` |
| `code/configs/protocols/` | Active | Independent, reviewable experiment definitions |
| `inputs/*.template.csv` | Template | Input schemas; copy to `inputs/clean_v1/` and replace placeholders |
| `tests/` | Active | Mathematical, integrity, protocol, and documentation checks |
| `docs/` | Mixed | Active documents plus explicitly retired v3 records |
| `data/raw_samples/` | Diagnostic | Traceability exports with source mapping |
| `data/raw_pet/`, `data/samples/` | Diagnostic | Uploaded arrays used for scale/operator audits |
| `code/src/original/` | Forensic | Original mixed-operator implementation snapshot |
| `code/src/*.py` | Forensic | v3 orchestration/audit sources |
| `code/scripts/_*.py` | Forensic | Machine-specific v3 drivers with known errors |
| `code/configs/*v3*`, `*v4*`, legacy headline/config CSV/YAML | Forensic | Exposed splits and superseded configuration |
| `results/metrics/`, `results/convergence/`, `results/reports/` | Forensic | Invalid v3 outputs retained for audit |
| `prompts/*.csv`, `prompts/*.json`, `prompts/*_samples/` | Forensic | Historical per-realization Qwen records and cache evidence |
| `logs/` | Forensic | Historical decoder logs |
| `arrays/system_matrix_A_n180_sha256.txt` | Forensic | Former matrix hash, not the clean operator identity |
| `v3_repo_complete.bundle` | Forensic | Offline Git transfer bundle; not data or experiment evidence |

## Active code dependency direction

```text
code/configs/experiment.yaml + code/configs/protocols/*.yaml
                              |
                              v
code/experiments/*.py --> code/sam_kem/*.py --> immutable clean artifacts
                              |
                              v
                         tests/*.py
```

Active code must never import from `code/src`, `code/src/original`, or
`code/scripts`.
The test suite enforces this boundary textually and exercises the numerical
operator, noise model, reconstruction methods, ingestion, splits, training
sampler, prompt cache, artifacts, and protocols.

## Generated clean namespaces

The following directories may not exist in a fresh clone.
They appear only during a clean run:

```text
inputs/clean_v1/       completed source, normalization and split manifests
arrays/clean_v1/       canonical NumPy arrays and JSON sidecars
prompts/clean_v1/      prompt manifests, response attempts, box caches and PNGs
checkpoints/clean_v1/  inference/resume checkpoints, training CSV and provenance
runs/clean_v1/         immutable output directory for each protocol
results/clean_v1/      case manifests and frozen hyperparameters
```

Bulk arrays, prompt pixels and checkpoint tensors are excluded from Git.
Their manifests and provenance are tracked, while the bulk data are copied to
the configured immutable archive.

## Safe navigation rules

- Run commands from the repository root.
- Prefer the `sam-kem` CLI or a file in `code/experiments/`.
- Never execute an underscore-prefixed file in `code/scripts/`.
- Never use `subject_split_v3*.csv` or `subject_split_v4.csv` for `clean_v1`.
- Never use a CSV containing `_v3` in a manuscript table.
- Never copy a historical decoder SHA into the clean config merely to satisfy
  validation.
- Never delete or rewrite forensic evidence to make an audit pass.
- A changed scientific choice requires a new run ID and downstream rerun.
