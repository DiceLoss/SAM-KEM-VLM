# Active experiment entry points

Run these files from the repository root after editable installation.
They are thin wrappers around tested functions in `sam_kem`; scientific choices
belong in the configuration/protocol YAML, not in ad-hoc command edits.

| Entry point | Split/purpose |
|---|---|
| `audit_traceability_samples.py` | Diagnostic-only summary of committed traceability archives |
| `audit_uploaded_samples.py` | Diagnostic-only operator/scale checks on committed samples |
| `run_normalization_calibration.py` | Train-fit/validation-diagnose normalization candidates |
| `train_decoder.py` | Train one dataset-specific, four-count-level MedSAM2 mask decoder |
| `generate_priors.py` | Generate continuous priors for explicitly named splits |
| `run_hyperparameter_sweep.py` | Validation-only selection and parameter freeze |
| `run_comparison.py` | Locked-test paired method comparison and subject statistics |
| `run_primary.py` | Immutable filter of comparison output at the primary count |
| `run_ablation.py` | True one-factor-at-a-time locked-test variants |
| `run_convergence.py` | Descriptive locked-test convergence curves only |
| `run_cross_count.py` | Duplicate-safe aggregation across count levels |
| `run_runtime.py` | Validation cached-input reconstruction benchmark |

The wrapper defaults point to `code/configs/experiment.yaml` and the matching
file in `code/configs/protocols/`.
Use the `sam-kem` CLI for ingestion, case preparation, prompt batches, cache
construction, manifest construction, single-case diagnosis, protocol checks,
and final validation.

Do not use `code/scripts/_*.py`, `code/src/v3_canonical.py`, or
`code/src/original/*.py` as alternatives.
