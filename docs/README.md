# Documentation index

The repository has one active `clean_v1` workflow and one retained v3 forensic
record.
Read documents in the order appropriate to your role.

## Researchers running experiments

1. Read [`RESEARCHER_HANDOFF_ZH.md`](RESEARCHER_HANDOFF_ZH.md) for the current
   blockers and the exact deliverables expected from the experimenter.
2. Read [`SCIENTIFIC_STATUS.md`](../SCIENTIFIC_STATUS.md) before making any PET,
   low-count, dose, or clinical claim.
3. Use [`EXECUTION.md`](EXECUTION.md) as the canonical end-to-end command guide.
4. Use [`OPERATIONS_RUNBOOK.md`](OPERATIONS_RUNBOOK.md) when resuming, archiving,
   diagnosing, or handing off a run.
5. Verify every result against [`ARTIFACT_SPEC.md`](ARTIFACT_SPEC.md).

## Reviewers and maintainers

- [`REPOSITORY_MAP.md`](REPOSITORY_MAP.md) separates active code, diagnostic
  samples, forensic files, generated artifacts, and external archives.
- [`DATA_DICTIONARY.md`](DATA_DICTIONARY.md) defines source-manifest fields,
  units, geometry, case keys, array layout, and provenance sidecars.
- [`EXPERIMENT_MATRIX.md`](EXPERIMENT_MATRIX.md) defines every experiment's
  split, methods, counts, selection role, prerequisites, and output.
- [`CLEAN_PIPELINE.md`](CLEAN_PIPELINE.md) gives the compact implementation
  boundary.
- [`PET_AND_METHOD_AUDIT.md`](PET_AND_METHOD_AUDIT.md) records the scientific
  faults that the clean implementation prevents.
- [`UPLOADED_SAMPLE_AUDIT.md`](UPLOADED_SAMPLE_AUDIT.md) records diagnostics run
  on the committed sample archives.

## Forensic documents

- [`decisions.md`](decisions.md) records retired v3 decisions.
- [`v3_workflow.md`](v3_workflow.md) describes the retired v3 workflow.

Forensic documents explain old artifacts; they are not execution instructions.
The historical CSVs under `results/metrics/` and `results/convergence/` are not
paper evidence.

## Single source of truth

When documents disagree, apply this precedence order:

1. `code/configs/experiment.yaml` after it has been reviewed, SHA-pinned, and
   changed to `locked: true`;
2. the strict protocol YAML for the experiment being run;
3. tested behavior in `code/sam_kem/`;
4. `EXECUTION.md` and `ARTIFACT_SPEC.md`;
5. all other active documentation;
6. forensic v3 material, which has no authority over a clean run.

Do not resolve a contradiction by silently changing generated data.
Open a new run ID, record the decision and rerun every affected downstream
artifact.
