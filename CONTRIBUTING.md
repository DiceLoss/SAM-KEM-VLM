# Contributing to the clean pipeline

Keep each scientific change reviewable and traceable.

1. Work on a named branch from a clean commit.
2. Put reusable behavior in `code/sam_kem/` and thin orchestration in
   `code/experiments/`.
3. Add or update tests for numerical, schema, or leakage-sensitive behavior.
4. Keep scientific choices in strict YAML protocols/configuration.
5. Run `make quality` before committing.
6. Use an imperative commit subject and describe any affected downstream
   artifacts in the body or pull request.

Do not import retired v3 modules, edit generated CSV/JSON files by hand, commit
secrets or patient data, reuse an output directory, weaken the locked-test gate,
or make a partial run paper-eligible.

A scientific change to normalization, split, operator, prompt, checkpoint,
sampling, method definition, metric support, or hyperparameter selection needs a
new run ID and regeneration of all affected descendants.

Documentation uses repository-relative Markdown links.
Any new active entry point or artifact type must be added to the repository map,
data dictionary/experiment matrix as applicable, and artifact specification.
