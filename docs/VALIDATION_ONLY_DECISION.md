# Validation-only study-scope decision

Decision date: 2026-09-15

The `dev_trainmax_v1` study is closed as a development/validation-only study.
It has no independent, untouched test cohort. This is a scientific scope
decision, not permission to relabel previously exposed subjects.

The evidence is unambiguous:

- `inputs/clean_v1/subject_inventory.csv` contains 244 subjects and every row
  has `previous_exposure_status=development_allowed` (SHA-256
  `1d55fc3e5e02111fa026fc1117c9078fad96e8b5c036ff896949ca64bba92470`).
- `inputs/clean_v1/subject_split.csv` contains BrainWeb 16 train / 4
  validation / 0 test and HECKTOR 179 train / 45 validation / 0 test
  (SHA-256
  `cffe83608da27f736abe02e8c0fa5d2f892df8773bbccb08ec1a916157c0a096`).
- `inputs/clean_v1/source_activity_manifest.csv` contains no `test_locked`
  source row (SHA-256
  `2ee28303d359ed7d967c6ac35e6291602665a135fa69357b8bfd228afbc0e02a`).
- The validation freeze completed with audit SHA-256
  `13bb3e7d45336c824efb4d6f5110dbcf7fd43b41552960a77174b21dfa779510`
  and frozen-parameter CSV SHA-256
  `4b7624c0339b4e387a7c7a1e52ed359f757a89516de437a6a8f73871f12c204e`.

Consequences:

1. Do not create a `locked_for_execution` configuration for this run.
2. Do not execute the test-release path or any protocol whose split is
   `test_locked`.
3. Do not reuse the exposed historical v3/v4 test results as an independent
   test evaluation.
4. Downstream reporting may only reuse the reviewed validation sweep at the
   frozen parameter tuples. It must label those results as descriptive internal
   validation and state that selection and reporting use the same split.
5. Claims of independent test performance, external generalization, one-time
   final-test evaluation, and unbiased post-selection performance are
   prohibited.

The corresponding researcher instruction is recorded in GitHub issue #2,
comment `5673717940`.
