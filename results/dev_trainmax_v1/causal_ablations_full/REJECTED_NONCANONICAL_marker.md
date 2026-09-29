# REJECTED_NONCANONICAL marker (read-only)

This directory `results/dev_trainmax_v1/causal_ablations_full/` contains the
**quarantined read-only residue** of the prior 7-variant run that was rejected
by researcher directive 5882184688 (2026-09-29).

Do NOT use any artifact under this directory as paper evidence.

The rejection reasons (see the directive):

1. The Q_SHUFFLED_MATCHED mapping used a cyclic successor within
   (dataset, count, realization). On BrainWeb this left 692/696 assignments
   within the same subject; it is therefore NOT a zero-same-subject
   intervention.

2. UNIFORM_FEATURE_MATCHED (a mandatory uniform-kernel control) was missing;
   7 variants / 62,244 endpoints cannot answer the learned-affinity
   mechanism question.

3. The run was launched from untracked source files (`code/sam_kem/causal_ablation.py`
   and `code/experiments/run_validation_only_causal_ablations.py`); a formal
   run must be reproducible from a pushed commit.

The corrected 8-variant protocol with the same-slice / cross-subject rotation,
UNIFORM_FEATURE_MATCHED control, and pushed-commit reproducibility lives
under `results/dev_trainmax_v1/causal_ablations_corrected_v2/`.
