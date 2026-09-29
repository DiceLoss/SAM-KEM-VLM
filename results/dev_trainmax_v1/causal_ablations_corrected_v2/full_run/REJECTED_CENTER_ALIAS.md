# REJECTED_CENTER_ALIAS — causal_ablations_corrected_v2 (read-only)

Per directive 5890879059 (independent audit by `DiceLoss`, 2026-09-29).

## Status

This directory and its sub-files are preserved byte-for-byte and are
**not accepted as paper evidence**. None of its endpoints may be reused
in the formal v3 package. The Stage C `PASS` claim made on 2026-09-29
(comment `5887217293`) is retracted.

## Reasons

1. `build_center_box` in `code/sam_kem/causal_ablation_v2.py:75-85`
   reconstructed the same-size box around the **Qwen-box center** rather
   than the **image geometric center**, so `CENTER_ADAPT` was an exact
   alias of `Q_ADAPT` and not a real control.
2. The committed tests hard-coded `D:/MIAProject/SAM_KEM_QWEN3_V3_REPO`,
   so `make quality` does not reproduce from a clean Linux checkout.
3. Reported runner SHA-256 (`2528acf99437...`) did not match the
   actual file SHA-256 (`bcc569af63de...`) at pushed commit `b5817e5`,
   and the v2 results were never committed to the repo.
4. Required acceptance analyses were absent (Q_ADAPT exact
   reproduction at 5e-7, subject-balanced means, paired deltas,
   bootstrap, kernel diagnostics, fixed-case arrays, timing).

## Containment

- This directory is **not deleted**; all per-variant `per_case.csv` and
  `integrity_audit.json` and `summary.json` are preserved as
  historical artifacts.
- New formal v3 package lives in
  `results/dev_trainmax_v1/causal_ablations_corrected_v3/`.
- No byte in this directory has been or will be mutated.

## Worker note

Re-runs of the formal v3 package will produce a separate, independently
auditable evidence archive (committed to `delivery/` or attached as a
GitHub release asset), bound by full 64-character SHA-256 values and
reproducible from the committed v3 implementation.