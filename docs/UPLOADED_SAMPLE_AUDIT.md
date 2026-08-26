# Uploaded PET sample operator and scale audit

This diagnostic uses the 32 committed samples per dataset at 200k expected
counts. It is not a paper result and does not select hyperparameters.

```bash
python code/experiments/audit_uploaded_samples.py
```

## Locked-operator checks

The 128x128, 180-angle operator has a relative adjoint error of
`2.04e-16` and maximum per-angle mass error of `2.88e-16`. It is therefore
internally consistent to floating-point precision.

The uploaded expected counts are not outputs of that operator. Their mean
relative L1 difference from count-normalized locked projections is `0.1228` for
BrainWeb and `0.3071` for HECKTOR. At 200k, the HECKTOR affine fit has mean slope
`0.8332` and intercept `1.4477` counts/bin. The intercept is approximately one
sixth of the mean expected bin count (`8.6806`), exactly matching the historical
20% flat-background construction: `0.2/(1+0.2)=1/6`.

For the first eight deterministic cases, 40-iteration MLEM with a newly sampled
matched observation gives mean NRMSE `0.4255` (BrainWeb) and `0.1379` (HECKTOR).
Feeding the uploaded observation to the locked reconstruction raises these to
`0.5775` and `0.2392`, respectively. This is a compatibility failure, not
evidence that either mathematical operator is a scanner model.

## Constant experiment

For both datasets, count-normalized expectations generated with `C=1` and
`C=30000` agree to relative L1 below `5e-16`. This is expected because
`N A(x/C) / sum(A(x/C))` cancels C. After removing a fixed positive iterate
floor from MLEM/OSEM/KEM, reconstructed images are also scale-equivariant to
about `2.5e-8` relative L1 in this numerical check.

Every uploaded target already has exact minimum 0 and maximum 1. Consequently,
global and per-image min-max make no numerical change to these arrays. The
samples therefore cannot identify the original C. BrainWeb extraction explicitly
divided each slice by its own maximum; the HECKTOR targets are approximately
binary and were loaded from the dataset `target` rather than the PET input
channel. Raw, train-split activity arrays are required for scale calibration.
