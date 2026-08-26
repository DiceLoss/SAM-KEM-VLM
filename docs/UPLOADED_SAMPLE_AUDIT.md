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
channel.

## Follow-up experiment on `data/raw_pet`

The later raw upload permits a provisional distribution experiment but is not
yet canonical input. Applying the declared BrainWeb 2x2 mean pooling to all v4
train slices gives minimum `0`, maximum `5.796875`, and 99.99th percentile
`5.431641`. Validation has maximum `5.213867`. Thus `C_BW=6` is a reasonable
rounded **data-derived** fixed-scale candidate with no train/validation clipping;
train-global min-max is nearly the same operation because its offset is zero.
It is not a physical calibration claim.

For HECKTOR, resizing the uploaded channel to 128x128 gives a provisional train
maximum of `26.973867` and validation maximum of `34.933079`. A train-max
min-max scale would clip `0.0242%` of validation pixels in one case. `C_HE=30`
would still clip `0.0154%` in that case, whereas `C_HE=35` covers train and
validation without clipping. `C_HE=30000` places the validation maximum at only
`0.001164`; it is incompatible with this uploaded representation even though C
cancels from the expected-count sinogram mathematically.

These values remain provisional for three reasons:

1. HECKTOR subject IDs are one-based, but `_make_raw_pet_samples.py` indexes the
   source as `input_arr[sid]`. Cross-matching uploaded masks proves the resulting
   keys are shifted: for example sample subject 107 exactly matches raw key 106,
   not raw key 107. Source index 0 is absent and subject 224 is skipped.
2. The script and manifest call channel 1 both PET and “HU scale,” while an older
   preprocessing file calls it CT. Its non-negative 0--35 distribution is
   consistent with a PET/SUV-like channel, but the original dataset metadata
   must settle the channel and unit claim.
3. BrainWeb was converted from float32 to float16 merely to split files, although
   each compressed float32 subject would fit below GitHub's file limit. The
   manifest still names a nonexistent combined `raw_pet_brainweb.npz` and records
   no source/array hashes.

The HE image square also does not fit the declared circular FOV: about `24.5%`
of provisional train activity lies outside the inscribed circle. Canonical
preprocessing must resize the entire 144x144 square to fit inside the circle and
center-pad it, or change and revalidate the acquisition geometry. Silently
discarding this mass is not acceptable.

Accordingly, `C_BW=6` and `C_HE=35` are candidates for a corrected train/validation
rerun, not values to paste into the locked configuration yet. The corrected raw
export must preserve float32, use `pid=sid-1`, omit segmentation targets and
locked-test data, identify the PET channel/unit, and record source plus array
SHA-256 values.
