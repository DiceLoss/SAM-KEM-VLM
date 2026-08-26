# PET simulation and reconstruction-method audit

## Canonical scientific claim

For a fixed-C normalized 2-D PET image `x`, the idealized observation is

```text
p = A x
s = N / sum(p)
y_i independently follows Poisson(s p_i)
```

`N` is the expected detected total count per slice experiment. It is a
low-count proxy, not a calibrated injected activity or clinical dose. Poisson
variation represents radioactive decay/detection counting statistics; it must
not be attributed to detector thermal noise.

`A` is the same locked 128x128, 180-angle, pixel-driven parallel-beam operator
in simulation, MLEM/OSEM/KEM and projection-domain deviance. It conserves
circular-FOV image mass at every angle and its backprojector is the exact sparse
transpose. It omits attenuation, scatter, randoms, sensitivity normalization,
PSF, TOF, dead time, axial response and oblique lines of response.

BrainWeb and HECKTOR use independent 2-D ring positions. HECKTOR begins with a
clinical reconstructed PET image, so its generated projection is a
pseudo-sinogram, not scanner raw data or a truly noise-free activity acquisition.

## Retired invalid paths

- `EM.py::PET2Sinogram` and `MydataLoader.py::PET2Sinogram` added an undeclared
  random detector gain and a flat additive background before Poisson sampling.
- `preprocessing.py::pet2sinogram` used a different `skimage.radon` geometry.
- historical WSL scripts mixed `torch_radon`, discrete Radon, 128/180 angles and
  Python `hash()` seeds.
- the historical Poisson audit treated spatial variation between heterogeneous
  projection bins as noise variance.

These functions now fail with a retirement message. Arrays/results produced by
them are not compatible with the canonical rerun.

## Provenance gates now enforced

Each target/projection, Poisson sinogram, MLEM-10 input, prompt rendering and
MedSAM2 prior has a JSON sidecar binding its exact array hash to the operator,
case identity, RNG seed, count scale, checkpoint and/or box as applicable. The
final manifest re-verifies the sidecars and the repository audit reopens and
hashes the arrays. A cached MLEM-10 is also recomputed numerically before an
evaluation method may use it.

The prompt box key is `(dataset, subject, slice, count)`. Qwen is called on the
configured anchor realization only and the same box is shared across
realizations. The centroid-box ablation now follows the same anchor/share rule,
so it changes only the box source.

## Method definitions and fair selection

| Method | Canonical implementation and selection |
|---|---|
| MLEM | Matched Poisson EM; stopping iteration selected on validation, independently of fixed prompt MLEM-10. |
| OSEM | Interleaved angular subsets; complete cycles selected on validation and total subset updates reported. |
| SelfKEM | Coefficient-space `x=K alpha`; kernel feature is fixed prompt MLEM-10. |
| MR-KEM / CT-KEM | Same coefficient-space update; only named anatomy baseline receives MR/CT. |
| Bowsher-KEM | Anatomy-selected neighbour coefficient kernel. This is not claimed as classic penalized MAP-Bowsher. Neighbours, radius and iterations are selected on validation. |
| SAM-KEM | Raw continuous decoder output is per-prior min-maxed only for kernel construction. Image display/post-fusion uses fixed-range clipping, not min-max or sigmoid. Sigma, radius, iterations and fusion alpha are selected on validation. |

All applicable methods use identical cases, realizations, measured sinograms,
projection gain, forward operator and target scale. Parameter selection is
subject-averaged validation only. A selected searchable grid boundary forces an
expanded rerun. Locked-test convergence is descriptive and cannot choose an
iteration. Cross-count summaries first average within subject, then across
subjects, preventing adjacent slices from becoming independent replicates.

## Inputs still missing from this checkout

The scientific validator intentionally remains fatal until the experimenter
provides and records:

1. exact fixed `C_BrainWeb` and scanner-derived `C_HECKTOR`;
2. a real MedSAM2/SAM2 Git checkout path and its immutable commit;
3. the official released MedSAM2 checkpoint and SHA-256;
4. dataset-specific trained decoder checkpoints and sidecars;
5. regenerated canonical arrays and their manifests;
6. the new one-key-per-slice/count Qwen cache;
7. validation sweep, frozen parameters and clean-run metadata.

Do not set `locked: true` or use old result CSVs to satisfy these checks. Follow
`docs/EXECUTION.md`; after every required value/artifact is present, the final
eligibility command is:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest -q
sam-kem validate --root . --config code/configs/experiment.yaml
```

Both commands must succeed before numbers are transferred into the manuscript.

## Normalization conflict found in the historical code

The historical sources do not establish one common scale. One HECKTOR/raw loader
uses `clip(target/30000,0,1)`, a later HECKTOR path reads an already normalized
and clipped `target.npy`, and the BrainWeb loader calls per-image `normalize`.
Consequently, `C=1`, `C=30000`, and per-image min-max describe different source
representations rather than interchangeable numerical choices.

The new normalization calibration protocol fits data-derived candidates from
train subjects only and inspects validation clipping without reading locked test.
Its global min-max arm is a sensitivity study; its per-image arm is diagnostic
only. A physical fixed C remains preferable when scanner/simulation provenance
exists. If a data-derived scale is selected, the manuscript must say that it is
not a preserved physical PET scale, and every decoder and reconstruction
hyperparameter must be retrained/reselected under that scale.
