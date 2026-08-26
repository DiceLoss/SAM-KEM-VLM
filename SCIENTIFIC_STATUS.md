# Scientific status

The CSV files currently under `results/` are retained for forensic comparison
only. They are not admissible as final paper evidence. They were produced before
the canonical implementation in `code/sam_kem/` and include known operator,
checkpoint, box-cache, normalization, coverage, and validation-sweep violations.

Development source preparation may proceed under the SHA-pinned 16/4 BrainWeb and
179/45 HECKTOR train/validation split. All 244 subjects are explicitly marked
as previously exposed development data; this split contains no `test_locked`
rows and cannot support final generalization claims. BrainWeb must expand every
source model into all finite, nonnegative, positive-activity axial planes; a
one-centre-slice manifest is invalid. The pinned development plane counts are
2815/696 for BrainWeb and 179/45 for HECKTOR. Step 3A found a zero training
minimum in both datasets. Train-global-minmax is therefore identical to the
train-derived fixed scale on every training array and differs only by validation
clipping. The GPU worker must train one BrainWeb and one HECKTOR decoder under
the non-clipping `train_max_fixed_C` configuration; the min-max arm remains a
diagnostic and must not receive duplicate training. Final evaluation remains
blocked until genuinely unexposed external subjects are acquired.

The generated development rule is named `divide_by_fixed_dataset_constant`.
The word “physical” is reserved for a future constant with external physical
provenance and must not label the present train-derived values.

The official MedSAM2 source is pinned at commit `332f30d420f1d1b08e2a79b3ae6a602458808383`
with `configs/sam2.1_hiera_t512.yaml`; the released checkpoint SHA is pinned in
the configuration. The fixed constant must match the declared activity unit. The uploaded HECKTOR
traceability arrays are labelled SUV and have maxima on the order of 4.5--15.2;
a constant around 30000 would only be meaningful for a different scanner-count
representation and must not be transplanted to these SUV arrays. The calibrated
development values are `C_BW=5.698180198669434` and
`C_HE=34.93264389038086`, fitted on train only. They are data-derived fixed
scales, not accepted physical constants.

No script may rename a failed or partial run to “final.” A run becomes eligible
for paper use only when `sam-kem validate` exits with code 0 and the raw-array,
prompt, training, sweep, checkpoint, and per-case manifests are archived together.

## PET simulation claim boundary

The canonical observation is an idealized independent count experiment:
`y_i ~ Poisson(s [A x]_i)`, where `s = N / sum(Ax)` and `N` is the expected
detected total count. Lower `N` is a low-count proxy; it is not a calibrated
injected activity or clinical dose. Poisson variation represents decay/detection
counting statistics, not detector thermal noise.

`A` is one mass-preserving, pixel-driven, non-TOF parallel-beam operator over
the complete square image grid. For a 128x128 image it uses 180 angles and 182
detector bins so the detector coordinate covers the image diagonal. The simulation and every reconstruction and
projection-domain metric use the identical implementation SHA. Attenuation,
scatter, random coincidences, detector sensitivity/normalization, PSF, TOF and
dead time are absent. HECKTOR starts from a clinical reconstructed PET image, so
its Radon/Poisson data are image-domain pseudo-sinograms, not scanner raw data or
a genuinely noise-free activity acquisition. The manuscript must use these
qualifications.

Image-domain metrics and KEM neighbourhoods use the same positive-sensitivity
full-square operator support. This geometry-derived support is fixed for every
method and does not use CT, MR, tumor labels, or any other anatomical mask.

Every axial slice is an independent 2-D ring-position experiment. The model has
no 3-D activity volume, axial detector response, oblique lines of response, or
cross-slice coincidence coupling. Adjacent slices may be used as separate cases,
but subject/source-model splitting remains mandatory and their statistical
dependence is handled by subject-level aggregation.

The former `EM.py`, `MydataLoader.py`, `preprocessing.py`, `torch_radon` and
`skimage.radon` generation paths are retired. Their generated arrays are
incompatible with the canonical protocol and must not be reused.

## Comparison-method naming

The implemented anatomy-neighbour baseline is a coefficient-space KEM whose
kernel support is selected by a Bowsher neighbourhood. It is named
`Bowsher-KEM`; it is not an unqualified classic penalized-likelihood MAP-Bowsher
implementation. MLEM-10 caches are recomputed with the locked operator and
observation model at evaluation time and must match before any method runs.

The MedSAM2 decoder artifact remains the continuous raw output. Kernel features
use per-prior min-max as declared. When the prior itself is displayed as an
image or enters the post-fusion convex combination, it is clipped to the fixed
physical target interval [0,1]; it is not min-max rescaled or passed through an
undocumented sigmoid. Inference resizes the official low-resolution decoder
logits directly to 128x128 using the same bilinear/`align_corners=False`
definition used for the L1 training loss; it does not resize an already
postprocessed 1024x1024 display mask.
