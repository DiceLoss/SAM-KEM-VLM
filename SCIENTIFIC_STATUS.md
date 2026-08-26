# Scientific status

The CSV files currently under `results/` are retained for forensic comparison
only. They are not admissible as final paper evidence. They were produced before
the canonical implementation in `code/sam_kem/` and include known operator,
checkpoint, box-cache, normalization, coverage, and validation-sweep violations.

The canonical rerun is blocked until the experimenter supplies both fixed
physical scale constants in `code/configs/experiment.yaml` and the dataset-specific
MedSAM2 decoder checkpoint files with verified SHA-256 values. The validator is
designed to fail until these requirements and all result-integrity checks pass.

No script may rename a failed or partial run to “final.” A run becomes eligible
for paper use only when `sam-kem validate` exits with code 0 and the raw-array,
prompt, training, sweep, checkpoint, and per-case manifests are archived together.

## PET simulation claim boundary

The canonical observation is an idealized independent count experiment:
`y_i ~ Poisson(s [A x]_i)`, where `s = N / sum(Ax)` and `N` is the expected
detected total count. Lower `N` is a low-count proxy; it is not a calibrated
injected activity or clinical dose. Poisson variation represents decay/detection
counting statistics, not detector thermal noise.

`A` is one mass-preserving, pixel-driven, non-TOF parallel-beam operator with an
inscribed circular field of view. The simulation and every reconstruction and
projection-domain metric use the identical implementation SHA. Attenuation,
scatter, random coincidences, detector sensitivity/normalization, PSF, TOF and
dead time are absent. HECKTOR starts from a clinical reconstructed PET image, so
its Radon/Poisson data are image-domain pseudo-sinograms, not scanner raw data or
a genuinely noise-free activity acquisition. The manuscript must use these
qualifications.

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
undocumented sigmoid.
