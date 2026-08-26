# HECKTOR DICOM and pixel-domain operator decision

Decision date: 2026-08-26.

The researcher confirmed that the HECKTOR input used by this study consists of
reconstructed PET slices exported directly from clinical DICOM. “Original” in
this provenance statement means the reconstructed clinical image exported from
DICOM; it does not mean detector list-mode events, coincidence records, or a
measured scanner sinogram.

The study therefore makes an explicitly image-derived simulation claim:

```text
reconstructed DICOM PET slice -> fixed-C scaling -> 2-D Radon projection
-> count-normalized Poisson draw -> reconstruction
```

The forward model is intentionally defined in pixel coordinates rather than as
a replica of a vendor scanner. Missing `PixelSpacing`, scanner reconstruction
diameter, and scanner-axis coordinates in `HECKTOR.npz` do not block this model.
If the DICOM headers remain available, those fields should still be archived as
source provenance and to verify quantitative units, but they do not parameterize
the Radon system matrix.

The former circular-FOV policy was rejected because it discarded 18--28% of the
activity in six traceability slices. The corrected canonical policy is:

- retain the complete DICOM-exported 144x144 square;
- resize it to 128x128 with pixel-centre-aligned bilinear interpolation;
- do not crop, recenter, or apply a circular corner mask;
- project with 180 angles and 182 detector bins, which cover the diagonal of the
  complete 128x128 grid;
- use the same matrix and exact transpose for simulation, MLEM/OSEM/KEM, and
  projection-domain metrics.

This decision resolves the former HECKTOR scanner-geometry ingestion blocker.
It does not resolve the quantitative normalization constant. The PET channel
unit, DICOM rescale/SUV conversion path, and fixed-scale candidate selection
must still be documented and validated independently.
