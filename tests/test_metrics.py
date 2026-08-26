import numpy as np
import pytest

from sam_kem.metrics import mae, psnr, ssim


def test_geometry_mask_excludes_unobservable_pixels_from_image_metrics():
  pytest.importorskip("skimage")
  target = np.zeros((32, 32), dtype=float)
  target[10:22, 10:22] = 0.5
  estimate = target.copy()
  estimate[:2, :2] = 1.0
  support = np.zeros_like(target, dtype=bool)
  support[8:24, 8:24] = True
  assert mae(target, estimate, support) == 0.0
  assert np.isinf(psnr(target, estimate, data_range=1.0, mask=support))
  assert np.isclose(ssim(target, estimate, data_range=1.0, mask=support), 1.0)
