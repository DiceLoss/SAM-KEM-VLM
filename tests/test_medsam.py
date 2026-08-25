import numpy as np

from sam_kem.boxes import Box
from sam_kem.medsam import predict_continuous_prior


class FakePredictor:
  def set_image(self, image):
    self.image = image

  def predict(self, **kwargs):
    assert kwargs["return_logits"] is True
    raw = np.linspace(-2, 2, 64, dtype=np.float32).reshape(1, 8, 8)
    return raw, np.array([0.9]), raw


def test_continuous_logits_are_requested_and_preserved():
  prior = predict_continuous_prior(
    FakePredictor(), np.arange(64, dtype=float).reshape(8, 8), Box(1, 1, 7, 7),
    output_shape=(8, 8), display_size=8,
  )
  assert prior.min() < 0 < prior.max()
  assert len(np.unique(prior)) > 2

