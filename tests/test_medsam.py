import numpy as np
from sam_kem.boxes import Box
from sam_kem.medsam import (
  freeze_except_mask_decoder,
  predict_continuous_prior,
  predict_prompt_object_logits,
)
from sam_kem.medsam_backend import MODEL_CONFIGS
from sam_kem.medsam_training import _box_coordinates_on_model_grid
from sam_kem.stage7a_prompt_plan import PromptObject


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


def test_prompt_object_decoder_preserves_points_and_continuous_logits():
  predictor = FakePredictor()
  obj = PromptObject(1, (1, 1, 7, 7), ((3, 3),), ((0, 0),), 1.0)
  decoded = predict_prompt_object_logits(
    predictor, np.arange(64, dtype=float).reshape(8, 8), (obj,),
    output_shape=(8, 8), display_size=8,
  )
  assert decoded[0].shape == (8, 8)
  assert decoded[0].dtype == np.float32
  assert len(np.unique(decoded[0])) > 2


def test_only_exact_sam_mask_decoder_namespace_is_trainable():
  class Parameter:
    def __init__(self, count):
      self.requires_grad = True
      self._count = count

    def numel(self):
      return self._count

  parameters = {
    "image_encoder.mask_decoder_adapter.weight": Parameter(2),
    "sam_prompt_encoder.weight": Parameter(3),
    "sam_mask_decoder.layers.0.weight": Parameter(4),
    "sam_mask_decoder.iou_prediction_head.layers.0.weight": Parameter(5),
    "sam_mask_decoder.output_hypernetworks_mlps.1.layers.0.weight": Parameter(6),
  }

  class Model:
    def named_parameters(self):
      return parameters.items()

  freeze_except_mask_decoder(Model())
  assert parameters["sam_mask_decoder.layers.0.weight"].requires_grad
  assert not parameters["sam_mask_decoder.iou_prediction_head.layers.0.weight"].requires_grad
  assert not parameters[
    "sam_mask_decoder.output_hypernetworks_mlps.1.layers.0.weight"
  ].requires_grad
  assert not parameters["image_encoder.mask_decoder_adapter.weight"].requires_grad
  assert not parameters["sam_prompt_encoder.weight"].requires_grad


def test_training_box_is_scaled_to_official_medsam2_512_grid():
  coordinates = _box_coordinates_on_model_grid(Box(16, 32, 96, 112), (128, 128), 512)
  np.testing.assert_array_equal(
    coordinates, np.array([[64, 128, 384, 448]], dtype=np.float32),
  )


def test_official_medsam2_configuration_path_is_used():
  assert MODEL_CONFIGS["sam2.1_hiera_t"] == "configs/sam2.1_hiera_t512.yaml"
