import pytest
from experiments.audit_decoder_training import _validate_scientific_record


def _record():
  inventory = {
    "sam_image_encoder.weight": {"numel": 10, "requires_grad": False},
    "sam_mask_decoder.weight": {"numel": 7, "requires_grad": True},
  }
  return {
    "dataset": "brainweb", "completed_epochs": 2, "global_step": 8,
    "training_count_samples": {"100": 4, "200": 4},
    "training_subject_samples": {"a": 4, "b": 4},
    "train_subject_ids": ["a", "b"], "non_mask_decoder_trainable_names": [],
    "loss": "l1_only_raw_decoder_output", "optimizer": "AdamW", "batch_size": 1,
    "weight_decay": 0.01, "sampling_unit": "count_then_subject",
    "epochs": 2, "steps_per_epoch": 4, "learning_rate": 0.001,
    "box_fallback_samples": 1, "parameter_inventory": inventory,
    "trainable_parameter_names": ["sam_mask_decoder.weight"],
  }


def _config():
  return {
    "acquisition": {"count_levels": [100, 200]},
    "medsam2": {"training": {
      "epochs": 2, "steps_per_epoch": 4, "learning_rate": 0.001,
      "weight_decay": 0.01, "sampling_unit": "count_then_subject",
    }},
  }


def test_completed_training_scientific_record_is_summarized():
  result = _validate_scientific_record(_record(), _config(), "brainweb", {"a", "b"})
  assert result["global_step"] == 8
  assert result["training_count_samples"] == {100: 4, 200: 4}
  assert result["trainable_parameter_count"] == 7


def test_unbalanced_training_is_rejected():
  record = _record()
  record["training_count_samples"] = {"100": 5, "200": 3}
  with pytest.raises(ValueError, match="frozen scientific protocol"):
    _validate_scientific_record(record, _config(), "brainweb", {"a", "b"})
