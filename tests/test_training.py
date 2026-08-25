from pathlib import Path

import numpy as np
import pytest

from sam_kem.training import TrainingSample, balanced_sample_weights, validate_training_samples


def sample(subject, count):
  return TrainingSample("brainweb", subject, "s1", count, Path("x"), Path("y"), (0, 0, 1, 1))


def test_four_count_training_and_balancing():
  samples = [sample("a", 100000), sample("a", 100000), sample("b", 200000), sample("b", 500000), sample("b", 1000000)]
  selected = validate_training_samples(
    samples, dataset="brainweb", train_subjects={"a", "b"},
    count_levels=(100000, 200000, 500000, 1000000),
  )
  weights = balanced_sample_weights(selected)
  assert weights[0] == weights[1] == 0.5
  assert np.all(weights[2:] == 1.0)


def test_validation_subject_is_rejected():
  with pytest.raises(ValueError, match="non-train subject"):
    validate_training_samples(
      [sample("validation_subject", 100000)], dataset="brainweb",
      train_subjects={"train_subject"}, count_levels=(100000,),
    )

