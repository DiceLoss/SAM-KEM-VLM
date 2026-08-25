"""Training utilities for one balanced four-count-level decoder per dataset."""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class TrainingSample:
  dataset: str
  subject_id: str
  slice_id: str
  count_level: int
  input_path: Path
  target_path: Path
  box: tuple[int, int, int, int]


def validate_training_samples(
  samples: Iterable[TrainingSample],
  *,
  dataset: str,
  train_subjects: set[str],
  count_levels: tuple[int, ...],
) -> list[TrainingSample]:
  selected = list(samples)
  if not selected:
    raise ValueError("training set is empty")
  for sample in selected:
    if sample.dataset != dataset:
      raise ValueError("a dataset-specific decoder cannot mix datasets")
    if sample.subject_id not in train_subjects:
      raise ValueError(f"non-train subject in decoder training: {sample.subject_id}")
    if sample.count_level not in count_levels:
      raise ValueError(f"unexpected count level: {sample.count_level}")
  observed = {sample.count_level for sample in selected}
  if observed != set(count_levels):
    raise ValueError(f"training must cover all count levels: observed={sorted(observed)}")
  return selected


def balanced_sample_weights(samples: list[TrainingSample]) -> np.ndarray:
  counts = Counter(sample.count_level for sample in samples)
  return np.asarray([1.0 / counts[sample.count_level] for sample in samples], dtype=np.float64)


def train_decoder_epoch(
  model: Any,
  batches: Iterable[Any],
  optimizer: Any,
  forward_logits: Callable[[Any, Any], Any],
) -> tuple[float, dict[int, int]]:
  """Run one L1-only epoch; batch must expose input, target and count_level."""
  import torch

  model.train()
  loss_total = 0.0
  n_batches = 0
  realized_counts: Counter[int] = Counter()
  for batch in batches:
    optimizer.zero_grad(set_to_none=True)
    prediction = forward_logits(model, batch)
    target = batch.target.to(prediction.device, dtype=prediction.dtype)
    if prediction.shape != target.shape:
      raise ValueError(f"decoder output shape {prediction.shape} != target {target.shape}")
    loss = torch.nn.functional.l1_loss(prediction, target)
    loss.backward()
    optimizer.step()
    loss_total += float(loss.detach().cpu())
    n_batches += 1
    for value in np.asarray(batch.count_level).reshape(-1):
      realized_counts[int(value)] += 1
  if n_batches == 0:
    raise ValueError("training loader yielded no batches")
  return loss_total / n_batches, dict(realized_counts)
