"""Qwen box-cache schema and deterministic centroid fallback."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from .artifacts import atomic_csv_writer, finalize_csv


@dataclass(frozen=True, order=True)
class BoxKey:
  dataset: str
  subject_id: str
  slice_id: str
  count_level: int


@dataclass(frozen=True)
class Box:
  xmin: int
  ymin: int
  xmax: int
  ymax: int

  def validate(self, *, width: int, height: int) -> None:
    if not (0 <= self.xmin < self.xmax <= width and 0 <= self.ymin < self.ymax <= height):
      raise ValueError(f"invalid box {self} for image {(height, width)}")


@dataclass(frozen=True)
class BoxRecord:
  key: BoxKey
  box: Box
  source: str
  request_id: str
  image_sha256: str
  source_mlem10_sha256: str
  prompt_sha256: str
  model: str


REQUIRED_COLUMNS = (
  "dataset", "subject_id", "slice_id", "count_level",
  "xmin", "ymin", "xmax", "ymax", "source", "request_id", "image_sha256",
  "source_mlem10_sha256", "prompt_sha256", "model",
)


class BoxCache:
  def __init__(self, records: list[BoxRecord]):
    self._records: dict[BoxKey, BoxRecord] = {}
    for record in records:
      if record.key in self._records:
        raise ValueError(f"duplicate box-cache key: {record.key}")
      self._records[record.key] = record

  @classmethod
  def read(cls, path: Path) -> "BoxCache":
    with path.open("r", encoding="utf-8", newline="") as handle:
      reader = csv.DictReader(handle)
      missing = set(REQUIRED_COLUMNS) - set(reader.fieldnames or ())
      if missing:
        raise ValueError(f"box cache missing columns: {sorted(missing)}")
      records = []
      for row in reader:
        box = Box(*(int(row[name]) for name in ("xmin", "ymin", "xmax", "ymax")))
        box.validate(width=128, height=128)
        records.append(BoxRecord(
          key=BoxKey(row["dataset"], row["subject_id"], row["slice_id"], int(row["count_level"])),
          box=box,
          source=row["source"], request_id=row["request_id"], image_sha256=row["image_sha256"],
          source_mlem10_sha256=row["source_mlem10_sha256"],
          prompt_sha256=row["prompt_sha256"], model=row["model"],
        ))
    return cls(records)

  def get(self, key: BoxKey) -> BoxRecord | None:
    return self._records.get(key)

  def __len__(self) -> int:
    return len(self._records)

  def write(self, path: Path) -> None:
    with atomic_csv_writer(path, list(REQUIRED_COLUMNS)) as writer:
      for key in sorted(self._records):
        record = self._records[key]
        writer.writerow({
          "dataset": key.dataset, "subject_id": key.subject_id,
          "slice_id": key.slice_id, "count_level": key.count_level,
          **record.box.__dict__, "source": record.source,
          "request_id": record.request_id, "image_sha256": record.image_sha256,
          "source_mlem10_sha256": record.source_mlem10_sha256,
          "prompt_sha256": record.prompt_sha256, "model": record.model,
        })
    finalize_csv(
      path, schema_name="qwen_anchor_box_cache_v1",
      primary_key=["dataset", "subject_id", "slice_id", "count_level"],
    )


def centroid_fallback(
  image: np.ndarray,
  *,
  box_width: int,
  box_height: int,
  relative_threshold: float = 0.5,
) -> Box:
  """Center a fixed-size box on the bright-pixel intensity centroid."""
  array = np.asarray(image, dtype=np.float64)
  if array.ndim != 2 or box_width <= 0 or box_height <= 0:
    raise ValueError("image must be 2-D and box dimensions must be positive")
  height, width = array.shape
  if box_width > width or box_height > height:
    raise ValueError("fallback box cannot exceed the image")
  peak = float(np.nanmax(array)) if array.size else 0.0
  selected = np.isfinite(array) & (array > relative_threshold * peak) if peak > 0 else np.zeros_like(array, bool)
  weights = np.where(selected, np.maximum(array, 0.0), 0.0)
  mass = float(weights.sum())
  if mass > 0:
    yy, xx = np.indices(array.shape)
    center_x = float((weights * xx).sum() / mass)
    center_y = float((weights * yy).sum() / mass)
  else:
    center_x, center_y = (width - 1) / 2.0, (height - 1) / 2.0
  xmin = min(max(int(round(center_x - box_width / 2)), 0), width - box_width)
  ymin = min(max(int(round(center_y - box_height / 2)), 0), height - box_height)
  return Box(xmin, ymin, xmin + box_width, ymin + box_height)
