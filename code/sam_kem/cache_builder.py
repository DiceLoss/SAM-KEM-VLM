"""Build the canonical four-dimensional box cache from archived Qwen calls."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from .boxes import Box, BoxCache, BoxKey, BoxRecord, centroid_fallback
from .config import load_contract
from .noise import array_sha256


def build_box_cache(
  manifest_path: Path, response_dir: Path, output_path: Path, config_path: Path,
) -> Path:
  cfg, _ = load_contract(config_path)
  responses: dict[BoxKey, dict] = {}
  for path in sorted(response_dir.glob("*.json")):
    document = json.loads(path.read_text(encoding="utf-8"))
    record = document.get("record", {})
    if not {"dataset", "subject_id", "slice_id", "count_level"}.issubset(record):
      continue
    key = BoxKey(record["dataset"], record["subject_id"], record["slice_id"], int(record["count_level"]))
    if key in responses:
      raise ValueError(f"multiple Qwen calls for one cache key: {key}")
    responses[key] = record
  manifest = list(csv.DictReader(manifest_path.open("r", encoding="utf-8", newline="")))
  required_keys = {
    BoxKey(row["dataset"], row["subject_id"], row["slice_id"], int(row["count_level"]))
    for row in manifest
  }
  records = []
  for key in sorted(required_keys):
    response = responses.get(key)
    if response is None:
      raise ValueError(f"no archived Qwen attempt for cache key: {key}")
    anchor_rows = [
      row for row in manifest
      if row["dataset"] == key.dataset and row["subject_id"] == key.subject_id
      and row["slice_id"] == key.slice_id and int(row["count_level"]) == key.count_level
      and int(row["realization_id"]) == int(cfg["prompts"]["anchor_realization"])
    ]
    if len(anchor_rows) != 1:
      raise ValueError(f"expected one anchor realization for {key}, got {len(anchor_rows)}")
    image_path = manifest_path.parents[2] / anchor_rows[0]["mlem10_input_path"]
    image = np.load(image_path)
    if response.get("source_mlem10_sha256") != array_sha256(image):
      raise ValueError(f"Qwen prompt image is not the declared anchor MLEM for {key}")
    if response is not None and response.get("box") is not None:
      box = Box(**{name: int(response["box"][name]) for name in ("xmin", "ymin", "xmax", "ymax")})
      source = "qwen3-vl-plus"
      request_id = str(response["request_id"])
      image_sha = str(response["image_sha256"])
      if str(response.get("model")) != str(cfg["prompts"]["model"]):
        raise ValueError(f"wrong VLM model for {key}: {response.get('model')}")
    else:
      fallback_size = cfg["prompts"]["fallback_box_size"]
      box = centroid_fallback(
        image, box_width=int(fallback_size[0]), box_height=int(fallback_size[1]),
        relative_threshold=float(cfg["prompts"]["fallback_relative_threshold"]),
      )
      source = "centroid_fallback_qwen_failure"
      request_id = str(response.get("request_id", ""))
      image_sha = str(response.get("image_sha256", ""))
    box.validate(width=128, height=128)
    records.append(BoxRecord(
      key, box, source, request_id, image_sha,
      str(response["source_mlem10_sha256"]),
      str(response.get("prompt_sha256", "")), str(response.get("model", "")),
    ))
  BoxCache(records).write(output_path)
  return output_path
