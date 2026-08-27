"""Build the canonical four-dimensional box cache from archived Qwen calls."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from .boxes import Box, BoxCache, BoxKey, BoxRecord, centroid_fallback
from .config import load_contract
from .noise import array_sha256
from .artifacts import file_sha256, verify_csv_artifact, verify_success_marker
from .qwen import instruction_sha256
from .qwen import load_request_policy


def build_box_cache(
  manifest_path: Path, response_dirs: list[Path] | tuple[Path, ...] | Path,
  output_path: Path, config_path: Path, policy_path: Path,
) -> Path:
  cfg, _ = load_contract(config_path)
  policy = load_request_policy(policy_path)
  policy_sha256 = file_sha256(policy_path)
  repo_root = config_path.resolve().parents[2]
  manifest_record = verify_csv_artifact(manifest_path)
  if manifest_record.get("schema_name") != "canonical_case_manifest_v3":
    raise ValueError("box-cache input must be a canonical v3 case manifest")
  responses: dict[BoxKey, dict] = {}
  directories = [response_dirs] if isinstance(response_dirs, Path) else list(response_dirs)
  if not directories:
    raise ValueError("at least one archived Qwen response directory is required")
  for directory in directories:
    verify_success_marker(directory)
  response_paths = sorted(
    path for directory in directories for path in directory.rglob("*.json")
    if path.name != "_SUCCESS.json"
  )
  for path in response_paths:
    document = json.loads(path.read_text(encoding="utf-8"))
    record = document.get("record", {})
    if not {"dataset", "subject_id", "slice_id", "count_level"}.issubset(record):
      continue
    key = BoxKey(record["dataset"], record["subject_id"], record["slice_id"], int(record["count_level"]))
    if record.get("run_id") != cfg["run_id"]:
      raise ValueError(f"Qwen response has wrong run_id: {path}")
    if (
      document.get("request_policy", {}).get("policy_id") != policy["policy_id"]
      or document.get("request_policy", {}).get("sha256") != policy_sha256
    ):
      raise ValueError(f"Qwen response has wrong request policy: {path}")
    if key in responses:
      raise ValueError(f"multiple Qwen calls for one cache key: {key}")
    responses[key] = record
  manifest = list(csv.DictReader(manifest_path.open("r", encoding="utf-8", newline="")))
  if not manifest or any(
    row.get("run_id") != cfg["run_id"]
    or row.get("split_id") != cfg["data"]["split_id"]
    or row.get("split_sha256") != cfg["data"]["split_sha256"]
    for row in manifest
  ):
    raise ValueError("box-cache manifest has the wrong clean run/split identity")
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
    image_path = repo_root / anchor_rows[0]["mlem10_input_path"]
    image = np.load(image_path)
    if response.get("source_mlem10_sha256") != array_sha256(image):
      raise ValueError(f"Qwen prompt image is not the declared anchor MLEM for {key}")
    prompt_image = Path(str(response.get("image_path", "")))
    if not prompt_image.is_absolute():
      prompt_image = repo_root / prompt_image
    prompt_metadata = prompt_image.with_suffix(".json")
    if not prompt_image.is_file() or not prompt_metadata.is_file():
      raise FileNotFoundError(f"archived Qwen prompt image/sidecar missing for {key}")
    render_record = json.loads(prompt_metadata.read_text(encoding="utf-8"))
    if (
      file_sha256(prompt_image) != response.get("image_sha256")
      or render_record.get("png_sha256") != response.get("image_sha256")
      or render_record.get("source_mlem10_sha256") != array_sha256(image)
      or response.get("prompt_sha256") != instruction_sha256(cfg["prompts"]["text"])
    ):
      raise ValueError(f"Qwen image/prompt provenance mismatch for {key}")
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
      source = (
        "centroid_fallback_qwen_no_detection"
        if response.get("source") == "qwen_no_detection"
        else "centroid_fallback_qwen_request_failure"
      )
      request_id = str(response.get("request_id", ""))
      image_sha = str(response.get("image_sha256", ""))
    box.validate(width=128, height=128)
    records.append(BoxRecord(
      key, cfg["run_id"], box, source, request_id, image_sha,
      str(response["source_mlem10_sha256"]),
      str(response.get("prompt_sha256", "")), str(response.get("model", "")),
      str(policy["policy_id"]), policy_sha256,
    ))
  BoxCache(records).write(output_path)
  return output_path
