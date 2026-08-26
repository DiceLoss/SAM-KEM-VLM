"""Batch creation of one immutable r=0 VLM prompt and request per box key."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from .artifacts import (
  atomic_csv_writer, file_sha256, finalize_csv, verify_csv_artifact,
  verify_success_marker, write_success_marker,
)
from .config import load_yaml_strict, require_test_release
from .display import render_prompt_png
from .noise import array_sha256
from .qwen import instruction_sha256, request_and_archive_box


PROMPT_FIELDS = [
  "run_id", "dataset", "subject_id", "slice_id", "count_level", "split",
  "source_mlem10_path", "source_mlem10_sha256", "prompt_image_path",
  "prompt_image_sha256", "prompt_metadata_path", "prompt_sha256", "model",
]


def _safe_component(value: str) -> str:
  if not value or value in {".", ".."} or "/" in value or "\\" in value:
    raise ValueError(f"identity is unsafe as a path component: {value!r}")
  return value


def _verify_render(image_path: Path, source_path: Path) -> dict:
  sidecar = image_path.with_suffix(".json")
  image_record = json.loads(sidecar.read_text(encoding="utf-8"))
  source = np.load(source_path, allow_pickle=False)
  if (
    image_record.get("schema_version") != 2
    or image_record.get("artifact_kind") != "vlm_prompt_render"
    or image_record.get("source_mlem10_sha256") != array_sha256(source)
    or image_record.get("png_sha256") != file_sha256(image_path)
  ):
    raise ValueError(f"prompt rendering failed provenance verification: {image_path}")
  return image_record


def prepare_prompt_manifest(
  root: Path, config_path: Path, case_manifest: Path, output_dir: Path, *,
  splits: set[str], resume_verified: bool = False,
) -> Path:
  cfg = load_yaml_strict(config_path)
  if "test_locked" in splits:
    if splits != {"test_locked"}:
      raise ValueError("locked-test prompt images require a separate post-freeze batch")
    require_test_release(root, cfg)
  manifest_record = verify_csv_artifact(case_manifest)
  if manifest_record.get("schema_name") != "canonical_case_manifest_v3":
    raise ValueError("prompt preparation requires a canonical v3 case manifest")
  rows = list(csv.DictReader(case_manifest.open("r", encoding="utf-8", newline="")))
  anchor = int(cfg["prompts"]["anchor_realization"])
  selected = [
    row for row in rows
    if row["split"] in splits and int(row["realization_id"]) == anchor
  ]
  keys = [
    (row["dataset"], row["subject_id"], row["slice_id"], row["count_level"])
    for row in selected
  ]
  if (
    not selected or len(keys) != len(set(keys))
    or any(
      row.get("run_id") != cfg["run_id"]
      or row.get("split_id") != cfg["data"]["split_id"]
      or row.get("split_sha256") != cfg["data"]["split_sha256"]
      for row in selected
    )
  ):
    raise ValueError("prompt batch requires exactly one anchor row per box-cache key")
  output_manifest = output_dir / "prompt_manifest.csv"
  output_dir.mkdir(parents=True, exist_ok=True)
  records = []
  for row in sorted(selected, key=lambda value: (
    value["dataset"], value["subject_id"], value["slice_id"], int(value["count_level"]),
  )):
    dataset = _safe_component(row["dataset"])
    subject = _safe_component(row["subject_id"])
    slice_id = _safe_component(row["slice_id"])
    source = root / row["mlem10_input_path"]
    image = output_dir / dataset / subject / slice_id / f"c{int(row['count_level'])}.png"
    sidecar = image.with_suffix(".json")
    exists = (image.is_file(), sidecar.is_file())
    if any(exists) and not all(exists):
      raise RuntimeError(f"partial prompt-render group requires quarantine: {image}")
    if all(exists):
      if not resume_verified:
        raise FileExistsError(f"prompt rendering already exists: {image}")
      image_record = _verify_render(image, source)
    else:
      render_prompt_png(source, image)
      image_record = _verify_render(image, source)
    records.append({
      "run_id": cfg["run_id"], "dataset": dataset, "subject_id": subject,
      "slice_id": slice_id, "count_level": int(row["count_level"]),
      "split": row["split"], "source_mlem10_path": row["mlem10_input_path"],
      "source_mlem10_sha256": image_record["source_mlem10_sha256"],
      "prompt_image_path": image.relative_to(root).as_posix(),
      "prompt_image_sha256": image_record["png_sha256"],
      "prompt_metadata_path": sidecar.relative_to(root).as_posix(),
      "prompt_sha256": instruction_sha256(cfg["prompts"]["text"]),
      "model": cfg["prompts"]["model"],
    })
  with atomic_csv_writer(output_manifest, PROMPT_FIELDS) as writer:
    writer.writerows(records)
  finalize_csv(
    output_manifest, schema_name="vlm_prompt_manifest_v1",
    primary_key=["run_id", "dataset", "subject_id", "slice_id", "count_level"],
  )
  return output_manifest


def request_prompt_manifest(
  root: Path, config_path: Path, prompt_manifest: Path, response_dir: Path, *,
  resume_verified: bool = False,
) -> dict[str, int]:
  cfg = load_yaml_strict(config_path)
  record = verify_csv_artifact(prompt_manifest)
  if record.get("schema_name") != "vlm_prompt_manifest_v1":
    raise ValueError("box requests require a canonical prompt manifest")
  rows = list(csv.DictReader(prompt_manifest.open("r", encoding="utf-8", newline="")))
  if not rows or any(row.get("run_id") != cfg["run_id"] for row in rows):
    raise ValueError("prompt manifest is empty or belongs to another run")
  if any(row["split"] == "test_locked" for row in rows):
    require_test_release(root, cfg)
  if (response_dir / "_SUCCESS.json").is_file():
    if not resume_verified:
      raise FileExistsError(f"Qwen request batch is already complete: {response_dir}")
    verify_success_marker(response_dir)
    return {"requested": 0, "reused": len(rows), "failed": 0}
  outputs = []
  summary = {"requested": 0, "reused": 0, "failed": 0}
  for row in rows:
    image = root / row["prompt_image_path"]
    output = (
      response_dir / _safe_component(row["dataset"])
      / _safe_component(row["subject_id"]) / _safe_component(row["slice_id"])
      / f"c{int(row['count_level'])}.json"
    )
    if output.is_file():
      if not resume_verified:
        raise FileExistsError(f"Qwen attempt already archived: {output}")
      archived = json.loads(output.read_text(encoding="utf-8")).get("record", {})
      if (
        archived.get("run_id") != cfg["run_id"]
        or archived.get("image_sha256") != row["prompt_image_sha256"]
        or archived.get("prompt_sha256") != row["prompt_sha256"]
        or archived.get("model") != row["model"]
      ):
        raise ValueError(f"archived Qwen attempt failed resume verification: {output}")
      summary["reused"] += 1
    else:
      _, success = request_and_archive_box(
        image, output, cfg=cfg, dataset=row["dataset"],
        subject_id=row["subject_id"], slice_id=row["slice_id"],
        count_level=int(row["count_level"]),
      )
      summary["requested"] += 1
      summary["failed"] += int(not success)
    outputs.append(output)
  write_success_marker(response_dir, [
    prompt_manifest,
    prompt_manifest.with_suffix(prompt_manifest.suffix + ".manifest.json"),
    *outputs,
  ])
  return summary
