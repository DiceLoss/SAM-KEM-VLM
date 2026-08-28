"""Audit Qwen box geometry and select deterministic images for visual review."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image


DIAGNOSTIC_FIELDS = [
  "dataset", "subject_id", "slice_id", "count_level", "split", "source",
  "attempt_count", "request_id", "response_path", "response_sha256",
  "image_path", "image_sha256", "xmin", "ymin", "xmax", "ymax", "width",
  "height", "box_area_fraction", "brightest_x", "brightest_y",
  "brightest_inside_box", "top_10_percent_recall", "top_5_percent_recall",
  "top_1_percent_recall",
]
SAMPLE_FIELDS = [
  "dataset", "subject_id", "slice_id", "count_level", "split", "source",
  "sample_reason", "sample_image_path", "image_sha256", "response_path",
  "response_sha256", "xmin", "ymin", "xmax", "ymax", "box_area_fraction",
  "brightest_inside_box", "top_10_percent_recall", "top_5_percent_recall",
  "top_1_percent_recall",
]


def _sha256(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as handle:
    for chunk in iter(lambda: handle.read(1024 * 1024), b""):
      digest.update(chunk)
  return digest.hexdigest()


def _write_csv(path: Path, fields: list[str], rows: list[dict]) -> None:
  temporary = path.with_suffix(path.suffix + ".tmp")
  with temporary.open("w", encoding="utf-8", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
  temporary.replace(path)


def _write_json(path: Path, value: dict) -> None:
  temporary = path.with_suffix(path.suffix + ".tmp")
  temporary.write_text(
    json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8",
  )
  temporary.replace(path)


def _metric_summary(values: list[float]) -> dict:
  if not values:
    return {"n": 0}
  array = np.asarray(values, dtype=np.float64)
  return {
    "n": len(values),
    "mean": float(array.mean()),
    "quantiles": {
      str(percent): float(np.percentile(array, percent))
      for percent in (0, 1, 5, 25, 50, 75, 95, 99, 100)
    },
  }


def _diagnose_image(image_path: Path, box: dict | None) -> dict:
  with Image.open(image_path) as image:
    array = np.asarray(image.convert("L"))
  y_max, x_max = np.unravel_index(int(array.argmax()), array.shape)
  output = {
    "brightest_x": int(x_max), "brightest_y": int(y_max),
    "brightest_inside_box": "", "top_10_percent_recall": "",
    "top_5_percent_recall": "", "top_1_percent_recall": "",
  }
  if box is None:
    return output
  x1, y1, x2, y2 = (int(box[name]) for name in ("xmin", "ymin", "xmax", "ymax"))
  if not (0 <= x1 < x2 <= array.shape[1] and 0 <= y1 < y2 <= array.shape[0]):
    raise ValueError(f"invalid archived box for {image_path}: {box}")
  inside = np.zeros(array.shape, dtype=bool)
  inside[y1:y2, x1:x2] = True
  positive = array[array > 0]
  recalls = {}
  for percent in (90, 95, 99):
    threshold = float(np.percentile(positive, percent)) if positive.size else 0.0
    bright = array >= threshold if positive.size else np.zeros_like(array, dtype=bool)
    recalls[100 - percent] = (
      float(np.logical_and(bright, inside).sum() / bright.sum()) if bright.any() else 0.0
    )
  output.update({
    "brightest_inside_box": bool(inside[y_max, x_max]),
    "top_10_percent_recall": recalls[10],
    "top_5_percent_recall": recalls[5],
    "top_1_percent_recall": recalls[1],
  })
  return output


def audit(
  root: Path, prompt_manifest: Path, responses: Path, diagnostics_path: Path,
  summary_path: Path, samples_path: Path, sample_dir: Path,
) -> dict:
  for path in (diagnostics_path, summary_path, samples_path, sample_dir):
    if path.exists():
      raise FileExistsError(f"refusing to overwrite semantic audit output: {path}")
  success_marker = responses / "_SUCCESS.json"
  if not success_marker.is_file():
    raise FileNotFoundError(f"completed response marker missing: {success_marker}")
  with prompt_manifest.open("r", encoding="utf-8", newline="") as handle:
    prompts = list(csv.DictReader(handle))
  if not prompts:
    raise ValueError("prompt manifest is empty")
  diagnostics = []
  seen_response_paths = set()
  for prompt in prompts:
    response_path = (
      responses / prompt["dataset"] / prompt["subject_id"] / prompt["slice_id"]
      / f"c{int(prompt['count_level'])}.json"
    )
    if not response_path.is_file():
      raise FileNotFoundError(f"response missing: {response_path}")
    seen_response_paths.add(response_path.resolve())
    document = json.loads(response_path.read_text(encoding="utf-8"))
    record = document.get("record", {})
    identity = (
      prompt["dataset"], prompt["subject_id"], prompt["slice_id"],
      int(prompt["count_level"]),
    )
    archived_identity = (
      str(record.get("dataset", "")), str(record.get("subject_id", "")),
      str(record.get("slice_id", "")), int(record.get("count_level", -1)),
    )
    image_path = root / prompt["prompt_image_path"]
    image_sha = _sha256(image_path)
    if (
      archived_identity != identity
      or record.get("image_sha256") != prompt["prompt_image_sha256"]
      or image_sha != prompt["prompt_image_sha256"]
      or record.get("prompt_sha256") != prompt["prompt_sha256"]
      or record.get("model") != prompt["model"]
    ):
      raise ValueError(f"response/prompt identity mismatch: {response_path}")
    box = record.get("box")
    geometry = {
      "xmin": "", "ymin": "", "xmax": "", "ymax": "", "width": "",
      "height": "", "box_area_fraction": "",
    }
    if box is not None:
      x1, y1, x2, y2 = (int(box[name]) for name in ("xmin", "ymin", "xmax", "ymax"))
      geometry = {
        "xmin": x1, "ymin": y1, "xmax": x2, "ymax": y2,
        "width": x2 - x1, "height": y2 - y1,
        "box_area_fraction": float((x2 - x1) * (y2 - y1) / (128 * 128)),
      }
    diagnostics.append({
      "dataset": identity[0], "subject_id": identity[1], "slice_id": identity[2],
      "count_level": identity[3], "split": prompt["split"],
      "source": record.get("source", ""),
      "attempt_count": int(record.get("attempt_count", 0)),
      "request_id": record.get("request_id", ""),
      "response_path": response_path.relative_to(root).as_posix(),
      "response_sha256": _sha256(response_path),
      "image_path": image_path.relative_to(root).as_posix(),
      "image_sha256": image_sha,
      **geometry, **_diagnose_image(image_path, box),
    })
  actual_responses = {
    path.resolve() for path in responses.rglob("c*.json")
    if path.name != "_SUCCESS.json"
  }
  if actual_responses != seen_response_paths:
    raise ValueError("response directory contains missing or extra cache-key JSON files")

  grouped: dict[tuple[str, int], list[dict]] = defaultdict(list)
  for row in diagnostics:
    grouped[(row["dataset"], int(row["count_level"]))].append(row)
  group_summaries = {}
  selected: dict[tuple[str, str, str, int], set[str]] = defaultdict(set)
  quantile_labels = (("area_min", 0.0), ("area_q25", 0.25), ("area_median", 0.5),
                     ("area_q75", 0.75), ("area_max", 1.0))
  for (dataset, count), rows in sorted(grouped.items()):
    successful = [row for row in rows if row["source"] == "qwen3-vl-plus"]
    failures = [row for row in rows if row["source"] != "qwen3-vl-plus"]
    ordered = sorted(successful, key=lambda row: (
      float(row["box_area_fraction"]), row["subject_id"], row["slice_id"],
    ))
    for label, quantile in quantile_labels:
      index = round(quantile * (len(ordered) - 1))
      row = ordered[index]
      selected[(dataset, row["subject_id"], row["slice_id"], count)].add(label)
    for row in failures:
      selected[(dataset, row["subject_id"], row["slice_id"], count)].add("request_failure")
    group_summaries[f"{dataset}/c{count}"] = {
      "records": len(rows), "successes": len(successful), "failures": len(failures),
      "failure_rate": float(len(failures) / len(rows)),
      "box_area_fraction": _metric_summary([
        float(row["box_area_fraction"]) for row in successful
      ]),
      "brightest_pixel_inside_rate": float(np.mean([
        float(row["brightest_inside_box"]) for row in successful
      ])),
      "top_10_percent_recall": _metric_summary([
        float(row["top_10_percent_recall"]) for row in successful
      ]),
      "top_5_percent_recall": _metric_summary([
        float(row["top_5_percent_recall"]) for row in successful
      ]),
      "top_1_percent_recall": _metric_summary([
        float(row["top_1_percent_recall"]) for row in successful
      ]),
      "boxes_below_one_percent_image_area": sum(
        float(row["box_area_fraction"]) < 0.01 for row in successful
      ),
      "boxes_above_seventy_five_percent_image_area": sum(
        float(row["box_area_fraction"]) > 0.75 for row in successful
      ),
    }

  by_key = {
    (row["dataset"], row["subject_id"], row["slice_id"], int(row["count_level"])): row
    for row in diagnostics
  }
  sample_dir.mkdir(parents=True)
  sample_rows = []
  for key, reasons in sorted(selected.items()):
    row = by_key[key]
    source_image = root / row["image_path"]
    destination = (
      sample_dir / key[0] / key[1] / key[2] / f"c{key[3]}.png"
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_image, destination)
    if _sha256(destination) != row["image_sha256"]:
      raise RuntimeError(f"visual sample copy differs from source: {destination}")
    sample_rows.append({
      **{field: row[field] for field in SAMPLE_FIELDS if field in row},
      "sample_reason": "+".join(sorted(reasons)),
      "sample_image_path": destination.relative_to(root).as_posix(),
    })
  diagnostics_path.parent.mkdir(parents=True, exist_ok=True)
  samples_path.parent.mkdir(parents=True, exist_ok=True)
  _write_csv(diagnostics_path, DIAGNOSTIC_FIELDS, diagnostics)
  _write_csv(samples_path, SAMPLE_FIELDS, sample_rows)
  summary = {
    "schema_version": 1,
    "artifact_kind": "qwen_box_semantic_diagnostic",
    "diagnostic_only_not_ground_truth": True,
    "records": len(diagnostics),
    "source_counts": dict(sorted(Counter(row["source"] for row in diagnostics).items())),
    "prompt_manifest_sha256": _sha256(prompt_manifest),
    "response_success_marker_sha256": _sha256(success_marker),
    "diagnostics_csv_sha256": _sha256(diagnostics_path),
    "visual_sample_manifest_sha256": _sha256(samples_path),
    "visual_sample_count": len(sample_rows),
    "group_summaries": group_summaries,
    "failure_records": [
      {key: row[key] for key in (
        "dataset", "subject_id", "slice_id", "count_level", "source",
        "response_path", "response_sha256",
      )}
      for row in diagnostics if row["source"] != "qwen3-vl-plus"
    ],
  }
  _write_json(summary_path, summary)
  return summary


def main() -> int:
  parser = argparse.ArgumentParser()
  parser.add_argument("--root", type=Path, default=Path("."))
  parser.add_argument("--prompt-manifest", type=Path, required=True)
  parser.add_argument("--responses", type=Path, required=True)
  parser.add_argument("--output-diagnostics", type=Path, required=True)
  parser.add_argument("--output-summary", type=Path, required=True)
  parser.add_argument("--output-samples", type=Path, required=True)
  parser.add_argument("--sample-dir", type=Path, required=True)
  args = parser.parse_args()
  summary = audit(
    args.root.resolve(), args.prompt_manifest.resolve(), args.responses.resolve(),
    args.output_diagnostics.resolve(), args.output_summary.resolve(),
    args.output_samples.resolve(), args.sample_dir.resolve(),
  )
  print(json.dumps(summary, indent=2, sort_keys=True))
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
