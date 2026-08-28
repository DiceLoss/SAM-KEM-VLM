import csv
import importlib.util
import json
from pathlib import Path

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "code/experiments/audit_qwen_box_semantics.py"
SPEC = importlib.util.spec_from_file_location("audit_qwen_box_semantics", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_semantic_audit_selects_quantiles_and_failures_without_changing_images(tmp_path: Path):
  prompt_sha = "a" * 64
  model = "qwen3-vl-plus"
  manifest = tmp_path / "prompt_manifest.csv"
  fields = [
    "dataset", "subject_id", "slice_id", "count_level", "split",
    "prompt_image_path", "prompt_image_sha256", "prompt_sha256", "model",
  ]
  prompt_rows = []
  responses = tmp_path / "responses"
  for subject, source, box in (
    ("s1", "qwen3-vl-plus", {"xmin": 40, "ymin": 40, "xmax": 90, "ymax": 90}),
    ("s2", "qwen_request_failure", None),
  ):
    image_path = tmp_path / f"images/{subject}.png"
    image_path.parent.mkdir(parents=True, exist_ok=True)
    array = np.zeros((128, 128), dtype=np.uint8)
    array[60:70, 60:70] = 255
    Image.fromarray(array).convert("RGB").save(image_path)
    image_sha = MODULE._sha256(image_path)
    prompt_rows.append({
      "dataset": "brainweb", "subject_id": subject, "slice_id": "z1",
      "count_level": 100000, "split": "train",
      "prompt_image_path": image_path.relative_to(tmp_path).as_posix(),
      "prompt_image_sha256": image_sha, "prompt_sha256": prompt_sha, "model": model,
    })
    response = responses / f"brainweb/{subject}/z1/c100000.json"
    response.parent.mkdir(parents=True, exist_ok=True)
    response.write_text(json.dumps({
      "record": {
        "dataset": "brainweb", "subject_id": subject, "slice_id": "z1",
        "count_level": 100000, "source": source, "attempt_count": 1,
        "request_id": "request-1" if box else "", "box": box,
        "image_sha256": image_sha, "prompt_sha256": prompt_sha, "model": model,
      }
    }), encoding="utf-8")
  with manifest.open("w", encoding="utf-8", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=fields)
    writer.writeheader()
    writer.writerows(prompt_rows)
  (responses / "_SUCCESS.json").write_text("{}\n", encoding="utf-8")
  summary = MODULE.audit(
    tmp_path, manifest, responses, tmp_path / "diagnostics.csv",
    tmp_path / "summary.json", tmp_path / "samples.csv", tmp_path / "sample_images",
  )
  assert summary["records"] == 2
  assert summary["visual_sample_count"] == 2
  group = summary["group_summaries"]["brainweb/c100000"]
  assert group["successes"] == 1 and group["failures"] == 1
  with (tmp_path / "samples.csv").open("r", encoding="utf-8", newline="") as handle:
    samples = list(csv.DictReader(handle))
  success = next(row for row in samples if row["source"] == "qwen3-vl-plus")
  assert success["sample_reason"] == "area_max+area_median+area_min+area_q25+area_q75"
  for row in samples:
    assert MODULE._sha256(tmp_path / row["sample_image_path"]) == row["image_sha256"]
