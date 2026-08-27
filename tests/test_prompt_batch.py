import csv
from pathlib import Path

import numpy as np
import yaml

from sam_kem.artifacts import atomic_csv_writer, finalize_csv, verify_csv_artifact
from sam_kem.prompt_batch import _select_pilot_rows, prepare_prompt_manifest


ROOT = Path(__file__).resolve().parents[1]


def test_prompt_batch_uses_one_anchor_realization_per_box_key(tmp_path: Path):
  cfg = yaml.safe_load((ROOT / "code/configs/experiment.yaml").read_text(encoding="utf-8"))
  cfg["data"]["split_sha256"] = "a" * 64
  config_path = tmp_path / "experiment.yaml"
  config_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
  rows = []
  for realization in (0, 1):
    image_path = tmp_path / f"arrays/brainweb/s1/z1/c200000/r{realization}/mlem10.npy"
    image_path.parent.mkdir(parents=True)
    np.save(image_path, np.arange(128 * 128, dtype=np.float32).reshape(128, 128))
    rows.append({
      "run_id": "clean_v1", "split_id": cfg["data"]["split_id"],
      "split_sha256": "a" * 64, "dataset": "brainweb", "subject_id": "s1",
      "slice_id": "z1", "count_level": 200000, "realization_id": realization,
      "split": "train", "mlem10_input_path": image_path.relative_to(tmp_path).as_posix(),
    })
  manifest = tmp_path / "cases.csv"
  with atomic_csv_writer(manifest, list(rows[0])) as writer:
    writer.writerows(rows)
  finalize_csv(
    manifest, schema_name="canonical_case_manifest_v3",
    primary_key=[
      "run_id", "dataset", "subject_id", "slice_id", "realization_id", "count_level",
    ],
  )
  output = prepare_prompt_manifest(
    tmp_path, config_path, manifest, tmp_path / "prompts",
    splits={"train"},
  )
  record = verify_csv_artifact(output)
  assert record["schema_name"] == "vlm_prompt_manifest_v1"
  with output.open("r", encoding="utf-8", newline="") as handle:
    prompt_rows = list(csv.DictReader(handle))
  assert len(prompt_rows) == 1
  assert prompt_rows[0]["source_mlem10_path"].endswith("r0/mlem10.npy")


def test_qwen_pilot_selects_one_deterministic_key_per_dataset_and_count():
  rows = [
    {"dataset": dataset, "count_level": str(count), "subject_id": subject, "slice_id": "z1"}
    for subject in ("s2", "s1")
    for count in (200000, 100000)
    for dataset in ("hecktor", "brainweb")
  ]
  selected = _select_pilot_rows(rows, {
    "acquisition": {"count_levels": [100000, 200000]},
  })
  assert len(selected) == 4
  assert {(row["dataset"], int(row["count_level"])) for row in selected} == {
    ("brainweb", 100000), ("brainweb", 200000),
    ("hecktor", 100000), ("hecktor", 200000),
  }
  assert {row["subject_id"] for row in selected} == {"s1"}
