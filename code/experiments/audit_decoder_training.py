"""Audit completed dataset-specific decoder training groups before handoff."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from sam_kem.artifacts import atomic_write_json, file_sha256
from sam_kem.config import load_yaml_strict
from sam_kem.medsam_training import verify_completed_training
from sam_kem.splits import load_configured_subject_splits


def _validate_scientific_record(
  record: dict, cfg: dict, dataset: str, expected_subjects: set[str],
) -> dict:
  training = cfg["medsam2"]["training"]
  count_levels = {int(value) for value in cfg["acquisition"]["count_levels"]}
  counts = {int(key): int(value) for key, value in record["training_count_samples"].items()}
  subject_counts = {
    str(key): int(value) for key, value in record["training_subject_samples"].items()
  }
  trainable_names = set(record["trainable_parameter_names"])
  inventory_trainable = {
    name for name, item in record["parameter_inventory"].items()
    if item["requires_grad"]
  }
  expected_steps = int(training["epochs"]) * int(training["steps_per_epoch"])
  if (
    record.get("dataset") != dataset
    or record.get("completed_epochs") != int(training["epochs"])
    or record.get("global_step") != expected_steps
    or set(counts) != count_levels
    or len(set(counts.values())) != 1
    or sum(counts.values()) != expected_steps
    or set(record.get("train_subject_ids", [])) != expected_subjects
    or not subject_counts
    or set(subject_counts) != expected_subjects
    or max(subject_counts.values()) - min(subject_counts.values()) > len(count_levels)
    or not trainable_names
    or trainable_names != inventory_trainable
    or any(not name.startswith("sam_mask_decoder.") for name in trainable_names)
    or record.get("non_mask_decoder_trainable_names") != []
    or record.get("loss") != "l1_only_raw_decoder_output"
    or record.get("optimizer") != "AdamW"
    or float(record.get("weight_decay", -1)) != float(training["weight_decay"])
    or int(record.get("batch_size", -1)) != 1
    or record.get("sampling_unit") != training["sampling_unit"]
    or int(record.get("epochs", -1)) != int(training["epochs"])
    or int(record.get("steps_per_epoch", -1)) != int(training["steps_per_epoch"])
    or float(record.get("learning_rate", -1)) != float(training["learning_rate"])
  ):
    raise ValueError(f"completed training violates the frozen scientific protocol: {dataset}")
  return {
    "completed_epochs": record["completed_epochs"],
    "global_step": record["global_step"],
    "training_count_samples": dict(sorted(counts.items())),
    "train_subject_count": len(expected_subjects),
    "box_fallback_samples": int(record["box_fallback_samples"]),
    "trainable_parameter_count": sum(
      int(item["numel"])
      for name, item in record["parameter_inventory"].items()
      if name in trainable_names
    ),
  }


def audit(root: Path, config_path: Path, output_path: Path) -> dict:
  cfg = load_yaml_strict(config_path)
  split_rows = load_configured_subject_splits(root, cfg)
  datasets = {}
  for dataset in ("brainweb", "hecktor"):
    checkpoint = root / cfg["medsam2"]["checkpoint_paths"][dataset]
    verified = verify_completed_training(checkpoint, root, config_path, cfg, dataset)
    provenance_path = checkpoint.with_suffix(".json")
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    expected_subjects = {
      row["subject_id"] for row in split_rows
      if row["dataset"] == dataset and row["split"] == "train"
    }
    datasets[dataset] = {
      "checkpoint_path": checkpoint.relative_to(root).as_posix(),
      "checkpoint_sha256": verified["sha256"],
      "verified_existing_group": bool(verified["reused"]),
      "provenance_path": provenance_path.relative_to(root).as_posix(),
      "provenance_sha256": file_sha256(provenance_path),
      **_validate_scientific_record(provenance, cfg, dataset, expected_subjects),
    }
  result = {
    "schema_version": 1,
    "artifact_kind": "decoder_training_handoff_audit",
    "status": "complete",
    "run_id": cfg["run_id"],
    "config_path": config_path.relative_to(root).as_posix(),
    "config_sha256": file_sha256(config_path),
    "datasets": datasets,
  }
  atomic_write_json(output_path, result)
  return result


def main() -> int:
  parser = argparse.ArgumentParser()
  parser.add_argument("--root", type=Path, default=Path("."))
  parser.add_argument("--config", type=Path, required=True)
  parser.add_argument("--output", type=Path, required=True)
  args = parser.parse_args()
  root = args.root.resolve()
  result = audit(root, args.config.resolve(), args.output.resolve())
  print(json.dumps(result, indent=2, sort_keys=True))
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
