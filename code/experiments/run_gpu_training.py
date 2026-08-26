"""Run the preflight and both dataset-specific decoder trainings sequentially."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
from pathlib import Path

from gpu_preflight import preflight
from sam_kem.config import load_yaml_strict
from sam_kem.medsam import sha256_file
from sam_kem.medsam_training import train_dataset_decoder


def _verify_completed_training(output: Path, config_path: Path, cfg: dict, dataset: str) -> dict:
  """Accept an existing final checkpoint only with its complete exact provenance group."""
  provenance_path = output.with_suffix(".json")
  metrics_path = output.with_suffix(".training.csv")
  metrics_manifest = metrics_path.with_suffix(metrics_path.suffix + ".manifest.json")
  resume_path = output.with_suffix(".resume.pt")
  required = (output, provenance_path, metrics_path, metrics_manifest, resume_path)
  missing = [str(path) for path in required if not path.is_file()]
  if missing:
    raise RuntimeError(f"partial completed-training group for {dataset}: {missing}")
  record = json.loads(provenance_path.read_text(encoding="utf-8"))
  config_sha = hashlib.sha256(config_path.read_bytes()).hexdigest()
  expected = {
    "schema_version": 1,
    "artifact_kind": "checkpoint_provenance",
    "run_id": cfg["run_id"],
    "dataset": dataset,
    "normalization_id": cfg["normalization"]["normalization_id"],
    "config_sha256": config_sha,
    "medsam2_source_commit": cfg["medsam2"]["source_commit"],
    "completed_epochs": int(cfg["medsam2"]["training"]["epochs"]),
  }
  mismatches = {
    key: (record.get(key), value) for key, value in expected.items()
    if record.get(key) != value
  }
  hashes_match = (
    record.get("checkpoint_sha256") == sha256_file(output)
    and record.get("resume_checkpoint_sha256") == sha256_file(resume_path)
    and record.get("training_metrics_sha256") == sha256_file(metrics_path)
    and record.get("training_metrics_manifest_sha256") == sha256_file(metrics_manifest)
  )
  if mismatches or not hashes_match:
    raise RuntimeError(
      f"existing completed training failed provenance verification for {dataset}: "
      f"mismatches={mismatches}, hashes_match={hashes_match}"
    )
  return {"path": str(output), "sha256": sha256_file(output), "reused": True}


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--config", type=Path, required=True)
  args = parser.parse_args()
  root = Path(__file__).resolve().parents[2]
  config_path = args.config.resolve()
  cfg = load_yaml_strict(config_path)
  print(json.dumps(preflight(root, config_path), indent=2))
  gc.collect()
  import torch
  torch.cuda.empty_cache()
  training = cfg["medsam2"]["training"]
  manifest = root / cfg["outputs"]["pretraining_manifest"]
  pretrained = root / cfg["medsam2"]["pretrained_checkpoint_path"]
  completed = {}
  for dataset in ("brainweb", "hecktor"):
    output = root / cfg["medsam2"]["checkpoint_paths"][dataset]
    if output.is_file():
      completed[dataset] = _verify_completed_training(output, config_path, cfg, dataset)
      continue
    resume = output.with_suffix(".resume.pt")
    train_dataset_decoder(
      root, config_path, dataset=dataset, pretrained_checkpoint=pretrained,
      output_checkpoint=output, manifest_path=manifest,
      epochs=int(training["epochs"]), steps_per_epoch=int(training["steps_per_epoch"]),
      learning_rate=float(training["learning_rate"]),
      resume_checkpoint=resume if resume.is_file() else None,
    )
    completed[dataset] = {"path": str(output), "sha256": sha256_file(output), "reused": False}
    gc.collect()
    torch.cuda.empty_cache()
  print(json.dumps({"status": "GPU_TRAINING_COMPLETE", "checkpoints": completed}, indent=2))


if __name__ == "__main__":
  main()
