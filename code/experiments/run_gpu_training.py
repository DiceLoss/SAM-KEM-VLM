"""Run the preflight and both dataset-specific decoder trainings sequentially."""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

from gpu_preflight import preflight
from sam_kem.config import load_yaml_strict
from sam_kem.medsam import sha256_file
from sam_kem.medsam_training import train_dataset_decoder, verify_completed_training


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
      completed[dataset] = verify_completed_training(output, root, config_path, cfg, dataset)
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
