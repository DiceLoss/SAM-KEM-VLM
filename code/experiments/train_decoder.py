"""Train one dataset-specific decoder using all four count levels."""
import argparse
from pathlib import Path
from sam_kem.medsam_training import train_dataset_decoder
from sam_kem.config import load_yaml_strict

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--dataset", choices=("brainweb", "hecktor"), required=True)
  parser.add_argument("--pretrained", type=Path, required=True)
  parser.add_argument("--output", type=Path, required=True)
  parser.add_argument("--manifest", type=Path)
  parser.add_argument("--epochs", type=int, required=True)
  parser.add_argument("--steps-per-epoch", type=int, required=True)
  parser.add_argument("--learning-rate", type=float, default=1e-4)
  parser.add_argument("--resume", type=Path)
  args = parser.parse_args()
  root = Path(__file__).resolve().parents[2]
  config_path = root / "code/configs/experiment.yaml"
  cfg = load_yaml_strict(config_path)
  manifest = root / (
    args.manifest if args.manifest is not None
    else Path(str(cfg["outputs"]["pretraining_manifest"]))
  )
  print(train_dataset_decoder(
    root, config_path, dataset=args.dataset,
    pretrained_checkpoint=args.pretrained, output_checkpoint=args.output,
    manifest_path=manifest,
    epochs=args.epochs, steps_per_epoch=args.steps_per_epoch,
    learning_rate=args.learning_rate,
    resume_checkpoint=args.resume,
  ))
