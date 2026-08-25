"""Train one dataset-specific decoder using all four count levels."""
import argparse
from pathlib import Path
from sam_kem.medsam_training import train_dataset_decoder

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--dataset", choices=("brainweb", "hecktor"), required=True)
  parser.add_argument("--pretrained", type=Path, required=True)
  parser.add_argument("--output", type=Path, required=True)
  parser.add_argument("--manifest", type=Path, default=Path("results/metrics/case_manifest_inputs.csv"))
  parser.add_argument("--epochs", type=int, required=True)
  parser.add_argument("--steps-per-epoch", type=int, required=True)
  parser.add_argument("--learning-rate", type=float, default=1e-4)
  parser.add_argument("--resume", type=Path)
  args = parser.parse_args()
  root = Path(__file__).resolve().parents[2]
  print(train_dataset_decoder(
    root, root / "code/configs/experiment.yaml", dataset=args.dataset,
    pretrained_checkpoint=args.pretrained, output_checkpoint=args.output,
    manifest_path=root / args.manifest,
    epochs=args.epochs, steps_per_epoch=args.steps_per_epoch,
    learning_rate=args.learning_rate,
    resume_checkpoint=args.resume,
  ))
