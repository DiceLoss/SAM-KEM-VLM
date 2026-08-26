import csv
from pathlib import Path

import yaml

from sam_kem.candidate_config import generate_candidate_configuration
from sam_kem.splits import load_configured_subject_splits


ROOT = Path(__file__).resolve().parents[1]


def test_candidate_config_is_normalization_and_output_namespaced(tmp_path):
  candidates = tmp_path / "candidates.csv"
  with candidates.open("w", encoding="utf-8", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=(
      "dataset", "candidate", "available", "offset", "scale", "fitted_from",
    ))
    writer.writeheader()
    writer.writerows([
      {"dataset": "brainweb", "candidate": "train_max_fixed_C", "available": "true", "offset": 0, "scale": 6, "fitted_from": "train_only_global_max"},
      {"dataset": "hecktor", "candidate": "train_max_fixed_C", "available": "true", "offset": 0, "scale": 35, "fitted_from": "train_only_global_max"},
    ])
  output = tmp_path / "config.yaml"
  protocols = tmp_path / "protocols"
  generate_candidate_configuration(
    ROOT, ROOT / "code/configs/experiment.yaml", candidates,
    candidate="train_max_fixed_C", run_id="dev_trainmax_v1",
    output_path=output, protocol_output_dir=protocols,
  )
  cfg = yaml.safe_load(output.read_text(encoding="utf-8"))
  assert cfg["normalization"]["scale_constants"] == {"brainweb": 6.0, "hecktor": 35.0}
  assert cfg["data"]["split_file"] == "inputs/clean_v1/subject_split.csv"
  assert cfg["data"]["split_id"] == "clean_v1_development_train_validation"
  assert len(load_configured_subject_splits(ROOT, cfg)) == 244
  assert cfg["data"]["canonical_activity_root"] == "arrays/dev_trainmax_v1"
  assert cfg["medsam2"]["checkpoint_paths"]["hecktor"].startswith(
    "checkpoints/dev_trainmax_v1/"
  )
  assert (protocols / "training.yaml").is_file()
