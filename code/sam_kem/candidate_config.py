"""Generate isolated development-run configurations from calibration output."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import yaml

from .artifacts import atomic_write_json, atomic_write_text, file_sha256
from .config import ScientificContract, load_yaml_strict


def _replace_namespace(value, source: str, destination: str):
  if isinstance(value, dict):
    return {key: _replace_namespace(item, source, destination) for key, item in value.items()}
  if isinstance(value, list):
    return [_replace_namespace(item, source, destination) for item in value]
  if isinstance(value, str):
    return value.replace(source, destination)
  return value


def _portable_path(path: Path, root: Path) -> str:
  try:
    return path.resolve().relative_to(root.resolve()).as_posix()
  except ValueError:
    return str(path.resolve())


def generate_candidate_configuration(
  root: Path,
  base_config_path: Path,
  candidates_path: Path,
  *,
  candidate: str,
  run_id: str,
  output_path: Path,
  protocol_output_dir: Path,
) -> Path:
  base = load_yaml_strict(base_config_path)
  if base.get("status") != "development_only_no_locked_test":
    raise ValueError("candidate configs must be derived from the development-only base")
  if not run_id or not all(character.isalnum() or character in "_-" for character in run_id):
    raise ValueError("run_id must contain only letters, digits, '_' or '-'")
  if output_path.exists() or protocol_output_dir.exists():
    raise FileExistsError("candidate configuration/protocol output already exists")
  with candidates_path.open("r", encoding="utf-8", newline="") as handle:
    rows = [row for row in csv.DictReader(handle) if row["candidate"] == candidate]
  decision_path = root / "inputs/clean_v1/normalization_decision.json"
  if not decision_path.is_file():
    raise FileNotFoundError("audited normalization decision is required before training")
  decision = json.loads(decision_path.read_text(encoding="utf-8"))
  selected_candidate = decision.get("selected_candidate")
  if candidate != selected_candidate:
    raise ValueError(
      f"candidate {candidate!r} is not the audited development training candidate "
      f"{selected_candidate!r}; see {decision_path}"
    )
  by_dataset = {row["dataset"]: row for row in rows if row.get("available", "").lower() == "true"}
  if set(by_dataset) != {"brainweb", "hecktor"}:
    raise ValueError(f"candidate {candidate!r} is not available for both datasets")

  source_run = str(base["run_id"])
  cfg = _replace_namespace(base, source_run, run_id)
  # The split and extracted-source manifest are shared immutable development inputs.
  cfg["data"]["split_file"] = base["data"]["split_file"]
  cfg["data"]["split_id"] = base["data"]["split_id"]
  cfg["data"]["source_activity_manifest"] = base["data"]["source_activity_manifest"]
  cfg["medsam2"]["source_checkout_path"] = base["medsam2"]["source_checkout_path"]
  cfg["medsam2"]["pretrained_checkpoint_path"] = base["medsam2"]["pretrained_checkpoint_path"]
  cfg["run_id"] = run_id
  cfg["status"] = "development_candidate_ready"
  cfg["locked"] = False
  cfg["normalization"]["normalization_id"] = candidate
  cfg["normalization"]["development_training_candidate"] = selected_candidate
  cfg["normalization"]["development_decision"] = (
    f"{_portable_path(decision_path, root)}:sha256={file_sha256(decision_path)}"
  )
  cfg["normalization"]["scale_constants"] = {
    dataset: float(by_dataset[dataset]["scale"]) for dataset in ("brainweb", "hecktor")
  }
  cfg["normalization"]["offsets"] = {
    dataset: float(by_dataset[dataset]["offset"] or 0.0)
    for dataset in ("brainweb", "hecktor")
  }
  cfg["normalization"]["scale_constant_sources"] = {
    dataset: (
      f"{_portable_path(candidates_path, root)}:sha256={file_sha256(candidates_path)};"
      f"candidate={candidate};fitted_from={by_dataset[dataset]['fitted_from']}"
    ) for dataset in ("brainweb", "hecktor")
  }
  if candidate == "train_max_fixed_C":
    cfg["normalization"]["rule"] = "divide_by_fixed_dataset_constant"
    cfg["normalization"]["clip_to_expected_target_range"] = False
  elif candidate == "configured_physical_C":
    cfg["normalization"]["rule"] = "divide_by_fixed_dataset_physical_constant"
    cfg["normalization"]["clip_to_expected_target_range"] = False
  elif candidate == "train_global_minmax":
    cfg["normalization"]["rule"] = "training_global_minmax_affine_clip01"
    cfg["normalization"]["clip_to_expected_target_range"] = True
  else:
    raise ValueError(f"candidate {candidate!r} is not eligible for decoder training")
  ScientificContract.from_mapping(cfg)
  atomic_write_text(output_path, yaml.safe_dump(cfg, sort_keys=False))

  protocol_output_dir.mkdir(parents=True)
  generated_protocols = []
  for protocol_path in sorted((root / "code/configs/protocols").glob("*.yaml")):
    if protocol_path.name == "normalization_calibration.yaml":
      continue
    protocol = load_yaml_strict(protocol_path)
    protocol = _replace_namespace(protocol, source_run, run_id)
    generated_path = protocol_output_dir / protocol_path.name
    atomic_write_text(generated_path, yaml.safe_dump(protocol, sort_keys=False))
    generated_protocols.append(
      {"path": _portable_path(generated_path, root), "sha256": file_sha256(generated_path)}
    )
  generator_path = root / "code/sam_kem/candidate_config.py"
  atomic_write_json(protocol_output_dir / "generation_manifest.json", {
    "schema_version": 1,
    "artifact_kind": "development_candidate_configuration_bundle",
    "candidate": candidate,
    "run_id": run_id,
    "base_config": {
      "path": _portable_path(base_config_path, root),
      "sha256": file_sha256(base_config_path),
    },
    "calibration_candidates": {
      "path": _portable_path(candidates_path, root),
      "sha256": file_sha256(candidates_path),
    },
    "normalization_decision": {
      "path": _portable_path(decision_path, root),
      "sha256": file_sha256(decision_path),
    },
    "generator": {
      "path": _portable_path(generator_path, root),
      "sha256": file_sha256(generator_path),
    },
    "configuration": {
      "path": _portable_path(output_path, root),
      "sha256": file_sha256(output_path),
    },
    "protocols": generated_protocols,
    "excluded_upstream_protocols": ["normalization_calibration.yaml"],
  })
  return output_path
