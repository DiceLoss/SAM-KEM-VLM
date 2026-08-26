"""Reconstruction runtime benchmark with explicit scope and raw repetitions."""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from .config import load_contract, require_locked_execution
from .experiments import compute_mlem_input, frozen_for_method, load_case_manifest, load_frozen_hyperparameters, method_image, validate_case_contract
from .operator_factory import build_discrete_radon
from .protocols import load_protocol
from .runmeta import write_run_metadata
from .artifacts import atomic_csv_writer, finalize_csv, write_success_marker


def _synchronize() -> None:
  try:
    import torch
    if torch.cuda.is_available(): torch.cuda.synchronize()
  except ImportError:
    pass


def run_runtime(root: Path, config_path: Path, protocol_path: Path) -> Path:
  cfg, contract = load_contract(config_path); protocol = load_protocol(protocol_path)
  require_locked_execution(cfg)
  if protocol.kind != "runtime": raise ValueError("runtime runner requires runtime protocol")
  cases = load_case_manifest(root, root / str(cfg["outputs"]["raw_array_manifest"]), split=protocol.split)
  validate_case_contract(cases, contract)
  selected: dict[str, list] = {}
  for dataset in protocol.raw["datasets"]:
    dataset_cases = sorted(
      (
        case for case in cases
        if case.dataset == dataset
        and case.count_level == int(protocol.raw["count_level"])
        and case.realization_id == int(protocol.raw["realization"])
      ),
      key=lambda case: (case.subject_id, case.slice_id),
    )
    one_per_subject = []
    seen_subjects = set()
    for case in dataset_cases:
      if case.subject_id not in seen_subjects:
        one_per_subject.append(case); seen_subjects.add(case.subject_id)
    selected[dataset] = one_per_subject[:int(protocol.raw["cases_per_dataset"])]
  if any(len(values) != int(protocol.raw["cases_per_dataset"]) for values in selected.values()):
    raise ValueError("runtime protocol lacks enough distinct validation subjects")
  frozen = load_frozen_hyperparameters(
    root / str(cfg["outputs"]["frozen_hyperparameters"]),
    expected_run_id=contract.run_id,
  )
  operator = build_discrete_radon(
    repo_root=root, size=int(cfg["forward_model"]["image_size"]), angles=contract.angles,
    operator_id=contract.operator_id,
    expected_implementation_sha256=cfg["forward_model"]["implementation_sha256"],
  )
  output_dir = root / protocol.output; output_dir.mkdir(parents=True, exist_ok=True)
  run_metadata = write_run_metadata(output_dir, root=root, config_path=config_path, protocol_path=protocol_path)
  output = output_dir / "runtime_repetitions.csv"
  fields = [
    "run_id", "dataset", "subject_id", "slice_id", "realization_id", "count_level",
    "method", "repetition", "seconds", "scope",
  ]
  with atomic_csv_writer(output, fields) as writer:
    for dataset, dataset_cases in selected.items():
      for case in dataset_cases:
        measured = np.load(case.sinogram_path)
        mlem_image = compute_mlem_input(case, measured, operator, cfg)
        for method in protocol.methods:
          if method == "MR-KEM" and dataset != "brainweb" or method == "CT-KEM" and dataset != "hecktor": continue
          hp = frozen_for_method(frozen, dataset, case.count_level, method, cfg)
          for repetition in range(-int(protocol.raw["warmup_repetitions"]), int(protocol.raw["timed_repetitions"])):
            _synchronize(); start = time.perf_counter()
            method_image(method, case, measured, operator, cfg, hp, mlem_image=mlem_image)
            _synchronize(); elapsed = time.perf_counter() - start
            if repetition >= 0:
              writer.writerow({
                "run_id": contract.run_id, "dataset": dataset,
                "subject_id": case.subject_id,
                "slice_id": case.slice_id, "realization_id": case.realization_id,
                "count_level": case.count_level, "method": method,
                "repetition": repetition, "seconds": elapsed,
                "scope": protocol.raw["scope"],
              })
  result_manifest = finalize_csv(
    output, schema_name="runtime_repetitions_v2",
    primary_key=[
      "run_id", "dataset", "subject_id", "slice_id", "realization_id", "count_level",
      "method", "repetition",
    ],
    split=protocol.split,
  )
  write_success_marker(output_dir, [
    run_metadata, output_dir / "environment_freeze.txt", output, result_manifest,
  ])
  return output
