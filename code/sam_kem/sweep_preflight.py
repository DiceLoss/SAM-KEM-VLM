"""Read-only exact-grid runtime preflight for the validation sweep."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor
from collections import Counter
from itertools import product
from pathlib import Path

import numpy as np

from .artifacts import atomic_write_json, file_sha256
from .config import load_contract, require_development_execution
from .experiments import Case, load_case_manifest, validate_case_contract
from .metrics import nrmse, ssim
from .operator_factory import build_discrete_radon
from .prior_registration import validate_registered_validation_prior_configuration
from .protocols import Protocol, load_protocol
from .reconstruction import (
  bowsher_kernel,
  kem,
  mlem,
  normalize_kernel_feature,
  operator_support_mask,
  osem,
  rbf_kernel,
)


def sweep_workload(
  protocol: Protocol, case_counts: dict[str, int], *, prompt_mlem_iterations: int,
  osem_subsets: int,
) -> dict:
  """Return exact output-row and iterative-update counts for a declared grid."""
  search = protocol.raw["search_space"]
  sigmas = len(search["sigma"])
  radii = len(search["radius"])
  iterations = [int(value) for value in search["iterations"]]
  alphas = len(search["alpha"])
  bowsher_settings = len(search["bowsher_neighbours"]) * len(search["bowsher_radius"])
  per_dataset = {}
  for dataset, cases in case_counts.items():
    rows = 0
    updates = int(prompt_mlem_iterations)
    for method in protocol.methods:
      if method == "MR-KEM" and dataset != "brainweb":
        continue
      if method == "CT-KEM" and dataset != "hecktor":
        continue
      if method == "MLEM":
        rows += len(search["mlem_iterations"])
        updates += max(int(value) for value in search["mlem_iterations"])
      elif method == "OSEM":
        rows += len(search["osem_cycles"])
        updates += max(int(value) for value in search["osem_cycles"]) * osem_subsets
      elif method == "Bowsher-KEM":
        rows += bowsher_settings * len(iterations)
        updates += bowsher_settings * max(iterations)
      elif method.startswith("SAM"):
        rows += sigmas * radii * len(iterations) * alphas
        updates += sigmas * radii * max(iterations)
      else:
        rows += sigmas * radii * len(iterations)
        updates += sigmas * radii * max(iterations)
    per_dataset[dataset] = {
      "cases": cases, "result_rows_per_case": rows,
      "operator_updates_per_case": updates,
      "result_rows": cases * rows, "operator_updates": cases * updates,
    }
  return {
    "per_dataset": per_dataset,
    "total_cases": sum(case_counts.values()),
    "total_result_rows": sum(item["result_rows"] for item in per_dataset.values()),
    "total_operator_updates": sum(item["operator_updates"] for item in per_dataset.values()),
  }


def _clean_commit(root: Path) -> str:
  status = subprocess.run(
    ["git", "status", "--porcelain"], cwd=root, check=True,
    capture_output=True, text=True,
  ).stdout.strip()
  if status:
    raise RuntimeError(f"runtime preflight requires a clean worktree: {status.splitlines()}")
  return subprocess.run(
    ["git", "rev-parse", "HEAD"], cwd=root, check=True,
    capture_output=True, text=True,
  ).stdout.strip()


def _commit_allowing_untracked(root: Path, allowed: set[Path]) -> str:
  """Require a clean tree except for explicitly reviewed, untracked evidence."""
  lines = subprocess.run(
    ["git", "status", "--porcelain"], cwd=root, check=True,
    capture_output=True, text=True,
  ).stdout.splitlines()
  allowed_names = {
    path.resolve().relative_to(root.resolve()).as_posix() for path in allowed
  }
  unexpected = []
  for line in lines:
    if not line.startswith("?? ") or line[3:].replace("\\", "/") not in allowed_names:
      unexpected.append(line)
  if unexpected:
    raise RuntimeError(f"parallel preflight found unexpected Git status: {unexpected}")
  return subprocess.run(
    ["git", "rev-parse", "HEAD"], cwd=root, check=True,
    capture_output=True, text=True,
  ).stdout.strip()


def _metric_summary(values: list[tuple[float, float]]) -> dict:
  array = np.asarray(values, dtype=np.float64)
  if array.ndim != 2 or array.shape[1] != 2 or not np.all(np.isfinite(array)):
    raise ValueError("runtime preflight produced non-finite metrics")
  digest = hashlib.sha256(array.astype("<f8", copy=False).tobytes()).hexdigest()
  return {
    "evaluations": len(values), "metric_array_sha256": digest,
    "nrmse_min": float(array[:, 0].min()), "nrmse_max": float(array[:, 0].max()),
    "ssim_min": float(array[:, 1].min()), "ssim_max": float(array[:, 1].max()),
  }


def _score(target: np.ndarray, image: np.ndarray, support: np.ndarray) -> tuple[float, float]:
  if not np.all(np.isfinite(image)):
    raise ValueError("runtime preflight reconstruction is non-finite")
  return nrmse(target, image, support), ssim(target, image, data_range=1.0, mask=support)


def _time_phase(name: str, function, timings: dict[str, float]):
  started = time.perf_counter()
  result = function()
  timings[name] = time.perf_counter() - started
  return result


def _measure_case(case: Case, operator, cfg: dict, protocol: Protocol) -> dict:
  if case.anatomy_path is None or case.sam_prior_path is None:
    raise FileNotFoundError(f"runtime preflight case lacks anatomy or SAM prior: {case}")
  measured = np.load(case.sinogram_path, allow_pickle=False)
  target = np.load(case.target_path, allow_pickle=False)
  anatomy = np.load(case.anatomy_path, allow_pickle=False)
  prior = np.load(case.sam_prior_path, allow_pickle=False)
  if any(array.shape != (128, 128) for array in (target, anatomy, prior)):
    raise ValueError(f"preflight input shape mismatch: {case}")
  search = protocol.raw["search_space"]
  sigmas = [float(value) for value in search["sigma"]]
  radii = [int(value) for value in search["radius"]]
  iterations = [int(value) for value in search["iterations"]]
  alphas = [float(value) for value in search["alpha"]]
  bowsher_neighbours = [int(value) for value in search["bowsher_neighbours"]]
  bowsher_radii = [int(value) for value in search["bowsher_radius"]]
  mlem_iterations = [int(value) for value in search["mlem_iterations"]]
  osem_cycles = [int(value) for value in search["osem_cycles"]]
  support = operator_support_mask(operator)
  gain = np.full_like(measured, case.projection_scale, dtype=np.float64)
  timings: dict[str, float] = {}
  metrics: dict[str, dict] = {}

  prompt = _time_phase(
    "prompt_mlem10_seconds",
    lambda: mlem(
      measured, operator, iterations=int(cfg["reconstruction"]["prompt_mlem_iterations"]),
      multiplicative=gain,
    )[int(cfg["reconstruction"]["prompt_mlem_iterations"])],
    timings,
  )

  def baseline_mlem():
    outputs = mlem(
      measured, operator, iterations=max(mlem_iterations),
      save_iterations=mlem_iterations, multiplicative=gain,
    )
    return [_score(target, outputs[value], support) for value in mlem_iterations]

  metrics["MLEM"] = _metric_summary(_time_phase("MLEM_seconds", baseline_mlem, timings))

  def baseline_osem():
    outputs = osem(
      measured, operator, iterations=max(osem_cycles),
      subsets=int(cfg["reconstruction"]["osem_subsets"]),
      save_iterations=osem_cycles, multiplicative=gain,
    )
    return [_score(target, outputs[value], support) for value in osem_cycles]

  metrics["OSEM"] = _metric_summary(_time_phase("OSEM_seconds", baseline_osem, timings))

  def evaluate_rbf(feature: np.ndarray, *, sam: bool = False):
    values = []
    physical_prior = np.clip(
      prior, cfg["normalization"]["expected_target_range"][0],
      cfg["normalization"]["expected_target_range"][1],
    )
    for sigma, radius in product(sigmas, radii):
      kernel = rbf_kernel(feature, sigma=sigma, radius=radius, support=support)
      outputs = kem(
        measured, operator, kernel, iterations=max(iterations),
        save_iterations=iterations, multiplicative=gain,
      )
      for iteration in iterations:
        if sam:
          for alpha in alphas:
            image = alpha * physical_prior + (1.0 - alpha) * outputs[iteration]
            values.append(_score(target, np.maximum(image, 0.0), support))
        else:
          values.append(_score(target, outputs[iteration], support))
    return values

  metrics["SelfKEM"] = _metric_summary(_time_phase(
    "SelfKEM_seconds",
    lambda: evaluate_rbf(normalize_kernel_feature(prompt)), timings,
  ))
  anatomy_method = "MR-KEM" if case.dataset == "brainweb" else "CT-KEM"
  metrics[anatomy_method] = _metric_summary(_time_phase(
    f"{anatomy_method}_seconds",
    lambda: evaluate_rbf(normalize_kernel_feature(anatomy)), timings,
  ))

  def evaluate_bowsher():
    values = []
    feature = normalize_kernel_feature(anatomy)
    for radius, neighbours in product(bowsher_radii, bowsher_neighbours):
      kernel = bowsher_kernel(
        feature, neighbours=neighbours, radius=radius, support=support,
      )
      outputs = kem(
        measured, operator, kernel, iterations=max(iterations),
        save_iterations=iterations, multiplicative=gain,
      )
      values.extend(_score(target, outputs[value], support) for value in iterations)
    return values

  metrics["Bowsher-KEM"] = _metric_summary(_time_phase(
    "Bowsher-KEM_seconds", evaluate_bowsher, timings,
  ))
  metrics["SAM-KEM-MedSAM2-Qwen3-full"] = _metric_summary(_time_phase(
    "SAM-KEM-MedSAM2-Qwen3-full_seconds",
    lambda: evaluate_rbf(normalize_kernel_feature(prior), sam=True), timings,
  ))
  expected_evaluations = 580
  actual_evaluations = sum(item["evaluations"] for item in metrics.values())
  if actual_evaluations != expected_evaluations:
    raise ValueError(f"preflight grid evaluated {actual_evaluations}, expected 580")
  return {
    "dataset": case.dataset, "subject_id": case.subject_id,
    "slice_id": case.slice_id, "count_level": case.count_level,
    "realization_id": case.realization_id, "timings_seconds": timings,
    "total_seconds": sum(timings.values()), "metrics": metrics,
    "result_evaluations": actual_evaluations,
  }


def run_sweep_runtime_preflight(
  root: Path, config_path: Path, protocol_path: Path, output_path: Path,
) -> dict:
  """Measure the complete declared grid on one fixed validation case per dataset."""
  root = root.resolve()
  config_path = config_path.resolve()
  protocol_path = protocol_path.resolve()
  output_path = output_path.resolve()
  if output_path.exists():
    raise FileExistsError(f"refusing to overwrite runtime preflight: {output_path}")
  commit = _clean_commit(root)
  cfg, contract = load_contract(config_path)
  require_development_execution(cfg)
  registration = validate_registered_validation_prior_configuration(root, config_path, cfg)
  protocol = load_protocol(protocol_path)
  if protocol.kind != "hyperparameter_sweep" or protocol.split != "validation":
    raise ValueError("runtime preflight requires the validation hyperparameter sweep protocol")
  expected_methods = {
    "MLEM", "OSEM", "Bowsher-KEM", "SelfKEM", "MR-KEM", "CT-KEM",
    "SAM-KEM-MedSAM2-Qwen3-full",
  }
  if set(protocol.methods) != expected_methods or len(protocol.methods) != len(expected_methods):
    raise ValueError("runtime preflight requires the complete declared comparison method set")
  manifest_path = root / str(cfg["outputs"]["validation_manifest"])
  cases = load_case_manifest(root, manifest_path, split="validation")
  validate_case_contract(cases, contract)
  case_counts = dict(Counter(case.dataset for case in cases))
  if case_counts != {"brainweb": 8352, "hecktor": 540}:
    raise ValueError(f"unexpected validation case counts: {case_counts}")
  candidates = [
    case for case in cases if case.count_level == 200000 and case.realization_id == 0
  ]
  samples = [
    min((case for case in candidates if case.dataset == dataset),
        key=lambda case: (case.subject_id, case.slice_id))
    for dataset in ("brainweb", "hecktor")
  ]
  operator_started = time.perf_counter()
  operator = build_discrete_radon(
    repo_root=root, size=int(cfg["forward_model"]["image_size"]),
    angles=contract.angles, operator_id=contract.operator_id,
    expected_implementation_sha256=cfg["forward_model"]["implementation_sha256"],
  )
  operator_build_seconds = time.perf_counter() - operator_started
  measurements = [_measure_case(case, operator, cfg, protocol) for case in samples]
  workload = sweep_workload(
    protocol, case_counts,
    prompt_mlem_iterations=int(cfg["reconstruction"]["prompt_mlem_iterations"]),
    osem_subsets=int(cfg["reconstruction"]["osem_subsets"]),
  )
  seconds_by_dataset = {item["dataset"]: item["total_seconds"] for item in measurements}
  projected_seconds = sum(
    seconds_by_dataset[dataset] * count for dataset, count in case_counts.items()
  )
  result = {
    "artifact_kind": "validation_sweep_runtime_preflight", "schema_version": 1,
    "status": "PASS", "write_policy": "no_formal_sweep_outputs",
    "git_commit": commit, "config_path": config_path.relative_to(root).as_posix(),
    "config_sha256": file_sha256(config_path),
    "protocol_path": protocol_path.relative_to(root).as_posix(),
    "protocol_sha256": file_sha256(protocol_path),
    "manifest_path": manifest_path.relative_to(root).as_posix(),
    "manifest_sha256": file_sha256(manifest_path),
    "prior_registration": registration,
    "execution_backend": "scipy_sparse_csr_cpu_float64",
    "gpu_used": False, "operator_build_seconds": operator_build_seconds,
    "platform": platform.platform(), "cpu_count": os.cpu_count(),
    "thread_environment": {
      name: os.environ.get(name) for name in (
        "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
      )
    },
    "selection_rule": "lexicographically_first_validation_subject_slice_at_c200000_r0_per_dataset",
    "measurements": measurements, "declared_workload": workload,
    "projected_serial_seconds_from_two_case_measurement": projected_seconds,
    "projected_serial_hours": projected_seconds / 3600.0,
    "projection_interpretation": (
      "first-order serial estimate only; full-run sharding/resume and backend decision required"
    ),
    "test_locked_access": "prohibited",
  }
  atomic_write_json(output_path, result)
  return result


def _measure_case_process(payload: tuple[Path, Path, Path, Case]) -> dict:
  """Spawn-safe worker: construct the exact operator and run one complete case grid."""
  root, config_path, protocol_path, case = payload
  cfg, contract = load_contract(config_path)
  protocol = load_protocol(protocol_path)
  operator_started = time.perf_counter()
  operator = build_discrete_radon(
    repo_root=root, size=int(cfg["forward_model"]["image_size"]),
    angles=contract.angles, operator_id=contract.operator_id,
    expected_implementation_sha256=cfg["forward_model"]["implementation_sha256"],
  )
  operator_build_seconds = time.perf_counter() - operator_started
  result = _measure_case(case, operator, cfg, protocol)
  result["operator_build_seconds"] = operator_build_seconds
  result["process_id"] = os.getpid()
  return result


def run_parallel_sweep_scaling_preflight(
  root: Path, config_path: Path, protocol_path: Path, runtime_preflight_path: Path,
  output_path: Path, *, workers: int,
) -> dict:
  """Measure full-grid throughput across fixed cases using Windows-safe processes."""
  root = root.resolve()
  config_path = config_path.resolve()
  protocol_path = protocol_path.resolve()
  runtime_preflight_path = runtime_preflight_path.resolve()
  output_path = output_path.resolve()
  if output_path.exists():
    raise FileExistsError(f"refusing to overwrite scaling preflight: {output_path}")
  if workers != 8:
    raise ValueError("the reviewed scaling gate requires exactly 8 workers")
  commit = _commit_allowing_untracked(root, {runtime_preflight_path})
  runtime_record = json.loads(runtime_preflight_path.read_text(encoding="utf-8"))
  if (
    runtime_record.get("status") != "PASS"
    or runtime_record.get("execution_backend") != "scipy_sparse_csr_cpu_float64"
    or runtime_record.get("gpu_used") is not False
    or runtime_record.get("declared_workload", {}).get("total_result_rows") != 5_157_360
    or runtime_record.get("declared_workload", {}).get("total_operator_updates") != 41_347_800
  ):
    raise ValueError("serial runtime preflight is not the reviewed PASS artifact")
  cfg, contract = load_contract(config_path)
  require_development_execution(cfg)
  registration = validate_registered_validation_prior_configuration(root, config_path, cfg)
  protocol = load_protocol(protocol_path)
  expected_methods = {
    "MLEM", "OSEM", "Bowsher-KEM", "SelfKEM", "MR-KEM", "CT-KEM",
    "SAM-KEM-MedSAM2-Qwen3-full",
  }
  if set(protocol.methods) != expected_methods:
    raise ValueError("parallel scaling gate requires the complete method set")
  manifest_path = root / str(cfg["outputs"]["validation_manifest"])
  cases = load_case_manifest(root, manifest_path, split="validation")
  validate_case_contract(cases, contract)
  eligible = [
    case for case in cases if case.count_level == 200000 and case.realization_id == 0
  ]
  samples = []
  for dataset in ("brainweb", "hecktor"):
    dataset_cases = sorted(
      (case for case in eligible if case.dataset == dataset),
      key=lambda case: (case.subject_id, case.slice_id),
    )
    samples.extend(dataset_cases[:4])
  if len(samples) != workers or len({
    (case.dataset, case.subject_id, case.slice_id) for case in samples
  }) != workers:
    raise ValueError("parallel scaling selection did not produce 8 unique slices")
  payloads = [(root, config_path, protocol_path, case) for case in samples]
  started = time.perf_counter()
  with ProcessPoolExecutor(max_workers=workers) as executor:
    measurements = list(executor.map(_measure_case_process, payloads))
  wall_seconds = time.perf_counter() - started
  if any(item["result_evaluations"] != 580 for item in measurements):
    raise ValueError("parallel worker did not complete the exact 580-evaluation grid")
  serial_case_seconds = [
    float(item["total_seconds"]) for item in runtime_record["measurements"]
  ]
  serial_reference_seconds_per_case = float(np.mean(serial_case_seconds))
  serial_equivalent_seconds = serial_reference_seconds_per_case * len(measurements)
  speedup = serial_equivalent_seconds / wall_seconds
  throughput = len(measurements) / wall_seconds
  projected_hours = len(cases) / throughput / 3600.0
  result = {
    "artifact_kind": "validation_sweep_parallel_scaling_preflight",
    "schema_version": 1, "status": "PASS",
    "write_policy": "no_formal_sweep_outputs", "git_commit": commit,
    "config_path": config_path.relative_to(root).as_posix(),
    "config_sha256": file_sha256(config_path),
    "protocol_path": protocol_path.relative_to(root).as_posix(),
    "protocol_sha256": file_sha256(protocol_path),
    "manifest_path": manifest_path.relative_to(root).as_posix(),
    "manifest_sha256": file_sha256(manifest_path),
    "serial_runtime_preflight_path": runtime_preflight_path.relative_to(root).as_posix(),
    "serial_runtime_preflight_sha256": file_sha256(runtime_preflight_path),
    "prior_registration": registration,
    "execution_backend": "scipy_sparse_csr_cpu_float64_multiprocess",
    "gpu_used": False, "worker_processes": workers,
    "sample_count": len(measurements), "result_evaluations_per_sample": 580,
    "selection_rule": "first_four_unique_validation_slices_per_dataset_at_c200000_r0",
    "measurements": measurements, "wall_seconds": wall_seconds,
    "serial_reference_seconds_per_case": serial_reference_seconds_per_case,
    "serial_equivalent_seconds": serial_equivalent_seconds,
    "measured_speedup": speedup, "parallel_efficiency": speedup / workers,
    "measured_cases_per_second": throughput,
    "projected_full_sweep_hours_at_measured_throughput": projected_hours,
    "cpu_count": os.cpu_count(), "platform": platform.platform(),
    "thread_environment": {
      name: os.environ.get(name) for name in (
        "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
      )
    },
    "formal_output_created": False, "frozen_hyperparameters_created": False,
    "test_locked_access": "prohibited",
  }
  atomic_write_json(output_path, result)
  return result
