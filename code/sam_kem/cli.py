"""Command-line entry points for the auditable pipeline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .audit import audit_exit_code, audit_repository
from .config import load_contract
from .noise import write_low_count_simulation
from .operator_factory import build_discrete_radon
from .protocols import load_protocol
from .reproducibility import CaseKey


def _validate(args: argparse.Namespace) -> int:
  root = args.root.resolve()
  issues = audit_repository(root, (root / args.config).resolve())
  if not issues:
    print("PASS: no scientific-contract violations found")
  for issue in issues:
    print(f"{issue.severity.upper()} {issue.code}: {issue.message}")
  return audit_exit_code(issues)


def _simulate(args: argparse.Namespace) -> int:
  _, contract = load_contract(args.config)
  key = CaseKey(args.dataset, args.subject, args.slice, args.count, args.realization)
  output = write_low_count_simulation(
    args.projection, args.output,
    expected_true_counts=args.count,
    case=key,
    experiment_seed=contract.experiment_seed,
  )
  print(f"wrote {output}\nwrote {output.with_suffix('.json')}")
  return 0


def _reconstruct(args: argparse.Namespace) -> int:
  from .artifacts import atomic_save_numpy
  from .reconstruction import identity_kernel, kem, mlem, normalize_kernel_feature, rbf_kernel

  root = args.root.resolve()
  cfg, contract = load_contract(args.config)
  operator = build_discrete_radon(
    repo_root=root, size=args.size, angles=contract.angles,
    operator_id=contract.operator_id,
    expected_implementation_sha256=cfg["forward_model"]["implementation_sha256"],
  )
  measured = np.load(args.sinogram)
  gain = np.full_like(measured, args.projection_scale, dtype=np.float64)
  saves = tuple(args.save_iterations or [args.iterations])
  if args.method == "mlem":
    outputs = mlem(measured, operator, iterations=args.iterations, save_iterations=saves, multiplicative=gain)
  else:
    if args.feature is None:
      kernel = identity_kernel(operator.image_shape)
    else:
      feature = normalize_kernel_feature(np.load(args.feature))
      support = operator.adjoint(
        np.ones(operator.projection_shape, dtype=np.float64)
      ) > 0
      kernel = rbf_kernel(
        feature, sigma=args.sigma, radius=args.radius, support=support,
      )
    outputs = kem(measured, operator, kernel, iterations=args.iterations, save_iterations=saves, multiplicative=gain)
  args.output.mkdir(parents=True, exist_ok=True)
  for iteration, image in outputs.items():
    atomic_save_numpy(args.output / f"{args.method}_iter{iteration}.npy", image)
  print(f"wrote {len(outputs)} reconstruction array(s) to {args.output}")
  return 0


def _check_protocol(args: argparse.Namespace) -> int:
  protocol = load_protocol(args.protocol)
  print(json.dumps({
    "name": protocol.name,
    "kind": protocol.kind,
    "split": protocol.split,
    "methods": protocol.methods,
    "output": str(protocol.output),
  }, ensure_ascii=False, indent=2))
  return 0


def _qwen_box(args: argparse.Namespace) -> int:
  from .config import load_yaml_strict
  from .qwen import request_and_archive_box
  cfg = load_yaml_strict(args.config.resolve())
  model = str(args.model or cfg["prompts"]["model"])
  prompt = str(args.prompt or cfg["prompts"]["text"])
  if model != cfg["prompts"]["model"] or prompt != cfg["prompts"]["text"]:
    raise ValueError("Qwen model/prompt must exactly match the configured experiment")
  record, success = request_and_archive_box(
    args.image, args.output, cfg=cfg, dataset=args.dataset,
    subject_id=args.subject, slice_id=args.slice, count_level=args.count,
  )
  print(json.dumps(record, ensure_ascii=False))
  return 0 if success else 1


def _run_protocol(args: argparse.Namespace) -> int:
  from .experiments import run_protocol
  output = run_protocol(args.root.resolve(), args.config.resolve(), args.protocol.resolve())
  print(f"wrote {output}")
  return 0


def _derive_primary(args: argparse.Namespace) -> int:
  from .primary import derive_primary
  print(f"wrote {derive_primary(args.root.resolve(), args.config.resolve(), args.protocol.resolve())}")
  return 0


def _run_sweep(args: argparse.Namespace) -> int:
  from .sweep import run_validation_sweep
  raw, frozen = run_validation_sweep(args.root.resolve(), args.config.resolve(), args.protocol.resolve())
  print(f"wrote {raw}\nwrote {frozen}")
  return 0


def _run_ablation(args: argparse.Namespace) -> int:
  from .ablation import run_ablation
  print(f"wrote {run_ablation(args.root.resolve(), args.config.resolve(), args.protocol.resolve())}")
  return 0


def _run_convergence(args: argparse.Namespace) -> int:
  from .convergence import run_convergence
  print(f"wrote {run_convergence(args.root.resolve(), args.config.resolve(), args.protocol.resolve())}")
  return 0


def _aggregate(args: argparse.Namespace) -> int:
  from .aggregate import aggregate_cross_count
  print(f"wrote {aggregate_cross_count(args.per_case, args.output)}")
  return 0


def _build_box_cache(args: argparse.Namespace) -> int:
  from .cache_builder import build_box_cache
  print(f"wrote {build_box_cache(args.manifest, args.responses, args.output, args.config)}")
  return 0


def _prepare_prompt_batch(args: argparse.Namespace) -> int:
  from .prompt_batch import prepare_prompt_manifest
  output = prepare_prompt_manifest(
    args.root.resolve(), args.config.resolve(), args.manifest.resolve(),
    args.output.resolve(), splits=set(args.splits),
    resume_verified=args.resume_verified,
  )
  print(f"wrote {output}")
  return 0


def _request_prompt_batch(args: argparse.Namespace) -> int:
  from .prompt_batch import request_prompt_manifest
  summary = request_prompt_manifest(
    args.root.resolve(), args.config.resolve(), args.manifest.resolve(),
    args.responses.resolve(), resume_verified=args.resume_verified,
  )
  print(json.dumps(summary, indent=2, sort_keys=True))
  return 0


def _build_manifest(args: argparse.Namespace) -> int:
  from .manifest import build_case_manifest
  print(f"wrote {build_case_manifest(args.root.resolve(), args.config.resolve(), args.arrays.resolve(), args.output.resolve())}")
  return 0


def _render_prompt(args: argparse.Namespace) -> int:
  from .display import render_prompt_png
  print(f"wrote {render_prompt_png(args.input, args.output)}")
  return 0


def _project_activity(args: argparse.Namespace) -> int:
  from .pet_simulation import prepare_noise_free_projection
  target, projection = prepare_noise_free_projection(
    args.root.resolve(), args.config.resolve(), dataset=args.dataset,
    raw_activity_path=args.activity.resolve(),
    normalized_target_path=args.target_output.resolve(),
    projection_path=args.projection_output.resolve(),
  )
  print(f"wrote {target}\nwrote {projection}\nwrote {projection.with_suffix('.json')}")
  return 0


def _ingest_activity(args: argparse.Namespace) -> int:
  from .config import load_yaml_strict, require_test_release
  from .ingestion import ingest_manifest
  from .splits import load_configured_subject_splits
  root = args.root.resolve()
  cfg = load_yaml_strict(args.config.resolve())
  load_configured_subject_splits(root, cfg)
  if "test_locked" in args.splits:
    require_test_release(root, cfg)
  outputs = ingest_manifest(
    root, args.manifest.resolve(), root / str(cfg["data"]["split_file"]),
    args.output.resolve(),
    splits=set(args.splits), outside_mass_tolerance=args.outside_mass_tolerance,
  )
  print(f"wrote {len(outputs)} canonical activity slice(s) to {args.output}")
  return 0


def _prepare_mlem(args: argparse.Namespace) -> int:
  from .mlem_cache import write_mlem_input
  output = write_mlem_input(
    args.root.resolve(), args.config.resolve(), args.sinogram.resolve(),
    args.output.resolve(), projection_scale=args.projection_scale,
  )
  print(f"wrote {output}\nwrote {output.with_suffix('.json')}")
  return 0


def _prepare_cases(args: argparse.Namespace) -> int:
  from .case_preparation import prepare_cases
  summary = prepare_cases(
    args.root.resolve(), args.config.resolve(), args.arrays.resolve(),
    splits=set(args.splits), resume_verified=args.resume_verified,
  )
  print(json.dumps(summary, indent=2, sort_keys=True))
  return 0


def _run_normalization_calibration(args: argparse.Namespace) -> int:
  from .normalization import run_normalization_calibration
  candidates, diagnostics = run_normalization_calibration(
    args.root.resolve(), args.config.resolve(), args.protocol.resolve(),
  )
  print(f"wrote {candidates}\nwrote {diagnostics}")
  return 0


def _array_sha(args: argparse.Namespace) -> int:
  from .noise import array_sha256
  value = np.load(args.input)
  print(array_sha256(value))
  return 0


def _create_split(args: argparse.Namespace) -> int:
  from .split_creation import create_subject_split
  output = create_subject_split(
    args.root.resolve(), args.config.resolve(), args.inventory.resolve(), args.output.resolve(),
  )
  print(f"wrote {output}\nset data.split_sha256 to the output of: sam-kem file-sha --input {output}")
  return 0


def _file_sha(args: argparse.Namespace) -> int:
  from .artifacts import file_sha256
  print(file_sha256(args.input))
  return 0


def build_parser() -> argparse.ArgumentParser:
  parser = argparse.ArgumentParser(prog="sam-kem")
  sub = parser.add_subparsers(required=True)
  validate = sub.add_parser("validate", help="audit configuration, splits, cache, sweeps and results")
  validate.add_argument("--root", type=Path, default=Path("."))
  validate.add_argument("--config", type=Path, default=Path("code/configs/experiment.yaml"))
  validate.set_defaults(func=_validate)

  simulate = sub.add_parser("simulate", help="create one independent low-count sinogram")
  simulate.add_argument("--config", type=Path, required=True)
  simulate.add_argument("--projection", type=Path, required=True)
  simulate.add_argument("--output", type=Path, required=True)
  simulate.add_argument("--dataset", required=True)
  simulate.add_argument("--subject", required=True)
  simulate.add_argument("--slice", required=True)
  simulate.add_argument("--count", type=int, required=True)
  simulate.add_argument("--realization", type=int, required=True)
  simulate.set_defaults(func=_simulate)

  reconstruct = sub.add_parser("reconstruct", help="run reference MLEM or coefficient-space KEM")
  reconstruct.add_argument("--root", type=Path, default=Path("."))
  reconstruct.add_argument("--config", type=Path, required=True)
  reconstruct.add_argument("--sinogram", type=Path, required=True)
  reconstruct.add_argument("--output", type=Path, required=True)
  reconstruct.add_argument("--method", choices=("mlem", "kem"), required=True)
  reconstruct.add_argument("--feature", type=Path)
  reconstruct.add_argument("--size", type=int, default=128)
  reconstruct.add_argument("--iterations", type=int, required=True)
  reconstruct.add_argument("--projection-scale", type=float, required=True)
  reconstruct.add_argument("--save-iterations", type=int, nargs="*")
  reconstruct.add_argument("--sigma", type=float, default=0.05)
  reconstruct.add_argument("--radius", type=int, default=2)
  reconstruct.set_defaults(func=_reconstruct)

  prepare_mlem = sub.add_parser("prepare-mlem", help="write a provenance-bound MLEM-10 input")
  prepare_mlem.add_argument("--root", type=Path, default=Path("."))
  prepare_mlem.add_argument("--config", type=Path, required=True)
  prepare_mlem.add_argument("--sinogram", type=Path, required=True)
  prepare_mlem.add_argument("--output", type=Path, required=True)
  prepare_mlem.add_argument("--projection-scale", type=float, required=True)
  prepare_mlem.set_defaults(func=_prepare_mlem)

  prepare_cases = sub.add_parser(
    "prepare-cases", help="batch project, sample, and create MLEM-10 from ingested activity",
  )
  prepare_cases.add_argument("--root", type=Path, default=Path("."))
  prepare_cases.add_argument("--config", type=Path, required=True)
  prepare_cases.add_argument("--arrays", type=Path, required=True)
  prepare_cases.add_argument(
    "--splits", nargs="+", choices=("train", "validation", "test_locked"),
    default=("train", "validation"),
  )
  prepare_cases.add_argument("--resume-verified", action="store_true")
  prepare_cases.set_defaults(func=_prepare_cases)

  protocol = sub.add_parser("check-protocol", help="validate one experiment protocol")
  protocol.add_argument("--protocol", type=Path, required=True)
  protocol.set_defaults(func=_check_protocol)

  qwen = sub.add_parser("qwen-box", help="request and archive one anchor box")
  qwen.add_argument("--image", type=Path, required=True)
  qwen.add_argument("--config", type=Path, required=True)
  qwen.add_argument("--output", type=Path, required=True)
  qwen.add_argument("--dataset", required=True)
  qwen.add_argument("--subject", required=True)
  qwen.add_argument("--slice", required=True)
  qwen.add_argument("--count", type=int, required=True)
  qwen.add_argument("--model")
  qwen.add_argument("--prompt")
  qwen.set_defaults(func=_qwen_box)

  run = sub.add_parser("run-protocol", help="run a locked evaluation protocol from the canonical case manifest")
  run.add_argument("--root", type=Path, default=Path("."))
  run.add_argument("--config", type=Path, required=True)
  run.add_argument("--protocol", type=Path, required=True)
  run.set_defaults(func=_run_protocol)

  primary = sub.add_parser(
    "derive-primary", help="derive the headline rows from the immutable comparison output",
  )
  primary.add_argument("--root", type=Path, default=Path("."))
  primary.add_argument("--config", type=Path, required=True)
  primary.add_argument("--protocol", type=Path, required=True)
  primary.set_defaults(func=_derive_primary)

  sweep = sub.add_parser("run-sweep", help="run full validation-only SAM-KEM sweep")
  sweep.add_argument("--root", type=Path, default=Path("."))
  sweep.add_argument("--config", type=Path, required=True)
  sweep.add_argument("--protocol", type=Path, required=True)
  sweep.set_defaults(func=_run_sweep)

  ablation = sub.add_parser("run-ablation", help="run actual one-factor-at-a-time ablations")
  ablation.add_argument("--root", type=Path, default=Path("."))
  ablation.add_argument("--config", type=Path, required=True)
  ablation.add_argument("--protocol", type=Path, required=True)
  ablation.set_defaults(func=_run_ablation)

  convergence = sub.add_parser("run-convergence", help="run descriptive convergence curves")
  convergence.add_argument("--root", type=Path, default=Path("."))
  convergence.add_argument("--config", type=Path, required=True)
  convergence.add_argument("--protocol", type=Path, required=True)
  convergence.set_defaults(func=_run_convergence)

  aggregate = sub.add_parser("aggregate-cross-count", help="aggregate unique per-case results once")
  aggregate.add_argument("--per-case", type=Path, required=True)
  aggregate.add_argument("--output", type=Path, required=True)
  aggregate.set_defaults(func=_aggregate)

  cache = sub.add_parser("build-box-cache", help="build one box per dataset/subject/slice/count")
  cache.add_argument("--config", type=Path, required=True)
  cache.add_argument("--manifest", type=Path, required=True)
  cache.add_argument("--responses", type=Path, nargs="+", required=True)
  cache.add_argument("--output", type=Path, required=True)
  cache.set_defaults(func=_build_box_cache)

  prompt_batch = sub.add_parser(
    "prepare-prompt-images", help="render one r=0 prompt PNG per canonical box key",
  )
  prompt_batch.add_argument("--root", type=Path, default=Path("."))
  prompt_batch.add_argument("--config", type=Path, required=True)
  prompt_batch.add_argument("--manifest", type=Path, required=True)
  prompt_batch.add_argument("--output", type=Path, required=True)
  prompt_batch.add_argument(
    "--splits", nargs="+", choices=("train", "validation", "test_locked"),
    required=True,
  )
  prompt_batch.add_argument("--resume-verified", action="store_true")
  prompt_batch.set_defaults(func=_prepare_prompt_batch)

  request_batch = sub.add_parser(
    "request-boxes", help="request and archive every prompt-manifest Qwen attempt once",
  )
  request_batch.add_argument("--root", type=Path, default=Path("."))
  request_batch.add_argument("--config", type=Path, required=True)
  request_batch.add_argument("--manifest", type=Path, required=True)
  request_batch.add_argument("--responses", type=Path, required=True)
  request_batch.add_argument("--resume-verified", action="store_true")
  request_batch.set_defaults(func=_request_prompt_batch)

  manifest = sub.add_parser("build-manifest", help="scan canonical arrays and write a verified case manifest")
  manifest.add_argument("--root", type=Path, default=Path("."))
  manifest.add_argument("--config", type=Path, required=True)
  manifest.add_argument("--arrays", type=Path, required=True)
  manifest.add_argument("--output", type=Path, required=True)
  manifest.set_defaults(func=_build_manifest)

  render = sub.add_parser("render-prompt", help="render an exact 128x128 VLM PNG")
  render.add_argument("--input", type=Path, required=True)
  render.add_argument("--output", type=Path, required=True)
  render.set_defaults(func=_render_prompt)

  project = sub.add_parser("project-activity", help="divide by fixed C and apply the canonical forward operator")
  project.add_argument("--root", type=Path, default=Path("."))
  project.add_argument("--config", type=Path, required=True)
  project.add_argument("--dataset", choices=("brainweb", "hecktor"), required=True)
  project.add_argument("--activity", type=Path, required=True)
  project.add_argument("--target-output", type=Path, required=True)
  project.add_argument("--projection-output", type=Path, required=True)
  project.set_defaults(func=_project_activity)

  ingest = sub.add_parser(
    "ingest-activity", help="ingest source PET slices through an explicit spatial transform",
  )
  ingest.add_argument("--root", type=Path, default=Path("."))
  ingest.add_argument("--config", type=Path, default=Path("code/configs/experiment.yaml"))
  ingest.add_argument("--manifest", type=Path, required=True)
  ingest.add_argument("--output", type=Path, required=True)
  ingest.add_argument(
    "--splits", nargs="+", choices=("train", "validation", "test_locked"),
    default=("train", "validation"),
  )
  ingest.add_argument("--outside-mass-tolerance", type=float, default=1.0e-6)
  ingest.set_defaults(func=_ingest_activity)

  normalization = sub.add_parser(
    "calibrate-normalization",
    help="fit scale candidates on train data and diagnose them on validation",
  )
  normalization.add_argument("--root", type=Path, default=Path("."))
  normalization.add_argument("--config", type=Path, required=True)
  normalization.add_argument("--protocol", type=Path, required=True)
  normalization.set_defaults(func=_run_normalization_calibration)

  array_sha = sub.add_parser("array-sha", help="print the canonical dtype/shape/content SHA of a NumPy array")
  array_sha.add_argument("--input", type=Path, required=True)
  array_sha.set_defaults(func=_array_sha)
  file_sha = sub.add_parser("file-sha", help="print the SHA-256 of a file")
  file_sha.add_argument("--input", type=Path, required=True)
  file_sha.set_defaults(func=_file_sha)
  split = sub.add_parser("create-split", help="create a deterministic subject-level clean split")
  split.add_argument("--root", type=Path, default=Path("."))
  split.add_argument("--config", type=Path, required=True)
  split.add_argument("--inventory", type=Path, required=True)
  split.add_argument("--output", type=Path, required=True)
  split.set_defaults(func=_create_split)
  return parser


def main(argv: list[str] | None = None) -> int:
  args = build_parser().parse_args(argv)
  return int(args.func(args))


if __name__ == "__main__":
  raise SystemExit(main())
