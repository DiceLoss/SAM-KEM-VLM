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
from .qwen import request_box
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
      kernel = rbf_kernel(feature, sigma=args.sigma, radius=args.radius)
    outputs = kem(measured, operator, kernel, iterations=args.iterations, save_iterations=saves, multiplicative=gain)
  args.output.mkdir(parents=True, exist_ok=True)
  for iteration, image in outputs.items():
    np.save(args.output / f"{args.method}_iter{iteration}.npy", image)
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
  import hashlib
  image_metadata_path = args.image.with_suffix(".json")
  if not image_metadata_path.is_file():
    raise FileNotFoundError(f"prompt-render provenance missing: {image_metadata_path}")
  image_metadata = json.loads(image_metadata_path.read_text(encoding="utf-8"))
  actual_image_sha = hashlib.sha256(args.image.read_bytes()).hexdigest()
  if image_metadata.get("png_sha256") != actual_image_sha:
    raise ValueError("prompt PNG differs from its rendering sidecar")
  try:
    response = request_box(args.image, prompt=args.prompt, model=args.model)
  except Exception as error:
    record = {
      "dataset": args.dataset, "subject_id": args.subject, "slice_id": args.slice,
      "count_level": args.count, "source": "qwen_request_failure",
      "request_id": "", "image_sha256": hashlib.sha256(args.image.read_bytes()).hexdigest(),
      "source_mlem10_sha256": image_metadata["source_mlem10_sha256"],
      "prompt_sha256": "", "model": args.model, "latency_seconds": "", "box": None,
      "error": repr(error),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"record": record}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(record, ensure_ascii=False))
    return 1
  if response.box is not None:
    response.box.validate(width=128, height=128)
  record = {
    "dataset": args.dataset,
    "subject_id": args.subject,
    "slice_id": args.slice,
    "count_level": args.count,
    "source": "qwen3-vl-plus" if response.box is not None else "qwen_no_detection",
    "request_id": response.request_id,
    "image_sha256": response.image_sha256,
    "source_mlem10_sha256": image_metadata["source_mlem10_sha256"],
    "prompt_sha256": response.prompt_sha256,
    "model": response.model,
    "latency_seconds": response.latency_seconds,
    "box": None if response.box is None else response.box.__dict__,
  }
  args.output.parent.mkdir(parents=True, exist_ok=True)
  args.output.write_text(json.dumps({"record": record, "raw_response": response.raw_response}, ensure_ascii=False, indent=2), encoding="utf-8")
  print(json.dumps(record, ensure_ascii=False))
  return 0


def _run_protocol(args: argparse.Namespace) -> int:
  from .experiments import run_protocol
  output = run_protocol(args.root.resolve(), args.config.resolve(), args.protocol.resolve())
  print(f"wrote {output}")
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


def _prepare_mlem(args: argparse.Namespace) -> int:
  from .mlem_cache import write_mlem_input
  output = write_mlem_input(
    args.root.resolve(), args.config.resolve(), args.sinogram.resolve(),
    args.output.resolve(), projection_scale=args.projection_scale,
  )
  print(f"wrote {output}\nwrote {output.with_suffix('.json')}")
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

  protocol = sub.add_parser("check-protocol", help="validate one experiment protocol")
  protocol.add_argument("--protocol", type=Path, required=True)
  protocol.set_defaults(func=_check_protocol)

  qwen = sub.add_parser("qwen-box", help="request and archive one anchor box")
  qwen.add_argument("--image", type=Path, required=True)
  qwen.add_argument("--output", type=Path, required=True)
  qwen.add_argument("--dataset", required=True)
  qwen.add_argument("--subject", required=True)
  qwen.add_argument("--slice", required=True)
  qwen.add_argument("--count", type=int, required=True)
  qwen.add_argument("--model", default="qwen3-vl-plus")
  qwen.add_argument("--prompt", default="框选出所有明亮团块，返回包含它们的最小单个外接矩形框。")
  qwen.set_defaults(func=_qwen_box)

  run = sub.add_parser("run-protocol", help="run a locked evaluation protocol from the v2 case manifest")
  run.add_argument("--root", type=Path, default=Path("."))
  run.add_argument("--config", type=Path, required=True)
  run.add_argument("--protocol", type=Path, required=True)
  run.set_defaults(func=_run_protocol)

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
  cache.add_argument("--responses", type=Path, required=True)
  cache.add_argument("--output", type=Path, required=True)
  cache.set_defaults(func=_build_box_cache)

  manifest = sub.add_parser("build-manifest", help="scan canonical arrays and write v2 case manifest")
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
  return parser


def main(argv: list[str] | None = None) -> int:
  args = build_parser().parse_args(argv)
  return int(args.func(args))


if __name__ == "__main__":
  raise SystemExit(main())
