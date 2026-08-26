"""Black-box audit of the committed PET samples against the locked operator."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from sam_kem.config import load_yaml_strict
from sam_kem.metrics import mae, nrmse
from sam_kem.operator_factory import build_discrete_radon
from sam_kem.reconstruction import mlem


def _sha256(path: Path) -> str:
  return hashlib.sha256(path.read_bytes()).hexdigest()


def _summary(values: list[float]) -> dict[str, float]:
  array = np.asarray(values, dtype=np.float64)
  return {
    "mean": float(array.mean()), "minimum": float(array.min()),
    "maximum": float(array.max()),
  }


def audit(root: Path, *, count: int, reconstruction_cases: int, iterations: int) -> dict:
  cfg = load_yaml_strict(root / "code/configs/experiment.yaml")
  operator = build_discrete_radon(
    repo_root=root, size=int(cfg["forward_model"]["image_size"]),
    angles=int(cfg["forward_model"]["angles"]),
    operator_id=cfg["forward_model"]["operator_id"],
    expected_implementation_sha256=cfg["forward_model"]["implementation_sha256"],
  )
  rng = np.random.default_rng(20260826)
  test_image = rng.random(operator.image_shape)
  test_projection = rng.random(operator.projection_shape)
  forward = operator.forward(test_image)
  left = float(np.vdot(forward, test_projection))
  right = float(np.vdot(test_image, operator.adjoint(test_projection)))
  result = {
    "status": "diagnostic_only_not_paper_results",
    "count_level": count, "mlem_iterations": iterations,
    "operator_id": cfg["forward_model"]["operator_id"],
    "operator_implementation_sha256": cfg["forward_model"]["implementation_sha256"],
    "operator_adjoint_relative_error": abs(left - right) / max(abs(left), abs(right)),
    "operator_per_angle_mass_relative_error": float(
      np.max(np.abs(forward.sum(axis=1) - test_image.sum())) / test_image.sum()
    ),
    "datasets": {},
  }
  for dataset in ("brainweb", "hecktor"):
    sample_path = root / f"data/samples/sample_{dataset}.npz"
    with np.load(sample_path, allow_pickle=False) as archive:
      cases = sorted(key[:-8] for key in archive.files if key.endswith("__target"))
      targets = [archive[f"{case}__target"].astype(np.float64) for case in cases]
      uploaded_shapes = sorted({
        tuple(archive[f"{case}__lambda_{count}"].shape) for case in cases
      })
      if uploaded_shapes != [operator.projection_shape]:
        result["datasets"][dataset] = {
          "sample_path": str(sample_path.relative_to(root)),
          "sample_file_sha256": _sha256(sample_path),
          "cases": len(cases),
          "subjects": len({case.split("__", 1)[0] for case in cases}),
          "compatibility_status": "forensic_projection_shape_mismatch",
          "uploaded_projection_shapes": [list(shape) for shape in uploaded_shapes],
          "canonical_projection_shape": list(operator.projection_shape),
          "action": "Do not reuse uploaded projections or sinograms; regenerate from source activity with the full-square operator.",
        }
        continue
      lambda_l1 = []
      correlations = []
      slopes = []
      intercepts = []
      affine_l1 = []
      matched_metrics = []
      uploaded_metrics = []
      for index, (case, target) in enumerate(zip(cases, targets)):
        projection = operator.forward(target)
        canonical_lambda = count * projection / projection.sum()
        uploaded_lambda = archive[f"{case}__lambda_{count}"].astype(np.float64)
        lambda_l1.append(float(np.abs(uploaded_lambda - canonical_lambda).sum() / count))
        correlations.append(float(np.corrcoef(
          uploaded_lambda.ravel(), canonical_lambda.ravel(),
        )[0, 1]))
        design = np.column_stack((canonical_lambda.ravel(), np.ones(canonical_lambda.size)))
        slope, intercept = np.linalg.lstsq(
          design, uploaded_lambda.ravel(), rcond=None,
        )[0]
        fitted = slope * canonical_lambda + intercept
        slopes.append(float(slope)); intercepts.append(float(intercept))
        affine_l1.append(float(np.abs(uploaded_lambda - fitted).sum() / count))
        if index < reconstruction_cases:
          gain = np.full(operator.projection_shape, count / projection.sum())
          matched = rng.poisson(canonical_lambda).astype(np.int64)
          for measured, destination in (
            (matched, matched_metrics),
            (archive[f"{case}__sinogram_{count}"], uploaded_metrics),
          ):
            reconstruction = mlem(
              measured, operator, iterations=iterations, multiplicative=gain,
            )[iterations]
            destination.append({
              "nrmse": nrmse(target, reconstruction),
              "mae": mae(target, reconstruction),
            })
      first = targets[0]
      expected_by_C = []
      for constant in (1.0, 30000.0):
        projection = operator.forward(first / constant)
        expected_by_C.append(count * projection / projection.sum())
      target_maxima = [float(target.max()) for target in targets]
      target_minima = [float(target.min()) for target in targets]
      dataset_result = {
        "sample_path": str(sample_path.relative_to(root)),
        "sample_file_sha256": _sha256(sample_path), "cases": len(cases),
        "subjects": len({case.split("__", 1)[0] for case in cases}),
        "target_minimum": _summary(target_minima),
        "target_maximum": _summary(target_maxima),
        "target_exact_zero_or_one_fraction": float(np.mean([
          np.mean((target == 0) | (target == 1)) for target in targets
        ])),
        "uploaded_vs_locked_lambda_relative_l1": _summary(lambda_l1),
        "uploaded_vs_locked_lambda_correlation": _summary(correlations),
        "affine_fit_slope": _summary(slopes),
        "affine_fit_intercept": _summary(intercepts),
        "affine_fit_residual_relative_l1": _summary(affine_l1),
        "C_1_vs_30000_expected_count_relative_l1": float(
          np.abs(expected_by_C[0] - expected_by_C[1]).sum() / count
        ),
        "global_or_per_image_minmax_changes_uploaded_target": not (
          all(value == 0.0 for value in target_minima)
          and all(value == 1.0 for value in target_maxima)
        ),
        "matched_mlem": {
          "cases": len(matched_metrics),
          "mean_nrmse": float(np.mean([row["nrmse"] for row in matched_metrics])),
          "mean_mae": float(np.mean([row["mae"] for row in matched_metrics])),
        },
        "uploaded_mlem_with_locked_operator": {
          "cases": len(uploaded_metrics),
          "mean_nrmse": float(np.mean([row["nrmse"] for row in uploaded_metrics])),
          "mean_mae": float(np.mean([row["mae"] for row in uploaded_metrics])),
        },
      }
      result["datasets"][dataset] = dataset_result
  return result


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--root", type=Path, default=Path("."))
  parser.add_argument("--count", type=int, default=200000)
  parser.add_argument("--reconstruction-cases", type=int, default=8)
  parser.add_argument("--iterations", type=int, default=40)
  args = parser.parse_args()
  print(json.dumps(audit(
    args.root.resolve(), count=args.count,
    reconstruction_cases=args.reconstruction_cases, iterations=args.iterations,
  ), indent=2, sort_keys=True))


if __name__ == "__main__":
  main()
