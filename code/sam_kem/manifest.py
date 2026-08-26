"""Build the canonical case manifest from the documented array layout."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from .config import load_contract
from .reproducibility import CaseKey, case_seed
from .splits import load_subject_splits
from .noise import array_sha256
from .artifacts import atomic_csv_writer, finalize_csv


FIELDS = [
  "dataset", "subject_id", "slice_id", "realization_id", "count_level", "split",
  "seed", "normalization_id", "normalization_offset", "normalization_clip",
  "scale_constant_C", "operator_id", "target_path",
  "noise_free_projection_path", "projection_metadata_path", "projection_scale", "sinogram_path",
  "simulation_metadata_path", "realized_total_counts", "mlem10_input_path",
  "mlem10_metadata_path",
  "anatomy_path", "anatomy_sha256", "evaluation_mask_path", "sam_prior_path",
  "sam_prior_metadata_path", "sam_prior_centroid_path",
  "sam_prior_centroid_metadata_path", "sam_prior_released_path",
  "sam_prior_released_metadata_path",
]


def _relative(path: Path, root: Path) -> str:
  return path.relative_to(root).as_posix()


def build_case_manifest(root: Path, config_path: Path, arrays_root: Path, output: Path) -> Path:
  cfg, contract = load_contract(config_path)
  splits = load_subject_splits(root / str(cfg["data"]["split_file"]))
  split_index = {(row["dataset"], row["subject_id"]): row["split"] for row in splits}
  records = []
  for dataset_dir in sorted(path for path in arrays_root.iterdir() if path.is_dir()):
    dataset = dataset_dir.name
    for subject_dir in sorted(path for path in dataset_dir.iterdir() if path.is_dir()):
      key = (dataset, subject_dir.name)
      if key not in split_index:
        raise ValueError(f"array subject has no split assignment: {key}")
      for slice_dir in sorted(path for path in subject_dir.iterdir() if path.is_dir()):
        target = slice_dir / "target.npy"
        projection = slice_dir / "noise_free_projection.npy"
        projection_metadata = projection.with_suffix(".json")
        anatomy = slice_dir / "anatomy.npy"
        mask = slice_dir / "evaluation_mask.npy"
        if mask.is_file() and cfg["evaluation"]["region"] == "full_128x128_fov":
          raise ValueError(
            f"evaluation mask is prohibited by the full-FOV protocol: {mask}"
          )
        for count in contract.count_levels:
          for realization in contract.realizations:
            case_dir = slice_dir / f"c{count}" / f"r{realization}"
            sinogram = case_dir / "sinogram.npy"
            simulation_metadata = sinogram.with_suffix(".json")
            mlem10 = case_dir / "mlem10.npy"
            mlem10_metadata = mlem10.with_suffix(".json")
            required = (
              target, projection, projection_metadata,
              sinogram, simulation_metadata, mlem10, mlem10_metadata,
            )
            missing = [str(path) for path in required if not path.is_file()]
            if missing:
              raise FileNotFoundError(f"incomplete case {key}/{slice_dir.name}/c{count}/r{realization}: {missing}")
            identity = CaseKey(dataset, subject_dir.name, slice_dir.name, count, realization)
            projection_array = np.load(projection)
            sinogram_array = np.load(sinogram)
            projection_record = json.loads(projection_metadata.read_text(encoding="utf-8"))
            projection_sum = float(projection_array.sum())
            if projection_sum <= 0:
              raise ValueError(f"non-positive projection sum: {projection}")
            if (
              projection_record.get("dataset") != dataset
              or projection_record.get("operator_id") != contract.operator_id
              or projection_record.get("normalization_id") != contract.normalization_id
              or float(projection_record.get("normalization_offset", "nan"))
              != contract.normalization_offsets[dataset]
              or bool(projection_record.get("normalization_clip"))
              != contract.normalization_clip
              or projection_record.get("field_of_view") != cfg["forward_model"]["field_of_view"]
              or float(projection_record.get("out_of_fov_activity_mass_fraction", 1.0))
              > float(cfg["forward_model"]["out_of_fov_activity_mass_tolerance"])
              or float(projection_record.get("fixed_scale_C", -1))
              != contract.scale_constants[dataset]
              or projection_record.get("target_sha256") != array_sha256(np.load(target))
              or projection_record.get("noise_free_projection_sha256")
              != array_sha256(projection_array)
            ):
              raise ValueError(f"noise-free projection provenance mismatch: {projection_metadata}")
            expected_seed = case_seed(identity, experiment_seed=contract.experiment_seed)
            expected_scale = float(count) / projection_sum
            metadata = json.loads(simulation_metadata.read_text(encoding="utf-8"))
            expected_identity = {
              "dataset": dataset, "subject_id": subject_dir.name,
              "slice_id": slice_dir.name, "count_level": count,
              "realization_id": realization,
            }
            if any(metadata.get(name) != value for name, value in expected_identity.items()):
              raise ValueError(f"simulation identity mismatch: {simulation_metadata}")
            if (
              metadata.get("observation_model")
              != "independent_poisson_scaled_noise_free_projection"
              or metadata.get("rng") != "numpy.random.Generator(PCG64)"
            ):
              raise ValueError(f"simulation model/RNG mismatch: {simulation_metadata}")
            if int(metadata.get("seed", -1)) != expected_seed:
              raise ValueError(f"simulation seed mismatch: {simulation_metadata}")
            if not np.isclose(float(metadata.get("projection_scale", -1)), expected_scale, rtol=1e-12):
              raise ValueError(f"simulation scale mismatch: {simulation_metadata}")
            if metadata.get("noise_free_projection_sha256") != array_sha256(projection_array):
              raise ValueError(f"projection hash mismatch: {simulation_metadata}")
            if metadata.get("sinogram_sha256") != array_sha256(sinogram_array):
              raise ValueError(f"sinogram hash mismatch: {simulation_metadata}")
            mlem_record = json.loads(mlem10_metadata.read_text(encoding="utf-8"))
            mlem_array = np.load(mlem10)
            if (
              mlem_record.get("method") != "MLEM"
              or int(mlem_record.get("iterations", -1)) != contract.prompt_mlem_iterations
              or mlem_record.get("operator_id") != contract.operator_id
              or mlem_record.get("operator_implementation_sha256")
              != cfg["forward_model"]["implementation_sha256"]
              or mlem_record.get("sinogram_sha256") != array_sha256(sinogram_array)
              or mlem_record.get("mlem10_sha256") != array_sha256(mlem_array)
              or not np.isclose(
                float(mlem_record.get("projection_scale", -1)), expected_scale, rtol=1e-12,
              )
            ):
              raise ValueError(f"MLEM-10 provenance mismatch: {mlem10_metadata}")
            def optional_prior(name: str) -> tuple[str, str]:
              path = case_dir / name
              if not path.is_file():
                return "", ""
              sidecar = path.with_suffix(".json")
              if not sidecar.is_file():
                raise FileNotFoundError(f"prior lacks provenance sidecar: {sidecar}")
              record = json.loads(sidecar.read_text(encoding="utf-8"))
              prior = np.load(path)
              expected_checkpoint = (
                cfg["medsam2"]["pretrained_checkpoint_sha256"]
                if name == "sam_prior_released.npy"
                else cfg["medsam2"]["checkpoint_sha256"][dataset]
              )
              if (
                record.get("prior_sha256") != array_sha256(prior)
                or record.get("input_mlem10_sha256") != array_sha256(np.load(mlem10))
                or record.get("dataset") != dataset
                or record.get("subject_id") != subject_dir.name
                or record.get("slice_id") != slice_dir.name
                or int(record.get("count_level", -1)) != count
                or int(record.get("realization_id", -1)) != realization
                or record.get("normalization_id") != contract.normalization_id
                or record.get("decoder_checkpoint_sha256") != expected_checkpoint
              ):
                raise ValueError(f"prior provenance mismatch: {sidecar}")
              return _relative(path, root), _relative(sidecar, root)
            sam_prior, sam_prior_meta = optional_prior("sam_prior.npy")
            centroid_prior, centroid_meta = optional_prior("sam_prior_centroid.npy")
            released_prior, released_meta = optional_prior("sam_prior_released.npy")
            records.append({
              "dataset": dataset, "subject_id": subject_dir.name,
              "slice_id": slice_dir.name, "realization_id": realization,
              "count_level": count, "split": split_index[key],
              "seed": expected_seed,
              "normalization_id": contract.normalization_id,
              "normalization_offset": contract.normalization_offsets[dataset],
              "normalization_clip": contract.normalization_clip,
              "scale_constant_C": contract.scale_constants[dataset],
              "operator_id": contract.operator_id,
              "target_path": _relative(target, root),
              "noise_free_projection_path": _relative(projection, root),
              "projection_metadata_path": _relative(projection_metadata, root),
              "projection_scale": expected_scale,
              "sinogram_path": _relative(sinogram, root),
              "simulation_metadata_path": _relative(simulation_metadata, root),
              "realized_total_counts": int(metadata["realized_total_counts"]),
              "mlem10_input_path": _relative(mlem10, root),
              "mlem10_metadata_path": _relative(mlem10_metadata, root),
              "anatomy_path": _relative(anatomy, root) if anatomy.is_file() else "",
              "anatomy_sha256": array_sha256(np.load(anatomy)) if anatomy.is_file() else "",
              "evaluation_mask_path": _relative(mask, root) if mask.is_file() else "",
              "sam_prior_path": sam_prior,
              "sam_prior_metadata_path": sam_prior_meta,
              "sam_prior_centroid_path": centroid_prior,
              "sam_prior_centroid_metadata_path": centroid_meta,
              "sam_prior_released_path": released_prior,
              "sam_prior_released_metadata_path": released_meta,
            })
  with atomic_csv_writer(output, FIELDS) as writer:
    writer.writerows(records)
  finalize_csv(
    output, schema_name="canonical_case_manifest_v2",
    primary_key=[
      "dataset", "subject_id", "slice_id", "realization_id", "count_level",
    ],
  )
  return output
