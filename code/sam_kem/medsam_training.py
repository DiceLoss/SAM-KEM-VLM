"""Manifest-driven MedSAM2 mask-decoder training."""

from __future__ import annotations

import csv
import hashlib
import random
import subprocess
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from .boxes import BoxCache, BoxKey
from .config import load_contract
from .artifacts import atomic_csv_writer, atomic_torch_save, atomic_write_json, file_sha256, finalize_csv, verify_csv_artifact
from .medsam import freeze_except_mask_decoder, pet_to_rgb, sha256_file
from .medsam_backend import build_medsam2
from .splits import load_configured_subject_splits


def _balanced_order(
  rows: list[dict[str, str]], count_levels: tuple[int, ...], *, seed: int,
  total_steps: int,
) -> list[dict[str, str]]:
  groups: dict[int, dict[str, list[dict[str, str]]]] = defaultdict(
    lambda: defaultdict(list)
  )
  for row in rows:
    groups[int(row["count_level"])][row["subject_id"]].append(row)
  if set(groups) != set(count_levels):
    raise ValueError(f"training rows do not cover all counts: {sorted(groups)}")
  subject_sets = {count: set(subject_rows) for count, subject_rows in groups.items()}
  if len({frozenset(subjects) for subjects in subject_sets.values()}) != 1:
    raise ValueError(f"each count must cover identical training subjects: {subject_sets}")
  rng = random.Random(seed)
  if total_steps <= 0 or total_steps % len(count_levels):
    raise ValueError("total_steps must be positive and divisible by count-level count")
  length = total_steps // len(count_levels)
  subjects_by_count = {}
  for count, subject_rows in groups.items():
    subjects = sorted(subject_rows); rng.shuffle(subjects)
    subjects_by_count[count] = subjects
    for cases in subject_rows.values():
      rng.shuffle(cases)
  order = []
  for index in range(length):
    counts = list(count_levels); rng.shuffle(counts)
    for count in counts:
      subjects = subjects_by_count[count]
      subject = subjects[index % len(subjects)]
      subject_cases = groups[count][subject]
      subject_cycle = index // len(subjects)
      order.append(subject_cases[subject_cycle % len(subject_cases)])
  return order


def _box_coordinates_on_model_grid(box, source_shape: tuple[int, int], model_size: int):
  """Map an XYXY box from the 2-D PET grid to MedSAM2's input grid."""
  height, width = source_shape
  if height < 1 or width < 1 or model_size < 1:
    raise ValueError("source shape and MedSAM2 model size must be positive")
  return np.array([[
    box.xmin * model_size / width, box.ymin * model_size / height,
    box.xmax * model_size / width, box.ymax * model_size / height,
  ]], dtype=np.float32)


def _forward_logits(
  model, predictor, image: np.ndarray, box, *, source_shape: tuple[int, int], device,
):
  import torch

  predictor.set_image(image)
  embedding = predictor.get_image_embedding()
  high_resolution = predictor._features["high_res_feats"]
  coordinates = torch.as_tensor(
    _box_coordinates_on_model_grid(box, source_shape, int(model.image_size)),
    dtype=torch.float32, device=device,
  )
  labels = torch.tensor([[2, 3]], dtype=torch.int32, device=device)
  sparse, dense = model.sam_prompt_encoder(
    points=(coordinates.reshape(-1, 2, 2), labels), boxes=None, masks=None,
  )
  low_resolution, _, _, _ = model.sam_mask_decoder(
    image_embeddings=embedding,
    image_pe=model.sam_prompt_encoder.get_dense_pe(),
    sparse_prompt_embeddings=sparse,
    dense_prompt_embeddings=dense,
    multimask_output=False,
    repeat_image=False,
    high_res_features=high_resolution,
  )
  return low_resolution


def train_dataset_decoder(
  root: Path,
  config_path: Path,
  *,
  dataset: str,
  pretrained_checkpoint: Path,
  output_checkpoint: Path,
  manifest_path: Path,
  epochs: int,
  steps_per_epoch: int,
  learning_rate: float,
  resume_checkpoint: Path | None = None,
) -> Path:
  import torch
  import torch.nn.functional as F

  cfg, contract = load_contract(config_path)
  code_commit = subprocess.run(
    ["git", "rev-parse", "HEAD"], cwd=root, check=True,
    capture_output=True, text=True,
  ).stdout.strip()
  dirty_lines = subprocess.run(
    ["git", "status", "--porcelain"], cwd=root, check=True,
    capture_output=True, text=True,
  ).stdout.splitlines()
  allowed_previous_outputs = set()
  for configured_path in cfg["medsam2"]["checkpoint_paths"].values():
    checkpoint = Path(str(configured_path))
    allowed_previous_outputs.update({
      str(checkpoint.with_suffix(".json")),
      str(checkpoint.with_suffix(".training.csv")),
      str(checkpoint.with_suffix(".training.csv.manifest.json")),
    })
  unexpected_dirty = []
  for line in dirty_lines:
    relative = line[3:].strip()
    if relative not in allowed_previous_outputs:
      unexpected_dirty.append(line)
  if unexpected_dirty:
    raise RuntimeError(
      "refusing to train with uncommitted inputs/code; unexpected Git status: "
      f"{unexpected_dirty[:5]}"
    )
  config_sha = hashlib.sha256(config_path.read_bytes()).hexdigest()
  declared_training = cfg["medsam2"]["training"]
  supplied = {
    "epochs": int(epochs), "steps_per_epoch": int(steps_per_epoch),
    "learning_rate": float(learning_rate),
  }
  declared = {
    "epochs": int(declared_training["epochs"]),
    "steps_per_epoch": int(declared_training["steps_per_epoch"]),
    "learning_rate": float(declared_training["learning_rate"]),
  }
  if supplied != declared:
    raise ValueError(f"training CLI differs from predeclared protocol: {supplied} != {declared}")
  seed = int(contract.experiment_seed)
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
  if cfg["reproducibility"].get("deterministic_algorithms", False):
    torch.use_deterministic_algorithms(True)
    if hasattr(torch.backends, "cudnn"):
      torch.backends.cudnn.benchmark = False
  if not pretrained_checkpoint.is_file():
    raise FileNotFoundError(f"pretrained MedSAM2 checkpoint missing: {pretrained_checkpoint}")
  expected_pretrained_sha = cfg["medsam2"].get("pretrained_checkpoint_sha256")
  actual_pretrained_sha = sha256_file(pretrained_checkpoint)
  if not expected_pretrained_sha or actual_pretrained_sha != expected_pretrained_sha:
    raise ValueError(
      f"pretrained MedSAM2 SHA mismatch/unresolved: expected={expected_pretrained_sha}, actual={actual_pretrained_sha}"
    )
  split_rows = load_configured_subject_splits(root, cfg)
  train_subjects = {
    row["subject_id"] for row in split_rows
    if row["dataset"] == dataset and row["split"] == "train"
  }
  manifest_record = verify_csv_artifact(manifest_path)
  if manifest_record.get("schema_name") != "canonical_case_manifest_v3":
    raise ValueError("decoder training requires a canonical v3 case manifest")
  rows = [
    row for row in csv.DictReader(manifest_path.open("r", encoding="utf-8", newline=""))
    if row["dataset"] == dataset and row["split"] == "train"
  ]
  if any(row["subject_id"] not in train_subjects for row in rows):
    raise ValueError("manifest contains non-train subject in training rows")
  for column in ("run_id", "split_id", "split_sha256", "mlem10_input_path", "target_path"):
    if not rows or column not in rows[0]:
      raise ValueError(f"training manifest missing {column}")
  for row in rows:
    if row["run_id"] != contract.run_id:
      raise ValueError(f"training case run_id mismatch: {row['subject_id']}/{row['slice_id']}")
    if row["split_id"] != contract.split_id or row["split_sha256"] != contract.split_sha256:
      raise ValueError(f"training case split mismatch: {row['subject_id']}/{row['slice_id']}")
    if row.get("operator_id") != contract.operator_id:
      raise ValueError(f"training case operator mismatch: {row['subject_id']}/{row['slice_id']}")
    if float(row.get("scale_constant_C", "nan")) != contract.scale_constants[dataset]:
      raise ValueError(f"training case scale mismatch: {row['subject_id']}/{row['slice_id']}")
    if (
      row.get("normalization_id") != contract.normalization_id
      or float(row.get("normalization_offset", "nan")) != contract.normalization_offsets[dataset]
      or (row.get("normalization_clip", "").lower() == "true") != contract.normalization_clip
    ):
      raise ValueError(f"training case normalization mismatch: {row['subject_id']}/{row['slice_id']}")
  box_cache = BoxCache.read(root / str(cfg["prompts"]["training_box_cache"]))

  medsam_source = root / str(cfg["medsam2"].get("source_checkout_path") or "")
  try:
    source_head = subprocess.run(
      ["git", "rev-parse", "HEAD"], cwd=medsam_source, check=True,
      capture_output=True, text=True,
    ).stdout.strip()
  except (OSError, subprocess.CalledProcessError) as error:
    raise ValueError(f"invalid MedSAM2 source checkout: {medsam_source}") from error
  if source_head != cfg["medsam2"].get("source_commit"):
    raise ValueError("MedSAM2 checkout does not match configured source_commit")
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  model = build_medsam2(
    source_checkout=medsam_source,
    architecture=str(cfg["medsam2"]["architecture"]),
    checkpoint=pretrained_checkpoint, device=device, mode="train",
  )
  from sam2.sam2_image_predictor import SAM2ImagePredictor
  predictor = SAM2ImagePredictor(model)
  freeze_except_mask_decoder(model)
  model.eval()
  model.sam_mask_decoder.train()
  model.to(device)
  optimizer = torch.optim.AdamW(
    [parameter for parameter in model.parameters() if parameter.requires_grad],
    lr=learning_rate, weight_decay=float(declared_training["weight_decay"]),
  )
  resume_path = output_checkpoint.with_suffix(".resume.pt")
  metrics_path = output_checkpoint.with_suffix(".training.csv")
  provenance = output_checkpoint.with_suffix(".json")
  immutable_outputs = (
    output_checkpoint, provenance, metrics_path,
    metrics_path.with_suffix(metrics_path.suffix + ".manifest.json"),
  )
  if any(path.exists() for path in immutable_outputs):
    raise FileExistsError(f"immutable training output already exists: {output_checkpoint}")
  training_counts: Counter[int] = Counter()
  training_subject_samples: Counter[str] = Counter()
  sampled_subjects: set[str] = set()
  sampled_case_counts: Counter[str] = Counter()
  fallbacks = 0
  epoch_summaries: list[dict] = []
  start_epoch = 0
  if resume_checkpoint is not None:
    if resume_checkpoint.resolve() != resume_path.resolve():
      raise ValueError(f"resume path must be the canonical rolling state: {resume_path}")
    state = torch.load(resume_checkpoint, map_location=device, weights_only=False)
    if (
      state.get("schema_version") != 1 or state.get("artifact_kind") != "training_resume"
      or state.get("dataset") != dataset
      or state.get("run_id") != contract.run_id
      or state.get("normalization_id") != contract.normalization_id
      or state.get("pretrained_checkpoint_sha256") != actual_pretrained_sha
      or state.get("config_sha256") != config_sha
    ):
      raise ValueError("resume checkpoint identity/protocol mismatch")
    model.sam_mask_decoder.load_state_dict(state["mask_decoder_state"], strict=True)
    optimizer.load_state_dict(state["optimizer_state"])
    random.setstate(state["python_rng_state"])
    np.random.set_state(state["numpy_rng_state"])
    torch.set_rng_state(state["torch_cpu_rng_state"])
    if torch.cuda.is_available() and state.get("torch_cuda_rng_states") is not None:
      torch.cuda.set_rng_state_all(state["torch_cuda_rng_states"])
    start_epoch = int(state["completed_epochs"])
    training_counts.update({int(k): int(v) for k, v in state["training_count_samples"].items()})
    training_subject_samples.update(state["training_subject_samples"])
    sampled_subjects.update(state["sampled_subjects"])
    sampled_case_counts.update(state["sampled_case_counts"])
    fallbacks = int(state["box_fallback_samples"])
    epoch_summaries = list(state["epoch_summaries"])
  elif resume_path.exists():
    raise FileExistsError(
      f"rolling resume state exists; pass --resume {resume_path} or move it aside"
    )
  if start_epoch >= epochs:
    raise ValueError(f"resume checkpoint already completed {start_epoch}/{epochs} epochs")
  complete_order = _balanced_order(
    rows, contract.count_levels, seed=contract.experiment_seed,
    total_steps=epochs * steps_per_epoch,
  )
  for epoch in range(start_epoch, epochs):
    epoch_losses: list[float] = []
    order = complete_order[epoch * steps_per_epoch:(epoch + 1) * steps_per_epoch]
    if steps_per_epoch % len(contract.count_levels):
      raise ValueError("steps_per_epoch must be divisible by the number of count levels")
    for row in order[:steps_per_epoch]:
      mlem_image = np.load(root / row["mlem10_input_path"]).astype(np.float32)
      target = np.load(root / row["target_path"]).astype(np.float32)
      key = BoxKey(dataset, row["subject_id"], row["slice_id"], int(row["count_level"]))
      record = box_cache.get(key)
      if record is None:
        raise ValueError(f"canonical box cache is incomplete for training key: {key}")
      if record.run_id != contract.run_id:
        raise ValueError(f"box cache run_id mismatch for training key: {key}")
      box = record.box
      fallbacks += int(record.source != "qwen3-vl-plus")
      logits = _forward_logits(
        model, predictor, pet_to_rgb(mlem_image), box,
        source_shape=mlem_image.shape, device=device,
      )
      prediction = F.interpolate(logits, size=target.shape, mode="bilinear", align_corners=False)[0, 0]
      target_tensor = torch.from_numpy(target).to(device=device, dtype=prediction.dtype)
      loss = F.l1_loss(prediction, target_tensor)
      optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
      epoch_losses.append(float(loss.detach().cpu()))
      training_counts[int(row["count_level"])] += 1
      training_subject_samples[row["subject_id"]] += 1
      sampled_subjects.add(row["subject_id"])
      sampled_case_counts[
        "|".join((
          row["subject_id"], row["slice_id"], row["count_level"],
          row["realization_id"],
        ))
      ] += 1
    epoch_summaries.append({
      "epoch": epoch + 1, "global_step": (epoch + 1) * steps_per_epoch,
      "mean_l1": float(np.mean(epoch_losses)), "std_l1": float(np.std(epoch_losses)),
      "min_l1": float(np.min(epoch_losses)), "max_l1": float(np.max(epoch_losses)),
      "samples": len(epoch_losses),
    })
    atomic_torch_save({
      "schema_version": 1, "artifact_kind": "training_resume", "dataset": dataset,
      "run_id": contract.run_id,
      "completed_epochs": epoch + 1, "global_step": (epoch + 1) * steps_per_epoch,
      "mask_decoder_state": model.sam_mask_decoder.state_dict(),
      "optimizer_state": optimizer.state_dict(),
      "python_rng_state": random.getstate(), "numpy_rng_state": np.random.get_state(),
      "torch_cpu_rng_state": torch.get_rng_state(),
      "torch_cuda_rng_states": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
      "training_count_samples": dict(training_counts),
      "training_subject_samples": dict(training_subject_samples),
      "sampled_subjects": sorted(sampled_subjects),
      "sampled_case_counts": dict(sampled_case_counts),
      "box_fallback_samples": fallbacks, "epoch_summaries": epoch_summaries,
      "pretrained_checkpoint_sha256": actual_pretrained_sha,
      "normalization_id": contract.normalization_id,
      "config_sha256": config_sha, "code_git_commit": code_commit,
      "medsam2_source_commit": source_head,
    }, resume_path, replace=True)

  if sampled_subjects != train_subjects:
    raise RuntimeError(
      "training schedule did not cover every declared training subject: "
      f"missing={sorted(train_subjects - sampled_subjects)}"
    )

  output_checkpoint.parent.mkdir(parents=True, exist_ok=True)
  parameter_inventory = {
    name: {"requires_grad": bool(parameter.requires_grad), "numel": parameter.numel()}
    for name, parameter in model.named_parameters()
  }
  atomic_torch_save({
    "schema_version": 1, "artifact_kind": "inference_mask_decoder",
    "run_id": contract.run_id,
    "selection_policy": "fixed_budget_final_checkpoint_no_best_claim",
    "mask_decoder_state": model.sam_mask_decoder.state_dict(), "dataset": dataset,
    "normalization_id": contract.normalization_id,
    "train_subject_ids": sorted(sampled_subjects), "training_count_samples": dict(training_counts),
    "training_subject_samples": dict(sorted(training_subject_samples.items())),
    "sampled_case_counts": dict(sorted(sampled_case_counts.items())),
    "epochs": epochs, "steps_per_epoch": steps_per_epoch, "learning_rate": learning_rate,
    "loss": "l1_only_raw_decoder_output", "epoch_summaries": epoch_summaries,
    "optimizer": "AdamW", "weight_decay": float(declared_training["weight_decay"]),
    "batch_size": 1, "sampling_unit": declared_training["sampling_unit"],
    "box_fallback_samples": fallbacks,
    "config_path": str(config_path), "pretrained_checkpoint_sha256": actual_pretrained_sha,
    "config_sha256": config_sha, "code_git_commit": code_commit,
    "medsam2_source_commit": source_head,
  }, output_checkpoint)
  metrics_fields = [
    "epoch", "global_step", "mean_l1", "std_l1", "min_l1", "max_l1", "samples",
  ]
  with atomic_csv_writer(metrics_path, metrics_fields) as writer:
    writer.writerows(epoch_summaries)
  metrics_manifest = finalize_csv(
    metrics_path, schema_name="medsam2_training_epoch_metrics_v1",
    primary_key=["epoch"], split="train",
  )
  checkpoint_sha = file_sha256(output_checkpoint)
  resume_sha = file_sha256(resume_path)
  atomic_write_json(provenance, {
    "schema_version": 1, "artifact_kind": "checkpoint_provenance",
    "run_id": contract.run_id,
    "checkpoint_sha256": checkpoint_sha, "checkpoint_scope": "mask_decoder_only",
    "resume_checkpoint_sha256": resume_sha,
    "training_metrics_sha256": file_sha256(metrics_path),
    "training_metrics_manifest_sha256": file_sha256(metrics_manifest),
    "selection_policy": "fixed_budget_final_checkpoint_no_best_claim",
    "completed_epochs": epochs, "global_step": epochs * steps_per_epoch,
    "dataset": dataset,
    "normalization_id": contract.normalization_id,
    "train_subject_ids": sorted(sampled_subjects), "training_count_samples": dict(training_counts),
    "training_subject_samples": dict(sorted(training_subject_samples.items())),
    "sampled_case_counts": dict(sorted(sampled_case_counts.items())),
    "experiment_seed": seed,
    "config_sha256": config_sha, "code_git_commit": code_commit,
    "medsam2_source_commit": source_head,
    "deterministic_algorithms": bool(cfg["reproducibility"].get("deterministic_algorithms", False)),
    "box_fallback_samples": fallbacks,
    "trainable_parameter_names": sorted(
      name for name, parameter in model.named_parameters() if parameter.requires_grad
    ),
    "non_mask_decoder_trainable_names": sorted(
      name for name, parameter in model.named_parameters()
      if parameter.requires_grad and not name.startswith("sam_mask_decoder.")
    ),
    "loss": "l1_only_raw_decoder_output",
    "optimizer": "AdamW", "weight_decay": float(declared_training["weight_decay"]),
    "batch_size": 1, "sampling_unit": declared_training["sampling_unit"],
    "parameter_inventory": parameter_inventory,
  })
  return output_checkpoint
