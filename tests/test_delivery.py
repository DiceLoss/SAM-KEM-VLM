import json
import subprocess
import zipfile

import pytest
from sam_kem.artifacts import (
  atomic_csv_writer,
  file_sha256,
  finalize_csv,
)
from sam_kem.delivery import build_decoder_checkpoint_bundle


def test_decoder_bundle_verifies_groups_and_refuses_overwrite(tmp_path):
  root = tmp_path / "repository"
  root.mkdir()
  subprocess.run(["git", "init"], cwd=root, check=True, capture_output=True)
  config = root / "code/configs/generated/test.yaml"
  config.parent.mkdir(parents=True)
  config.write_text(
    "run_id: test_run\n"
    "medsam2:\n"
    "  checkpoint_paths:\n"
    "    brainweb: checkpoints/test/brainweb.pt\n"
    "    hecktor: checkpoints/test/hecktor.pt\n",
    encoding="utf-8",
  )
  protocol_dir = config.with_name("test_protocols")
  protocol_dir.mkdir()
  (protocol_dir / "training.yaml").write_text("protocol: test\n", encoding="utf-8")

  dataset_records = {}
  for dataset in ("brainweb", "hecktor"):
    checkpoint = root / f"checkpoints/test/{dataset}.pt"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    checkpoint.write_bytes(f"{dataset}-checkpoint".encode())
    resume = checkpoint.with_suffix(".resume.pt")
    resume.write_bytes(f"{dataset}-resume".encode())
    metrics = checkpoint.with_suffix(".training.csv")
    with atomic_csv_writer(metrics, ["epoch", "mean_l1"]) as writer:
      writer.writerow({"epoch": 1, "mean_l1": 0.1})
    metrics_manifest = finalize_csv(
      metrics, schema_name="test_training", primary_key=["epoch"], split="train",
    )
    provenance = checkpoint.with_suffix(".json")
    provenance.write_text(json.dumps({
      "artifact_kind": "checkpoint_provenance",
      "dataset": dataset,
      "run_id": "test_run",
      "config_sha256": file_sha256(config),
      "checkpoint_sha256": file_sha256(checkpoint),
      "resume_checkpoint_sha256": file_sha256(resume),
      "training_metrics_sha256": file_sha256(metrics),
      "training_metrics_manifest_sha256": file_sha256(metrics_manifest),
      "code_git_commit": "a" * 40 if dataset == "brainweb" else "b" * 40,
    }) + "\n", encoding="utf-8")
    dataset_records[dataset] = {
      "checkpoint_sha256": file_sha256(checkpoint),
    }

  logs = root / "logs"
  logs.mkdir()
  (logs / "gpu_test_run_complete.log").write_text("training complete\n", encoding="utf-8")
  (logs / "decoder_training_handoff_audit.json").write_text(json.dumps({
    "status": "complete", "run_id": "test_run",
    "datasets": dataset_records,
  }) + "\n", encoding="utf-8")
  (logs / "decoder_training_commit_identity_correction.json").write_text(
    '{"status":"corrected"}\n', encoding="utf-8",
  )
  subprocess.run(["git", "add", "."], cwd=root, check=True)
  subprocess.run(
    [
      "git", "-c", "user.name=Test", "-c", "user.email=test@example.invalid",
      "commit", "-m", "fixture",
    ],
    cwd=root, check=True, capture_output=True,
  )

  archive = tmp_path / "decoder_bundle.zip"
  record = tmp_path / "decoder_bundle.json"
  result = build_decoder_checkpoint_bundle(root, config, archive, record)

  assert result["status"] == "local_bundle_complete_pending_immutable_upload"
  assert result["archive_sha256"] == file_sha256(archive)
  assert result["datasets"]["brainweb"]["checkpoint_sha256"] == (
    dataset_records["brainweb"]["checkpoint_sha256"]
  )
  with zipfile.ZipFile(archive) as bundle:
    names = set(bundle.namelist())
    assert "MANIFEST.sha256" in names
    assert "environment/BUNDLE_METADATA.json" in names
    assert "checkpoints/test/brainweb.pt" in names
    assert "checkpoints/test/hecktor.resume.pt" in names
    assert len(bundle.read("MANIFEST.sha256").splitlines()) == result["manifest_member_count"]

  with pytest.raises(FileExistsError, match="overwrite"):
    build_decoder_checkpoint_bundle(root, config, archive, record)
