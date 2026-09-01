import hashlib
from pathlib import Path

import pytest

from sam_kem.artifacts import atomic_csv_writer, atomic_write_json, file_sha256, finalize_csv
from sam_kem.sweep_evaluation import SWEEP_FIELDS
from sam_kem.sweep_sharded import (
  PRIMARY_KEY,
  line_ending_equivalent_sha256,
  verify_sweep_shard,
)


def _shard() -> dict:
  return {
    "shard_id": "shard_0000", "dataset": "brainweb",
    "subject_id": "subject04", "slice_id": "z038",
    "case_count": 1, "expected_result_rows": 580,
    "case_keys": [{"count_level": 200000, "realization_id": 0}],
  }


def _write_complete_shard(output_dir: Path, shard: dict) -> None:
  directory = output_dir / "brainweb" / "subject04" / "z038" / "shard_0000"
  fragment = directory / "results.csv"
  with atomic_csv_writer(fragment, SWEEP_FIELDS) as writer:
    for index in range(580):
      writer.writerow({
        "run_id": "run", "dataset": "brainweb", "subject_id": "subject04",
        "slice_id": "z038", "realization_id": 0, "count_level": 200000,
        "method": f"method_{index}", "split": "validation", "sigma": 0.0,
        "radius": 0, "neighbours": 0, "iterations": 1, "osem_subsets": 1,
        "effective_em_updates": 1, "alpha": 0.0, "nrmse": 0.1, "ssim": 0.9,
      })
  sidecar = finalize_csv(
    fragment, schema_name="validation_sweep_slice_shard_v1",
    primary_key=PRIMARY_KEY, split="validation",
  )
  atomic_write_json(directory / "_SUCCESS.json", {
    "artifact_kind": "validation_sweep_slice_shard_success",
    "status": "complete", "shard_id": "shard_0000",
    "execution_git_commit": "commit", "execution_record_sha256": "e" * 64,
    "plan_sha256": "p" * 64, "row_count": 580,
    "fragment_sha256": file_sha256(fragment),
    "fragment_manifest_sha256": file_sha256(sidecar),
  })


def test_complete_shard_passes_strict_resume_verification(tmp_path):
  output_dir = tmp_path / "runs" / "run" / "hyperparameter_sweep_shards"
  shard = _shard()
  _write_complete_shard(output_dir, shard)
  result = verify_sweep_shard(
    output_dir, shard, execution_commit="commit",
    execution_record_sha256="e" * 64, plan_sha256="p" * 64, run_id="run",
  )
  assert result is not None
  assert result["row_count"] == 580
  assert result["fragment_sha256"]


def test_partial_shard_is_never_treated_as_resumable(tmp_path):
  output_dir = tmp_path / "runs" / "run" / "hyperparameter_sweep_shards"
  fragment = (
    output_dir / "brainweb" / "subject04" / "z038" / "shard_0000" / "results.csv"
  )
  fragment.parent.mkdir(parents=True)
  fragment.write_text("incomplete\n", encoding="utf-8")
  with pytest.raises(ValueError, match="incomplete shard"):
    verify_sweep_shard(
      output_dir, _shard(), execution_commit="commit",
      execution_record_sha256="e" * 64, plan_sha256="p" * 64, run_id="run",
    )


def test_unpublished_transaction_directory_does_not_claim_shard(tmp_path):
  output_dir = tmp_path / "runs" / "run" / "hyperparameter_sweep_shards"
  temporary = output_dir / "brainweb" / "subject04" / "z038" / ".shard_0000.crash"
  temporary.mkdir(parents=True)
  (temporary / "results.csv").write_text("interrupted\n", encoding="utf-8")
  assert verify_sweep_shard(
    output_dir, _shard(), execution_commit="commit",
    execution_record_sha256="e" * 64, plan_sha256="p" * 64, run_id="run",
  ) is None


def test_protocol_sha_accepts_only_line_ending_equivalence(tmp_path):
  path = tmp_path / "protocol.yaml"
  path.write_bytes(b"name: sweep\nkind: hyperparameter_sweep\n")
  crlf_sha = hashlib.sha256(
    b"name: sweep\r\nkind: hyperparameter_sweep\r\n"
  ).hexdigest()
  assert line_ending_equivalent_sha256(path, crlf_sha)
  path.write_bytes(b"name: changed\nkind: hyperparameter_sweep\n")
  assert not line_ending_equivalent_sha256(path, crlf_sha)
