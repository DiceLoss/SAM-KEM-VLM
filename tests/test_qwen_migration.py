import csv
import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "code/experiments/migrate_qwen_partial_batch.py"
SPEC = importlib.util.spec_from_file_location("migrate_qwen_partial_batch", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_migration_preserves_originals_and_resolves_only_approved_incident(tmp_path: Path):
  manifest = tmp_path / "prompts.csv"
  fields = ["dataset", "subject_id", "slice_id", "count_level"]
  with manifest.open("w", encoding="utf-8", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=fields)
    writer.writeheader()
    writer.writerows([
      {"dataset": "brainweb", "subject_id": "s1", "slice_id": "z1", "count_level": 100000},
      {"dataset": "brainweb", "subject_id": "s1", "slice_id": "z1", "count_level": 200000},
      {"dataset": "brainweb", "subject_id": "s1", "slice_id": "z1", "count_level": 500000},
    ])
  policy_sha = "a" * 64
  implementation_sha = "b" * 64
  source = tmp_path / "source"
  success_path = source / "brainweb/s1/z1/c100000.json"
  incident_path = source / "brainweb/s1/z1/c200000.json"
  success_path.parent.mkdir(parents=True)
  success_path.write_text(json.dumps({
    "record": {
      "dataset": "brainweb", "subject_id": "s1", "slice_id": "z1",
      "count_level": 100000, "source": "qwen3-vl-plus",
    },
    "request_policy": {"sha256": policy_sha},
    "request_implementation": {"sha256": implementation_sha},
  }), encoding="utf-8")
  incident_path.write_text(json.dumps({
    "record": {
      "dataset": "brainweb", "subject_id": "s1", "slice_id": "z1",
      "count_level": 200000, "source": "qwen_batch_configuration_failure",
    },
    "attempts": [{
      "status": "failure", "http_status": 400,
      "error": "invalid_parameter_error: provided URL does not appear valid",
    }],
    "request_policy": {"sha256": policy_sha},
    "request_implementation": {"sha256": implementation_sha},
  }), encoding="utf-8")
  original_success_sha = MODULE._sha256(success_path)
  original_incident_sha = MODULE._sha256(incident_path)
  source_set_sha = MODULE._set_sha256(source, MODULE._paths(source))
  forensic = tmp_path / "forensic.json"
  forensic.write_text("{}\n", encoding="utf-8")
  resolution = tmp_path / "resolution.json"
  resolution.write_text(json.dumps({
    "schema_version": 1,
    "resolution_id": "test_resolution",
    "forensic_audit": {"path": "forensic.json", "sha256": MODULE._sha256(forensic)},
    "source_response_set": {"records": 2, "sha256": source_set_sha},
    "request_policy_sha256": policy_sha,
    "request_implementation_sha256": implementation_sha,
    "incident": {
      "relative_path": "brainweb/s1/z1/c200000.json",
      "cache_key": ["brainweb", "s1", "z1", 200000],
      "expected_original_source": "qwen_batch_configuration_failure",
      "expected_http_status": 400,
      "expected_provider_error_code": "invalid_parameter_error",
      "resolved_source": "qwen_request_failure",
    },
  }), encoding="utf-8")
  destination = tmp_path / "destination"
  output_audit = tmp_path / "migration.json"
  audit = MODULE.migrate(
    tmp_path, manifest, source, destination, resolution, output_audit,
  )
  assert audit["byte_identical_copies"] == 1
  assert audit["transformed_records"] == 1
  assert audit["remaining_unprocessed_keys"] == 1
  assert MODULE._sha256(success_path) == original_success_sha
  assert MODULE._sha256(incident_path) == original_incident_sha
  assert MODULE._sha256(destination / success_path.relative_to(source)) == original_success_sha
  migrated = json.loads(
    (destination / incident_path.relative_to(source)).read_text(encoding="utf-8")
  )
  assert migrated["record"]["source"] == "qwen_request_failure"
  assert migrated["incident_resolution"]["original_file_sha256"] == original_incident_sha
  assert migrated["incident_resolution"]["additional_api_attempt"] is False
  assert not (destination / "_SUCCESS.json").exists()
