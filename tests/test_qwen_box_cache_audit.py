import csv
import json
from pathlib import Path

import pytest
from experiments.audit_qwen_box_cache import audit
from sam_kem.artifacts import finalize_csv


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
  with path.open("w", encoding="utf-8", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)


def _fixture(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
  diagnostics = tmp_path / "diagnostics.csv"
  diagnostic_rows = [
    {
      "dataset": "brainweb", "subject_id": "s1", "slice_id": "z001",
      "count_level": 100000, "source": "qwen3-vl-plus", "request_id": "r1",
      "image_sha256": "image-1", "xmin": 1, "ymin": 2, "xmax": 30, "ymax": 40,
    },
    {
      "dataset": "hecktor", "subject_id": "2", "slice_id": "z072",
      "count_level": 200000, "source": "qwen_request_failure", "request_id": "",
      "image_sha256": "image-2", "xmin": "", "ymin": "", "xmax": "", "ymax": "",
    },
  ]
  _write_csv(diagnostics, diagnostic_rows)
  from sam_kem.artifacts import file_sha256

  summary = tmp_path / "summary.json"
  summary.write_text(json.dumps({
    "diagnostics_csv_sha256": file_sha256(diagnostics),
    "failure_records": [{
      "dataset": "hecktor", "subject_id": "2", "slice_id": "z072",
      "count_level": 200000,
    }],
  }) + "\n", encoding="utf-8")
  decision = tmp_path / "decision.json"
  decision.write_text(json.dumps({
    "decision": {
      "status": "approved_to_build_development_box_cache_without_post_hoc_box_changes",
    },
    "evidence": {
      "semantic_summary_sha256": file_sha256(summary),
      "diagnostics_csv_sha256_lf": file_sha256(diagnostics),
    },
  }) + "\n", encoding="utf-8")
  cache = tmp_path / "cache.csv"
  base = {
    "run_id": "run", "source_mlem10_sha256": "mlem", "prompt_sha256": "prompt",
    "model": "qwen3-vl-plus", "request_policy_id": "policy",
    "request_policy_sha256": "policy-sha",
  }
  _write_csv(cache, [
    {
      **base, "dataset": "brainweb", "subject_id": "s1", "slice_id": "z001",
      "count_level": 100000, "xmin": 1, "ymin": 2, "xmax": 30, "ymax": 40,
      "source": "qwen3-vl-plus", "request_id": "r1", "image_sha256": "image-1",
    },
    {
      **base, "dataset": "hecktor", "subject_id": "2", "slice_id": "z072",
      "count_level": 200000, "xmin": 48, "ymin": 48, "xmax": 80, "ymax": 80,
      "source": "centroid_fallback_qwen_request_failure", "request_id": "",
      "image_sha256": "image-2",
    },
  ])
  finalize_csv(
    cache, schema_name="qwen_anchor_box_cache_v3",
    primary_key=["run_id", "dataset", "subject_id", "slice_id", "count_level"],
  )
  return cache, diagnostics, summary, decision


def test_audit_accepts_exact_qwen_and_declared_failure_fallback(tmp_path: Path):
  cache, diagnostics, summary, decision = _fixture(tmp_path)
  result = audit(cache, diagnostics, summary, decision, tmp_path / "audit.json")
  assert result["records"] == 2
  assert result["exact_qwen_boxes"] == 1
  assert result["centroid_fallback_boxes"] == 1


def test_audit_rejects_changed_qwen_coordinates(tmp_path: Path):
  cache, diagnostics, summary, decision = _fixture(tmp_path)
  rows = list(csv.DictReader(cache.open("r", encoding="utf-8", newline="")))
  rows[0]["xmin"] = "2"
  cache.unlink()
  cache.with_suffix(".csv.manifest.json").unlink()
  _write_csv(cache, rows)
  finalize_csv(
    cache, schema_name="qwen_anchor_box_cache_v3",
    primary_key=["run_id", "dataset", "subject_id", "slice_id", "count_level"],
  )
  with pytest.raises(ValueError, match="changed coordinates"):
    audit(cache, diagnostics, summary, decision, tmp_path / "audit.json")


def test_audit_rejects_crlf_evidence(tmp_path: Path):
  cache, diagnostics, summary, decision = _fixture(tmp_path)
  diagnostics.write_bytes(diagnostics.read_bytes().replace(b"\n", b"\r\n"))
  with pytest.raises(ValueError, match="canonical LF"):
    audit(cache, diagnostics, summary, decision, tmp_path / "audit.json")
