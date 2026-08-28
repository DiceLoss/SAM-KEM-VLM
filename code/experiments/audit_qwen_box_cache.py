"""Fail-closed audit of a Qwen box cache against the frozen semantic diagnostics."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

from sam_kem.artifacts import atomic_write_json, file_sha256, verify_csv_artifact

BOX_FIELDS = ("xmin", "ymin", "xmax", "ymax")


def _require_canonical_lf(path: Path) -> None:
  payload = path.read_bytes()
  if b"\r" in payload or (payload and not payload.endswith(b"\n")):
    raise ValueError(f"text evidence must use canonical LF and end with LF: {path}")


def _portable_path(path: Path) -> str:
  resolved = path.resolve()
  for candidate in resolved.parents:
    if (candidate / "pyproject.toml").is_file() and (candidate / "code" / "sam_kem").is_dir():
      return resolved.relative_to(candidate).as_posix()
  return resolved.as_posix()


def _key(row: dict[str, str]) -> tuple[str, str, str, int]:
  return row["dataset"], row["subject_id"], row["slice_id"], int(row["count_level"])


def _read_unique(path: Path) -> dict[tuple[str, str, str, int], dict[str, str]]:
  with path.open("r", encoding="utf-8", newline="") as handle:
    rows = list(csv.DictReader(handle))
  keyed = {_key(row): row for row in rows}
  if len(keyed) != len(rows):
    raise ValueError(f"duplicate dataset/subject/slice/count keys: {path}")
  return keyed


def audit(
  cache_path: Path,
  diagnostics_path: Path,
  semantic_summary_path: Path,
  decision_path: Path,
  output_path: Path,
) -> dict:
  for path in (
    cache_path, cache_path.with_suffix(cache_path.suffix + ".manifest.json"),
    diagnostics_path, semantic_summary_path, decision_path,
  ):
    _require_canonical_lf(path)
  cache_manifest = verify_csv_artifact(cache_path)
  if cache_manifest.get("schema_name") != "qwen_anchor_box_cache_v3":
    raise ValueError("expected qwen_anchor_box_cache_v3 cache")
  cache = _read_unique(cache_path)
  diagnostics = _read_unique(diagnostics_path)
  if cache.keys() != diagnostics.keys():
    missing = sorted(diagnostics.keys() - cache.keys())
    unknown = sorted(cache.keys() - diagnostics.keys())
    raise ValueError(f"cache-key mismatch: missing={missing[:5]}, unknown={unknown[:5]}")

  summary = json.loads(semantic_summary_path.read_text(encoding="utf-8"))
  if summary.get("diagnostics_csv_sha256") != file_sha256(diagnostics_path):
    raise ValueError("semantic summary does not bind the diagnostics CSV")
  decision = json.loads(decision_path.read_text(encoding="utf-8"))
  if (
    decision.get("decision", {}).get("status")
    != "approved_to_build_development_box_cache_without_post_hoc_box_changes"
    or decision.get("evidence", {}).get("semantic_summary_sha256")
    != file_sha256(semantic_summary_path)
    or decision.get("evidence", {}).get("diagnostics_csv_sha256_lf")
    != file_sha256(diagnostics_path)
  ):
    raise ValueError("semantic decision is missing, unapproved, or bound to other evidence")
  expected_failures = {
    (
      row["dataset"], row["subject_id"], row["slice_id"], int(row["count_level"]),
    )
    for row in summary.get("failure_records", [])
  }

  exact_qwen_boxes = 0
  fallback_keys = []
  policy_pairs = Counter()
  source_counts = Counter()
  for key, cached in cache.items():
    diagnostic = diagnostics[key]
    coordinates = tuple(int(cached[field]) for field in BOX_FIELDS)
    xmin, ymin, xmax, ymax = coordinates
    if not (0 <= xmin < xmax <= 128 and 0 <= ymin < ymax <= 128):
      raise ValueError(f"invalid cached box for {key}: {coordinates}")
    source_counts[cached["source"]] += 1
    policy_pairs[(cached["request_policy_id"], cached["request_policy_sha256"])] += 1
    if diagnostic["source"] == "qwen3-vl-plus":
      if cached["source"] != "qwen3-vl-plus":
        raise ValueError(f"successful Qwen box changed source for {key}")
      expected_box = tuple(int(diagnostic[field]) for field in BOX_FIELDS)
      if coordinates != expected_box:
        raise ValueError(f"successful Qwen box changed coordinates for {key}")
      if (
        cached["request_id"] != diagnostic["request_id"]
        or cached["image_sha256"] != diagnostic["image_sha256"]
      ):
        raise ValueError(f"successful Qwen provenance changed for {key}")
      exact_qwen_boxes += 1
    else:
      if key not in expected_failures:
        raise ValueError(f"unexpected fallback key: {key}")
      if cached["source"] != "centroid_fallback_qwen_request_failure":
        raise ValueError(f"wrong fallback source for {key}: {cached['source']}")
      fallback_keys.append({
        "dataset": key[0], "subject_id": key[1], "slice_id": key[2],
        "count_level": key[3], "box_xyxy": list(coordinates),
      })

  observed_fallback_keys = {
    (row["dataset"], row["subject_id"], row["slice_id"], row["count_level"])
    for row in fallback_keys
  }
  if observed_fallback_keys != expected_failures:
    raise ValueError("fallback keys differ from the frozen terminal-failure set")
  if len(policy_pairs) != 1:
    raise ValueError(f"cache contains multiple request policies: {dict(policy_pairs)}")
  if source_counts.get("qwen_batch_configuration_failure", 0):
    raise ValueError("batch-configuration failure leaked into box cache")

  policy_id, policy_sha256 = next(iter(policy_pairs))
  result = {
    "schema_version": 1,
    "artifact_kind": "qwen_box_cache_audit",
    "status": "complete",
    "text_binding": "canonical_lf",
    "cache_path": _portable_path(cache_path),
    "cache_sha256": file_sha256(cache_path),
    "cache_manifest_sha256": file_sha256(
      cache_path.with_suffix(cache_path.suffix + ".manifest.json")
    ),
    "diagnostics_path": _portable_path(diagnostics_path),
    "diagnostics_sha256": file_sha256(diagnostics_path),
    "semantic_summary_path": _portable_path(semantic_summary_path),
    "semantic_summary_sha256": file_sha256(semantic_summary_path),
    "semantic_decision_path": _portable_path(decision_path),
    "semantic_decision_sha256": file_sha256(decision_path),
    "records": len(cache),
    "unique_keys": len(cache),
    "exact_qwen_boxes": exact_qwen_boxes,
    "centroid_fallback_boxes": len(fallback_keys),
    "source_counts": dict(sorted(source_counts.items())),
    "request_policy_id": policy_id,
    "request_policy_sha256": policy_sha256,
    "fallback_records": fallback_keys,
  }
  atomic_write_json(output_path, result)
  return result


def main() -> int:
  parser = argparse.ArgumentParser()
  parser.add_argument("--cache", type=Path, required=True)
  parser.add_argument("--diagnostics", type=Path, required=True)
  parser.add_argument("--semantic-summary", type=Path, required=True)
  parser.add_argument("--decision", type=Path, required=True)
  parser.add_argument("--output", type=Path, required=True)
  args = parser.parse_args()
  result = audit(
    args.cache.resolve(), args.diagnostics.resolve(),
    args.semantic_summary.resolve(), args.decision.resolve(), args.output.resolve(),
  )
  print(json.dumps(result, indent=2, sort_keys=True))
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
