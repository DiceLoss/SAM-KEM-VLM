import pytest
from sam_kem.sweep_incremental_round6_audit import (
  _require_finite_fragment_metrics,
  expected_round6_method_counts,
)


def test_round6_method_counts_include_reviewed_empty_hecktor_domain():
  counts = expected_round6_method_counts()
  assert counts["brainweb"] == {"MR-KEM": 6_264}
  assert counts["hecktor"] == {}
  assert sum(sum(values.values()) for values in counts.values()) == 6_264


def test_round6_metric_audit_accepts_header_only_but_rejects_nonfinite(tmp_path):
  fragment = tmp_path / "results.csv"
  fragment.write_text("nrmse,ssim\n", encoding="utf-8")
  _require_finite_fragment_metrics(fragment)
  fragment.write_text("nrmse,ssim\nnan,0.9\n", encoding="utf-8")
  with pytest.raises(ValueError, match="non-finite"):
    _require_finite_fragment_metrics(fragment)
