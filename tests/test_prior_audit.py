import numpy as np

from sam_kem.prior_audit import summarize_values


def test_prior_audit_summary_records_distribution_quantiles():
  summary = summarize_values([0.0, 1.0, 2.0, 3.0])
  assert summary["n"] == 4
  assert summary["mean"] == 1.5
  assert summary["quantiles"]["0"] == 0.0
  assert summary["quantiles"]["50"] == 1.5
  assert summary["quantiles"]["100"] == 3.0


def test_prior_audit_summary_handles_empty_input():
  assert summarize_values([]) == {"n": 0}
  assert np.isfinite(summarize_values([1.0])["mean"])
