from pathlib import Path

from sam_kem.traceability import audit_traceability_archives


ROOT = Path(__file__).resolve().parents[1]


def test_traceability_archives_preserve_declared_arrays_and_splits():
  audit = audit_traceability_archives(ROOT)
  assert audit["locked_test_present"] is False
  assert len(audit["datasets"]["brainweb"]["cases"]) == 3
  assert len(audit["datasets"]["hecktor"]["cases"]) == 6
  assert all(
    row["mean_pool_max_abs_error"] == 0.0
    and abs(row["mean_pool_mass_ratio"] - 1.0) < 1e-7
    and row["target_outside_circular_fov_mass_fraction"] == 0.0
    for row in audit["datasets"]["brainweb"]["cases"]
  )
  assert audit["datasets"]["hecktor"]["all_subject_minus_one_mappings_valid"]


def test_traceability_audit_preserves_hecktor_square_support():
  audit = audit_traceability_archives(ROOT)
  hecktor = audit["datasets"]["hecktor"]
  assert hecktor["legacy_circle_discard_fraction_min"] > 0.15
  assert hecktor["canonical_support_policy"] == (
    "complete_square_no_content_crop_or_circular_mask"
  )
  assert not any("transaxial FOV" in item for item in audit["unresolved"])
