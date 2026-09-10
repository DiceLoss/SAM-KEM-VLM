from sam_kem.sweep_incremental_round2_audit import expected_round2_method_counts


def test_round2_method_counts_sum_to_reviewed_dataset_rows():
  counts = expected_round2_method_counts()
  assert sum(counts["brainweb"].values()) == 45_936
  assert sum(counts["hecktor"].values()) == 2_295
  assert sum(sum(values.values()) for values in counts.values()) == 48_231
