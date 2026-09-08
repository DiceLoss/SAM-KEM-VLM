from sam_kem.sweep_incremental_audit import expected_incremental_method_counts


def test_incremental_method_counts_sum_to_reviewed_dataset_rows():
  counts = expected_incremental_method_counts()
  assert sum(counts["brainweb"].values()) == 75_168
  assert sum(counts["hecktor"].values()) == 5_400
  assert sum(sum(values.values()) for values in counts.values()) == 80_568
