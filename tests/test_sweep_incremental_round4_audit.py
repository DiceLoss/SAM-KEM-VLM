from sam_kem.sweep_incremental_round4_audit import expected_round4_method_counts


def test_round4_method_counts_sum_to_reviewed_dataset_rows():
  counts = expected_round4_method_counts()
  assert sum(counts["brainweb"].values()) == 18_792
  assert sum(counts["hecktor"].values()) == 135
  assert sum(sum(values.values()) for values in counts.values()) == 18_927
