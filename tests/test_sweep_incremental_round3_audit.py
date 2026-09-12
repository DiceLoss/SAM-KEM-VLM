from sam_kem.sweep_incremental_round3_audit import expected_round3_method_counts


def test_round3_method_counts_sum_to_reviewed_dataset_rows():
  counts = expected_round3_method_counts()
  assert sum(counts["brainweb"].values()) == 31_320
  assert sum(counts["hecktor"].values()) == 675
  assert sum(sum(values.values()) for values in counts.values()) == 31_995
