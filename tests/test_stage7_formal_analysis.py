import numpy as np
import pytest
from sam_kem.stage7_formal_analysis import (
    analyse_subject_cells,
    holm_adjust,
    paired_case_deltas,
    sign_flip_pvalue,
    subject_deltas,
)


def test_pairing_and_subject_aggregation():
    rows = []
    for role, nrmse, ssim in (("policy", 0.2, 0.9), ("paired_comparator", 0.3, 0.8)):
        rows.append({
            "dataset": "brainweb", "subject_id": "s1", "slice_id": "z1",
            "count_level": 100000, "realization_id": 0, "role": role,
            "nrmse": nrmse, "ssim": ssim, "fallback_used": "false",
        })
    cases = paired_case_deltas(rows)
    assert cases[0]["delta_nrmse"] == pytest.approx(-0.1)
    assert cases[0]["delta_ssim"] == pytest.approx(0.1)
    subjects = subject_deltas(cases)
    assert subjects[0]["case_count"] == 1


def test_exact_sign_flip_and_holm():
    pvalue, mode = sign_flip_pvalue(
        np.array([1.0, 1.0]), exact_limit=20,
        monte_carlo_samples=100, seed=1,
    )
    assert mode == "exact"
    assert pvalue == 0.5
    assert holm_adjust([0.01, 0.04, 0.03]) == [0.03, 0.06, 0.06]


def test_five_cell_analysis_is_deterministic():
    rows = []
    for dataset, counts in (("brainweb", (100000,)),
                            ("hecktor", (100000, 200000, 500000, 1000000))):
        for count in counts:
            for subject, delta in (("a", -0.02), ("b", -0.01)):
                rows.append({
                    "dataset": dataset, "count_level": count,
                    "subject_id": subject, "mean_delta_nrmse": delta,
                    "mean_delta_ssim": -delta, "case_count": 1,
                    "fallback_case_count": 0,
                })
    first = analyse_subject_cells(
        rows, bootstrap_samples=100, permutation_samples=100, seed=9,
    )
    second = analyse_subject_cells(
        rows, bootstrap_samples=100, permutation_samples=100, seed=9,
    )
    assert first == second
    assert len(first) == 10
