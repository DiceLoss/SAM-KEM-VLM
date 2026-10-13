import sys
from pathlib import Path

EXPERIMENTS = Path(__file__).resolve().parents[1] / "code" / "experiments"
sys.path.insert(0, str(EXPERIMENTS))

from analyze_stage7_formal_secondary import box_iou, prompt_case_contrasts


def test_box_iou_uses_half_open_boxes():
    assert box_iou([0, 0, 10, 10], [0, 0, 10, 10]) == 1
    assert box_iou([0, 0, 10, 10], [10, 0, 20, 10]) == 0


def test_prompt_contrasts_exclude_source_fallback_and_preserve_component_order():
    identity = {"dataset": "hecktor", "subject_id": "1", "slice_id": "z1",
                "count_level": "100000", "realization_id": "0"}
    main = [
        {**identity, "role": "policy", "fallback_used": "False",
         "nrmse": "4", "ssim": "8"},
        {**identity, "role": "paired_comparator", "fallback_used": "False",
         "nrmse": "7", "ssim": "5"},
    ]
    ablation = [
        {**identity, "arm": "lesion_box_only", "nrmse": "6", "ssim": "6"},
        {**identity, "arm": "lesion_box_positive", "nrmse": "5", "ssim": "7"},
    ]
    rows = prompt_case_contrasts(main, ablation)
    assert [row["contrast"] for row in rows] == [
        "box_only_vs_det", "positive_added", "negative_added"]
    assert [row["delta_nrmse"] for row in rows] == [-1, -1, -1]
