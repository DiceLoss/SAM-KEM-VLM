import sys
from pathlib import Path

EXPERIMENTS = Path(__file__).resolve().parents[1] / "code" / "experiments"
sys.path.insert(0, str(EXPERIMENTS))

from run_stage7_formal_prompt_ablation import _objects


def test_prompt_factor_removal_preserves_boxes_and_identity():
    source = [{
        "object_id": 1, "box_xyxy": [1, 2, 30, 40],
        "positive_points_xy": [[10, 11]],
        "negative_points_xy": [[0, 0], [100, 100]], "confidence": 0.8,
    }]
    box_only = _objects(source, positive=False, negative=False)[0]
    positive = _objects(source, positive=True, negative=False)[0]
    full = _objects(source, positive=True, negative=True)[0]
    assert box_only.box_xyxy == positive.box_xyxy == full.box_xyxy
    assert box_only.positive_points_xy == ()
    assert positive.positive_points_xy == ((10, 11),)
    assert positive.negative_points_xy == ()
    assert full.negative_points_xy == ((0, 0), (100, 100))
