import sys
from pathlib import Path

EXPERIMENTS = Path(__file__).resolve().parents[1] / "code" / "experiments"
sys.path.insert(0, str(EXPERIMENTS))

from measure_stage7_formal_efficiency import _objects


def test_efficiency_objects_preserve_frozen_prompt():
    value = [{"object_id": 1, "box_xyxy": [1, 2, 3, 4],
              "positive_points_xy": [[2, 3]], "negative_points_xy": [[0, 0]],
              "confidence": 0.5}]
    result = _objects(value)[0]
    assert result.box_xyxy == (1, 2, 3, 4)
    assert result.positive_points_xy == ((2, 3),)
    assert result.negative_points_xy == ((0, 0),)
