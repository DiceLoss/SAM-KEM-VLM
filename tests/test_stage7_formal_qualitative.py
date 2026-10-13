import sys
from pathlib import Path

import numpy as np

EXPERIMENTS = Path(__file__).resolve().parents[1] / "code" / "experiments"
sys.path.insert(0, str(EXPERIMENTS))

from generate_stage7_formal_qualitative import _compose, _panel, _prompt_panel


def test_qualitative_panel_layout_and_prompt_overlay():
    array = np.arange(128 * 128, dtype=float).reshape(128, 128)
    prompt = [{"box_xyxy": [10, 20, 40, 60],
               "positive_points_xy": [[20, 30]], "negative_points_xy": [[100, 100]]}]
    panels = [_panel(array, "gray"), _panel(array, "jet", jet=True),
              _prompt_panel(array, prompt, "prompt")]
    image = _compose(panels)
    assert image.size == (1024, 284)
