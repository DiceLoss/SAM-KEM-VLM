import json

import numpy as np
import pytest
from sam_kem.stage7_formal_runtime import (
    FormalPromptError,
    aggregate_logits,
    parse_formal_prompt,
    render_pet_display,
    render_pet_views,
)


def response(objects, *, abstain=False, reason=""):
    return json.dumps({
        "schema_version": 4, "objects": objects,
        "abstain": abstain, "fallback_reason": reason,
    })


def obj(i, box, pos, neg=()):
    return {"object_id": i, "box_xyxy": list(box),
            "positive_points_xy": [list(x) for x in pos],
            "negative_points_xy": [list(x) for x in neg],
            "confidence": 0.75}


def test_brainweb_contract_accepts_three_local_objects():
    objects = [obj(i, (i, i, i + 10, i + 10), ((i + 1, i + 1),)) for i in range(1, 4)]
    assert len(parse_formal_prompt(response(objects), dataset="brainweb")) == 3


def test_hecktor_contract_accepts_background_negative():
    objects = [obj(1, (10, 10, 20, 20), ((15, 15),), ((3, 3),))]
    assert len(parse_formal_prompt(response(objects), dataset="hecktor")) == 1


@pytest.mark.parametrize("raw", [
    '{"schema_version":4,"schema_version":4,"objects":[],"abstain":false,"fallback_reason":""}',
    response([], abstain=True, reason="uncertain"),
    response([obj(1, (0, 0, 129, 2), ((1, 1),))] * 3),
])
def test_malformed_response_is_rejected(raw):
    with pytest.raises(FormalPromptError):
        parse_formal_prompt(raw, dataset="brainweb")


def test_hecktor_negative_must_be_outside_box():
    raw = response([obj(1, (10, 10, 20, 20), ((15, 15),), ((11, 11),))])
    with pytest.raises(FormalPromptError):
        parse_formal_prompt(raw, dataset="hecktor")


def test_pixelwise_max_aggregation():
    first = np.array([[1, 4], [2, 0]], dtype=np.float32)
    second = np.array([[3, 2], [1, 5]], dtype=np.float32)
    np.testing.assert_array_equal(aggregate_logits((first, second)), [[3, 4], [2, 5]])


def test_render_is_deterministic_and_view_sensitive():
    image = np.arange(128 * 128, dtype=np.float32).reshape(128, 128)
    first, first_sha = render_pet_views(image)
    second, second_sha = render_pet_views(image)
    assert first == second
    assert first_sha == second_sha
    assert first.startswith(b"\x89PNG")
    assert render_pet_display(image, view_mode="grayscale_only")[1] != first_sha
    assert render_pet_display(image, view_mode="jet_only")[1] != first_sha
    with pytest.raises(ValueError):
        render_pet_display(image, view_mode="unknown")
