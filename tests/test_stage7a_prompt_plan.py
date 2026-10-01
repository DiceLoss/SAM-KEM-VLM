"""Step 7A portable tests for the prompt-plan module + helpers.

Per directive 5924162430 + 5925510360. Pure-function tests (no VLM,
no GPU, no prior generation, no KEM execution). Tests derive paths
from ``Path(__file__).resolve()`` with env-var fallback.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))

from sam_kem.stage7a_prompt_plan import (
    ARM_RULES, PREREGISTERED_ARMS, FORBIDDEN_RESPONSE_FIELDS,
    Point, Box, Detection, reject_duplicate_boxes, reject_out_of_bounds,
    reject_non_finite,
    package_per_object_prompts, max_object_continuous_logit,
    second_round_packaging, leakage_field_rejection, record_fallback,
    det_component_boxes_from_mlem10,
    parse_prompt_plan_json, PromptPlanSchemaError,
    build_det_component_plan,
    CallCounter, enforce_max_calls,
)


# ============================================================================
# Box / Point / Detection
# ============================================================================

def test_box_xyxy_strictly_xy0_lt_xy1():
    with pytest.raises(ValueError):
        Box(10, 10, 10, 20)
    with pytest.raises(ValueError):
        Box(10, 10, 5, 20)


def test_box_validate_in_bounds():
    b = Box(10, 10, 50, 50)
    b.validate_in_bounds(128)
    with pytest.raises(ValueError):
        b.validate_in_bounds(40)  # x1 out of range
    with pytest.raises(ValueError):
        Box(0, 0, 200, 10).validate_in_bounds(128)


def test_point_label_must_be_positive_or_negative():
    Point(0, 0, "positive")
    Point(0, 0, "negative")
    with pytest.raises(ValueError):
        Point(0, 0, "unknown")


def test_detection_confidence_range():
    Detection(Box(0, 0, 1, 1), 0.0)
    Detection(Box(0, 0, 1, 1), 1.0)
    with pytest.raises(ValueError):
        Detection(Box(0, 0, 1, 1), -0.01)
    with pytest.raises(ValueError):
        Detection(Box(0, 0, 1, 1), 1.01)


def test_detection_confidence_must_be_finite():
    with pytest.raises(ValueError):
        Detection(Box(0, 0, 1, 1), float("inf"))
    with pytest.raises(ValueError):
        Detection(Box(0, 0, 1, 1), float("nan"))


# ============================================================================
# Duplicate / OOB / finite rejection
# ============================================================================

def test_reject_duplicate_boxes_keeps_first_occurrence():
    boxes = [Box(0, 0, 10, 10), Box(0, 0, 10, 10), Box(10, 10, 20, 20)]
    kept = reject_duplicate_boxes(boxes)
    assert len(kept) == 2
    assert kept[0] == boxes[0]
    assert kept[1] == boxes[2]


def test_reject_out_of_bounds():
    boxes = [Box(0, 0, 50, 50), Box(100, 100, 200, 200)]
    kept = reject_out_of_bounds(boxes, image_dim=128)
    assert len(kept) == 1
    assert kept[0] == boxes[0]


def test_reject_non_finite_int_no_op():
    boxes = [Box(0, 0, 10, 10), Box(20, 20, 30, 30)]
    kept = reject_non_finite(boxes)
    assert kept == boxes


# ============================================================================
# max_object_continuous_logit
# ============================================================================

def test_max_object_continuous_logit_basic():
    boxes = [Box(0, 0, 10, 10), Box(20, 20, 30, 30), Box(40, 40, 50, 50)]
    logits = [0.1, 0.7, 0.3]
    assert max_object_continuous_logit(boxes, logits) == boxes[1]


def test_max_object_continuous_logit_empty_raises():
    with pytest.raises(ValueError):
        max_object_continuous_logit([], [])
    with pytest.raises(ValueError):
        max_object_continuous_logit([Box(0, 0, 1, 1)], [0.1, 0.2])


def test_max_object_continuous_logit_tie_keeps_first():
    boxes = [Box(0, 0, 10, 10), Box(20, 20, 30, 30)]
    logits = [0.5, 0.5]
    assert max_object_continuous_logit(boxes, logits) == boxes[0]


# ============================================================================
# second_round_packaging
# ============================================================================

def test_second_round_packaging_combines_deduplicates_and_caps():
    prev = [Box(0, 0, 10, 10), Box(0, 0, 10, 10)]  # dup pair
    new = [Box(20, 20, 30, 30), Box(150, 150, 200, 200)]  # OOB
    pos = [Point(5, 5, "positive"), Point(300, 300, "positive")]  # OOB point
    neg = [Point(0, 0, "negative")]
    boxes, p, n = second_round_packaging(
        prev_boxes=prev, prev_logits=[0.1, 0.2],
        new_boxes=new, new_points_pos=pos, new_points_neg=neg,
        image_dim=128, max_objects=4)
    assert len(boxes) == 2  # dedup'd and OOB removed
    assert all(isinstance(b, Box) for b in boxes)
    assert len(p) == 1  # OOB point removed
    assert len(n) == 1


# ============================================================================
# leakage_field_rejection
# ============================================================================

def test_leakage_field_rejection_removes_forbidden():
    r = {
        "box_xyxy": [0, 0, 1, 1],
        "confidence": 0.9,
        "reference_mask": "SECRET",
        "metric": 0.5,
    }
    out = leakage_field_rejection(r)
    assert "reference_mask" not in out
    assert "metric" not in out
    assert out["_leakage_fields_rejected"] == ["reference_mask", "metric"]
    assert out["box_xyxy"] == [0, 0, 1, 1]


def test_leakage_field_rejection_keeps_clean_response_unchanged():
    r = {"box_xyxy": [0, 0, 1, 1], "confidence": 0.5}
    out = leakage_field_rejection(r)
    assert "_leakage_fields_rejected" not in out
    assert out == r


# ============================================================================
# record_fallback
# ============================================================================

def test_record_fallback():
    fb = record_fallback("duplicate_box")
    assert fb == {"fallback_used": True, "fallback_reason": "duplicate_box",
                  "intention_to_treat": True}


# ============================================================================
# DET_COMPONENT deterministic planner
# ============================================================================

def test_det_component_deterministic_same_input_same_output():
    rng = np.random.default_rng(0)
    mlem = rng.standard_normal((128, 128)).astype(np.float32)
    a = det_component_boxes_from_mlem10(mlem, image_dim=128,
                                        threshold_quantile=0.85,
                                        padding_px=2,
                                        component_order="area_descending",
                                        max_objects=4)
    b = det_component_boxes_from_mlem10(mlem, image_dim=128,
                                        threshold_quantile=0.85,
                                        padding_px=2,
                                        component_order="area_descending",
                                        max_objects=4)
    assert a == b


def test_det_component_validates_input():
    rng = np.random.default_rng(1)
    with pytest.raises(ValueError):
        det_component_boxes_from_mlem10(rng.standard_normal((64, 64)))
    with pytest.raises(ValueError):
        det_component_boxes_from_mlem10(rng.standard_normal((128, 128)),
                                        threshold_quantile=0.0)
    with pytest.raises(ValueError):
        det_component_boxes_from_mlem10(rng.standard_normal((128, 128)),
                                        padding_px=-1)


def test_det_component_caps_at_max_objects():
    rng = np.random.default_rng(2)
    mlem = rng.standard_normal((128, 128)).astype(np.float32)
    boxes = det_component_boxes_from_mlem10(mlem, max_objects=2)
    assert len(boxes) <= 2


def test_det_component_no_silent_clipping_within_image():
    rng = np.random.default_rng(3)
    mlem = rng.standard_normal((128, 128)).astype(np.float32)
    boxes = det_component_boxes_from_mlem10(mlem, image_dim=128,
                                            padding_px=2)
    for b in boxes:
        # No box exceeds image bounds.
        b.validate_in_bounds(128)


def test_det_component_with_zero_signal():
    """With zero signal the planner must not crash. It returns
    zero or more boxes (the threshold quantile of a zero array is 0,
    so the mask is all-True, yielding one giant component)."""
    mlem = np.zeros((128, 128), dtype=np.float32)
    boxes = det_component_boxes_from_mlem10(mlem)
    assert isinstance(boxes, list)
    # No crash; returned list is well-formed (boxes are in-bounds or empty).
    for b in boxes:
        b.validate_in_bounds(128)


def test_parse_dedups_duplicate_boxes():
    """Duplicate boxes (IoU == 1.0) are dropped, not raised. The
    schema rejects duplicates by silently keeping the first occurrence."""
    p = _valid_payload()
    p["boxes"] = [[10, 10, 50, 50], [10, 10, 50, 50]]
    plan = parse_prompt_plan_json(p)
    assert len(plan.boxes) == 1
    assert plan.boxes[0] == Box(10, 10, 50, 50)


# ============================================================================
# parse_prompt_plan_json schema validation
# ============================================================================

def _valid_payload(arm="FULL_BOX"):
    return {
        "arm_name": arm,
        "dataset": "brainweb",
        "subject_id": "subject04",
        "slice_id": "z038",
        "count_level": 200000,
        "realization_id": 0,
        "boxes": [[10, 10, 50, 50]],
        "positive_points": [],
        "negative_points": [],
        "confidence": 0.9,
    }


def test_parse_valid_payload_full_box():
    plan = parse_prompt_plan_json(_valid_payload("FULL_BOX"))
    assert plan.arm_name == "FULL_BOX"
    assert len(plan.boxes) == 1


def test_parse_rejects_unknown_arm():
    p = _valid_payload()
    p["arm_name"] = "BOGUS_ARM"
    with pytest.raises(PromptPlanSchemaError):
        parse_prompt_plan_json(p)


def test_parse_rejects_missing_keys():
    p = _valid_payload()
    del p["confidence"]
    with pytest.raises(PromptPlanSchemaError):
        parse_prompt_plan_json(p)


def test_parse_rejects_too_many_boxes():
    p = _valid_payload("VLM_SINGLE_BOX")
    p["boxes"] = [[0, 0, 10, 10], [20, 20, 30, 30]]  # max 1
    with pytest.raises(PromptPlanSchemaError):
        parse_prompt_plan_json(p)


def test_parse_rejects_too_many_positive_points():
    p = _valid_payload("VLM_SINGLE_BOX")
    p["positive_points"] = [[5, 5], [10, 10]]  # max 1
    with pytest.raises(PromptPlanSchemaError):
        parse_prompt_plan_json(p)


def test_parse_rejects_zero_positive_points_for_vlm_single_box():
    p = _valid_payload("VLM_SINGLE_BOX")
    p["positive_points"] = []
    with pytest.raises(PromptPlanSchemaError):
        parse_prompt_plan_json(p)


def test_parse_rejects_out_of_bounds_box_with_schema_error():
    p = _valid_payload()
    p["boxes"] = [[0, 0, 200, 200]]  # x1 out of 128
    with pytest.raises(PromptPlanSchemaError):
        parse_prompt_plan_json(p)


def test_parse_rejects_non_strict_xy_lt_xy():
    p = _valid_payload()
    p["boxes"] = [[50, 10, 10, 50]]  # x0 > x1
    with pytest.raises(PromptPlanSchemaError):
        parse_prompt_plan_json(p)


def test_parse_rejects_invalid_box_shape():
    p = _valid_payload()
    p["boxes"] = [[1, 2, 3]]  # only 3 coords
    with pytest.raises(PromptPlanSchemaError):
        parse_prompt_plan_json(p)


def test_parse_rejects_non_finite_confidence():
    p = _valid_payload()
    p["confidence"] = "high"
    with pytest.raises(PromptPlanSchemaError):
        parse_prompt_plan_json(p)


def test_parse_rejects_confidence_out_of_range():
    p = _valid_payload()
    p["confidence"] = 1.5
    with pytest.raises(PromptPlanSchemaError):
        parse_prompt_plan_json(p)


def test_parse_vlm_box_point_requires_box_plus_points():
    p = _valid_payload("VLM_BOX_POINT")
    p["boxes"] = [[10, 10, 50, 50]]
    p["positive_points"] = [[20, 20]]
    parse_prompt_plan_json(p)  # OK
    p["boxes"] = []
    with pytest.raises(PromptPlanSchemaError):
        parse_prompt_plan_json(p)


def test_parse_rejects_non_finite_confidence_with_schema_error():
    """Non-finite confidence values (e.g. strings) must raise
    PromptPlanSchemaError, not a raw ValueError."""
    p = _valid_payload()
    p["confidence"] = "high"
    with pytest.raises(PromptPlanSchemaError):
        parse_prompt_plan_json(p)


# ============================================================================
# build_det_component_plan + leakage check
# ============================================================================

def test_build_det_component_plan_is_deterministic():
    rng = np.random.default_rng(0)
    mlem = rng.standard_normal((128, 128)).astype(np.float32)
    a = build_det_component_plan("brainweb", "subject04", "z038",
                                  200000, 0, mlem)
    b = build_det_component_plan("brainweb", "subject04", "z038",
                                  200000, 0, mlem)
    assert a == b


def test_build_det_component_plan_emits_no_leakage_fields():
    rng = np.random.default_rng(0)
    mlem = rng.standard_normal((128, 128)).astype(np.float32)
    plan = build_det_component_plan("brainweb", "subject04", "z038",
                                    200000, 0, mlem)
    items = package_per_object_prompts(plan)
    for f in FORBIDDEN_RESPONSE_FIELDS:
        for it in items:
            assert f not in it


def test_det_component_no_post_fusion_no_alpha_blend():
    # The plan is a pure box+points structure; the function does NOT
    # return any alpha/blend field. Verified by inspection.
    rng = np.random.default_rng(0)
    mlem = rng.standard_normal((128, 128)).astype(np.float32)
    plan = build_det_component_plan("brainweb", "subject04", "z038",
                                    200000, 0, mlem)
    items = package_per_object_prompts(plan)
    for it in items:
        assert "alpha" not in it
        assert "blend" not in it
        assert "fusion" not in it
        assert "post_fused" not in it


# ============================================================================
# enforce_max_calls
# ============================================================================

def test_enforce_max_calls_arm_full_box():
    arm = "FULL_BOX"
    cc = CallCounter()
    enforce_max_calls(arm, cc)  # 0 calls
    cc.incr()
    enforce_max_calls(arm, cc)  # 1 call
    cc.incr()
    with pytest.raises(PromptPlanSchemaError):
        enforce_max_calls(arm, cc)  # 2 calls > 1


def test_enforce_max_calls_arm_vlm_agent_refine_two_calls():
    arm = "VLM_AGENT_REFINE"
    cc = CallCounter()
    cc.incr()
    enforce_max_calls(arm, cc)
    cc.incr()
    enforce_max_calls(arm, cc)
    cc.incr()
    with pytest.raises(PromptPlanSchemaError):
        enforce_max_calls(arm, cc)


def test_enforce_max_calls_unknown_arm_raises():
    cc = CallCounter()
    with pytest.raises(KeyError):
        enforce_max_calls("BOGUS_ARM", cc)


# ============================================================================
# ARM_RULES sanity
# ============================================================================

def test_preregistered_arms_complete():
    assert set(PREREGISTERED_ARMS) == {"FULL_BOX", "DET_COMPONENT",
                                       "VLM_SINGLE_BOX", "VLM_MULTI_BOX",
                                       "VLM_BOX_POINT", "VLM_AGENT_REFINE"}


def test_all_arms_have_rules():
    for arm in PREREGISTERED_ARMS:
        assert arm in ARM_RULES, f"missing rules for {arm}"
        r = ARM_RULES[arm]
        assert "max_objects" in r
        assert "min_positive_points" in r
        assert "max_positive_points" in r
        assert "min_negative_points" in r
        assert "max_negative_points" in r
        assert "allow_multi_round" in r
        assert "max_calls_per_case" in r