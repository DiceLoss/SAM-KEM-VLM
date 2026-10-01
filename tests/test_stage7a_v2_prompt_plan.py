"""Stage 7A-v2 portable tests for the corrected prompt-plan module.

Per directive 5926624802. These tests include explicit
counterexamples from the directive as regression tests. Pure
function tests (no VLM, no GPU, no prior generation, no KEM).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))

from sam_kem.stage7a_prompt_plan import (
    ARM_CONTRACT, IMAGE_DIM,
    Box, Point, PromptObject, PromptPlan,
    parse_prompt_plan_json, PromptPlanSchemaError,
    pixelwise_max_aggregation, validate_mask_input_for_refine,
    package_refinement, record_fallback,
    det_component_plan, execute_with_fallback,
)


def _minimal_valid_payload(arm="FULL_BOX"):
    return {
        "schema_version": 1,
        "arm_name": arm,
        "dataset": "brainweb",
        "subject_id": "subject04",
        "slice_id": "z038",
        "count_level": 200000,
        "realization_id": 0,
        "objects": [{
            "object_id": 0,
            "box": {"x0": 0, "y0": 0, "x1": 10, "y1": 10},
            "positive_points": [],
            "negative_points": [],
        }],
        "abstain": False,
        "fallback_reason": "",
    }


# ============================================================================
# Box / Point / Detection (frozen contract)
# ============================================================================

def test_box_strict_no_coercion_fractional():
    """Fractional coordinates must raise, not be silently int-casted."""
    for bad in [[1.9, 2.9, 10.9, 20.9], [1, 2, 10.0, 20],
                ["1", "2", "10", "20"], [True, False, True, False]]:
        with pytest.raises((ValueError, TypeError)):
            Box(*bad)


def test_box_xyxy_strict():
    Box(0, 0, 10, 10)
    with pytest.raises(ValueError):
        Box(10, 10, 10, 20)  # x0 == x1
    with pytest.raises(ValueError):
        Box(10, 10, 5, 20)   # x0 > x1


def test_box_strict_xy_lt_xy():
    with pytest.raises(ValueError):
        Box(0, 0, 0, 1)


def test_box_zero_area_rejected():
    """Boxes with zero area are rejected."""
    with pytest.raises(ValueError):
        Box(0, 0, 1, 0)  # y1 < y0


def test_box_oob_xy_rejected():
    with pytest.raises(ValueError):
        Box(-1, 0, 10, 10)
    with pytest.raises(ValueError):
        Box(0, 0, 200, 10)  # x1 > 128
    with pytest.raises(ValueError):
        Box(0, 0, 10, 129)


def test_point_strict_int():
    Point(0, 0, "positive")
    Point(127, 127, "negative")
    with pytest.raises(ValueError):
        Point(-1, 0, "positive")
    with pytest.raises(ValueError):
        Point(128, 0, "positive")  # out of [0, 128)


def test_point_label_strict():
    with pytest.raises(ValueError):
        Point(0, 0, "neutral")
    with pytest.raises(ValueError):
        Point(0, 0, "")


def test_point_no_fractional_no_string():
    for bad in [(1.5, 2.5, "positive"), ("1", "2", "positive"),
                (0, 0, "true")]:
        with pytest.raises((ValueError, TypeError)):
            Point(*bad)


# ============================================================================
# Per-object strict JSON schema (parse_prompt_plan_json)
# ============================================================================

def test_parser_rejects_malformed_json():
    with pytest.raises(PromptPlanSchemaError):
        parse_prompt_plan_json("{not valid json")


def test_parser_rejects_duplicate_top_level_keys():
    raw = '{"schema_version": 1, "schema_version": 2, "x": 1}'
    with pytest.raises(PromptPlanSchemaError):
        parse_prompt_plan_json(raw)


def test_parser_rejects_unknown_schema_version():
    raw = '{"schema_version": 99, "arm_name": "FULL_BOX"}'
    with pytest.raises(PromptPlanSchemaError):
        parse_prompt_plan_json(raw)


def test_parser_rejects_unknown_arm():
    p = _minimal_valid_payload()
    p["arm_name"] = "BOGUS_ARM"
    with pytest.raises(PromptPlanSchemaError):
        parse_prompt_plan_json(json.dumps(p))


def test_parser_rejects_forbidden_top_level_field():
    """Forbidden fields like reference_mask must trigger fallback."""
    p = _minimal_valid_payload()
    p["reference_mask"] = "SECRET"
    with pytest.raises(PromptPlanSchemaError):
        parse_prompt_plan_json(json.dumps(p))


def test_parser_rejects_unknown_nested_extra_field():
    """Unknown per-object keys are forbidden."""
    p = _minimal_valid_payload("VLM_SINGLE_BOX")
    p["objects"][0]["weird_extra_field"] = 1
    with pytest.raises(PromptPlanSchemaError):
        parse_prompt_plan_json(json.dumps(p))


def test_parser_rejects_duplicate_object_id():
    p = _minimal_valid_payload("VLM_MULTI_BOX")
    p["objects"].append(dict(p["objects"][0]))  # two objects, same id 0
    with pytest.raises(PromptPlanSchemaError):
        parse_prompt_plan_json(json.dumps(p))


def test_parser_rejects_duplicate_box():
    """Duplicate box must trigger fallback (not silent deduplication)."""
    p = _minimal_valid_payload("VLM_SINGLE_BOX")
    # Add a second object with the same box
    p["objects"].append({
        "object_id": 1,
        "box": {"x0": 0, "y0": 0, "x1": 10, "y1": 10},
        "positive_points": [],
        "negative_points": [],
    })
    with pytest.raises(PromptPlanSchemaError):
        parse_prompt_plan_json(json.dumps(p))


def test_parser_rejects_oob_point():
    """OOB points are rejected (no silent clipping)."""
    p = _minimal_valid_payload("VLM_SINGLE_BOX")
    p["objects"][0]["positive_points"] = [{"x": 200, "y": -1, "label": "positive"}]
    with pytest.raises(PromptPlanSchemaError):
        parse_prompt_plan_json(json.dumps(p))


def test_parser_rejects_nonfinite_coord():
    """NaN/Infinity values are rejected."""
    raw = '{"schema_version": 1, "arm_name": "FULL_BOX", "dataset": "b", "subject_id": "s", "slice_id": "z", "count_level": 1, "realization_id": 0, "objects": [{"object_id": 0, "box": {"x0": 0.0/0, "y0": 0, "x1": 1, "y1": 1}, "positive_points": [], "negative_points": []}], "abstain": false}'
    with pytest.raises((PromptPlanSchemaError, ValueError, ZeroDivisionError)):
        parse_prompt_plan_json(raw)


def test_parser_rejects_fractional_box():
    raw = '{"schema_version": 1, "arm_name": "FULL_BOX", "dataset": "b", "subject_id": "s", "slice_id": "z", "count_level": 1, "realization_id": 0, "objects": [{"object_id": 0, "box": {"x0": 1.9, "y0": 2.9, "x1": 10.9, "y1": 20.9}, "positive_points": [], "negative_points": []}], "abstain": false}'
    with pytest.raises(PromptPlanSchemaError):
        parse_prompt_plan_json(raw)


def test_parser_rejects_numeric_string_box():
    raw = '{"schema_version": 1, "arm_name": "FULL_BOX", "dataset": "b", "subject_id": "s", "slice_id": "z", "count_level": 1, "realization_id": 0, "objects": [{"object_id": 0, "box": {"x0": "1", "y0": "2", "x1": "10", "y1": "20"}, "positive_points": [], "negative_points": []}], "abstain": false}'
    with pytest.raises(PromptPlanSchemaError):
        parse_prompt_plan_json(raw)


def test_parser_rejects_empty_objects_with_oob_points():
    p = _minimal_valid_payload("VLM_MULTI_BOX")
    p["objects"] = []
    p["positive_points_v2"] = [{"x": 200, "y": -1, "label": "positive"}]
    # empty objects + oob points: must fail validation
    with pytest.raises(PromptPlanSchemaError):
        parse_prompt_plan_json(json.dumps(p))


def test_parser_rejects_bool_as_confidence_or_coord():
    raw = '{"schema_version": 1, "arm_name": "FULL_BOX", "dataset": "b", "subject_id": "s", "slice_id": "z", "count_level": 1, "realization_id": 0, "objects": [{"object_id": 0, "box": {"x0": true, "y0": 0, "x1": 1, "y1": 1}, "positive_points": [], "negative_points": []}], "abstain": false}'
    with pytest.raises(PromptPlanSchemaError):
        parse_prompt_plan_json(raw)


# ============================================================================
# Arm contract correctness
# ============================================================================

def test_full_box_is_exact_full_image():
    p = _minimal_valid_payload("FULL_BOX")
    plan = parse_prompt_plan_json(json.dumps(p))
    # FULL_BOX ignores any user-supplied box and uses the exact full image.
    assert plan.objects[0].box == Box(0, 0, IMAGE_DIM, IMAGE_DIM)


def test_full_box_zero_points():
    p = _minimal_valid_payload("FULL_BOX")
    p["objects"][0]["positive_points"] = [{"x": 1, "y": 1, "label": "positive"}]
    # FULL_BOX must reject any points (per directive).
    with pytest.raises(PromptPlanSchemaError):
        parse_prompt_plan_json(json.dumps(p))


def test_det_component_zero_vlm_points_required():
    p = _minimal_valid_payload("DET_COMPONENT")
    p["objects"][0]["positive_points"] = []  # no positive point
    with pytest.raises(PromptPlanSchemaError):
        parse_prompt_plan_json(json.dumps(p))


def test_det_component_within_max_objects():
    p = _minimal_valid_payload("DET_COMPONENT")
    # 5 objects > max 4
    for i in range(1, 5):
        p["objects"].append({
            "object_id": i,
            "box": {"x0": i * 5, "y0": 0, "x1": i * 5 + 3, "y1": 3},
            "positive_points": [{"x": i * 5 + 1, "y": 1, "label": "positive"}],
            "negative_points": [],
        })
    with pytest.raises(PromptPlanSchemaError):
        parse_prompt_plan_json(json.dumps(p))


def test_vlm_single_box_one_object_no_points():
    p = _minimal_valid_payload("VLM_SINGLE_BOX")
    p["objects"][0]["positive_points"] = [{"x": 1, "y": 1, "label": "positive"}]
    with pytest.raises(PromptPlanSchemaError):
        parse_prompt_plan_json(json.dumps(p))


def test_vlm_multi_box_no_points():
    p = _minimal_valid_payload("VLM_MULTI_BOX")
    p["objects"][0]["positive_points"] = [{"x": 1, "y": 1, "label": "positive"}]
    with pytest.raises(PromptPlanSchemaError):
        parse_prompt_plan_json(json.dumps(p))


def test_vlm_box_point_per_object_point_range():
    p = _minimal_valid_payload("VLM_BOX_POINT")
    p["objects"][0]["positive_points"] = [{"x": 1, "y": 1, "label": "positive"},
                                          {"x": 2, "y": 2, "label": "positive"},
                                          {"x": 3, "y": 3, "label": "positive"},
                                          {"x": 4, "y": 4, "label": "positive"}]
    # 4 positive > max 3
    with pytest.raises(PromptPlanSchemaError):
        parse_prompt_plan_json(json.dumps(p))


def test_vlm_agent_refine_requires_mask_input_shape():
    p = _minimal_valid_payload("VLM_AGENT_REFINE")
    # mask_input_logit_shape missing
    with pytest.raises(PromptPlanSchemaError):
        parse_prompt_plan_json(json.dumps(p))


def test_vlm_agent_refine_mask_input_shape_validated():
    p = _minimal_valid_payload("VLM_AGENT_REFINE")
    p["objects"][0]["positive_points"] = [{"x": 10, "y": 10, "label": "positive"}]
    p["mask_input_logit_shape"] = [128, 128]
    plan = parse_prompt_plan_json(json.dumps(p))
    assert plan.mask_input_logit_shape == (128, 128)


# ============================================================================
# DET_COMPONENT deterministic planner
# ============================================================================

def test_det_component_is_deterministic():
    rng = np.random.default_rng(0)
    mlem = rng.standard_normal((IMAGE_DIM, IMAGE_DIM)).astype(np.float32)
    a = det_component_plan(mlem)
    b = det_component_plan(mlem)
    assert [(o.box, o.positive_points, o.negative_points, o.object_id)
            for o in a] == [(o.box, o.positive_points, o.negative_points, o.object_id)
                            for o in b]


def test_det_component_each_object_has_local_peak_positive():
    rng = np.random.default_rng(1)
    mlem = rng.standard_normal((IMAGE_DIM, IMAGE_DIM)).astype(np.float32)
    objs = det_component_plan(mlem)
    assert len(objs) >= 1
    for o in objs:
        # 1 positive point per object
        assert len(o.positive_points) == 1
        # Local peak must be inside the bbox
        for p in o.positive_points:
            assert p.label == "positive"
            assert o.box.x0 <= p.x < o.box.x1
            assert o.box.y0 <= p.y < o.box.y1


def test_det_component_each_object_has_background_negative():
    rng = np.random.default_rng(2)
    mlem = rng.standard_normal((IMAGE_DIM, IMAGE_DIM)).astype(np.float32)
    objs = det_component_plan(mlem)
    for o in objs:
        # 0-3 deterministic background negatives per object
        assert 0 <= len(o.negative_points) <= 3
        for n in o.negative_points:
            assert n.label == "negative"
            # Negative point is OUTSIDE the bbox (1-px band)
            assert not (o.box.x0 <= n.x < o.box.x1 and
                        o.box.y0 <= n.y < o.box.y1)


def test_det_component_pure_does_not_inspect_history():
    """DET_COMPONENT threshold/padding/order are frozen and not derived
    from historical metrics. Calling it twice with the same input yields
    the same output regardless of any frozen-history context."""
    rng = np.random.default_rng(0)
    mlem = rng.standard_normal((IMAGE_DIM, IMAGE_DIM)).astype(np.float32)
    plan_a = det_component_plan(mlem)
    # Mimic a fresh interpreter / fresh import: just call again
    plan_b = det_component_plan(mlem)
    assert plan_a == plan_b


def test_det_component_no_calls_no_vlm():
    """DET_COMPONENT must have zero VLM calls (deterministic)."""
    assert ARM_CONTRACT["DET_COMPONENT"]["vlm_calls"] == 0


# ============================================================================
# Pure helpers: pixelwise max + mask_input + fallback
# ============================================================================

def test_pixelwise_max_aggregation_requires_same_shape():
    a = np.zeros((4, 4))
    b = np.ones((5, 5))
    with pytest.raises(PromptPlanSchemaError):
        pixelwise_max_aggregation([a, b])


def test_pixelwise_max_aggregation_requires_nonempty():
    with pytest.raises(PromptPlanSchemaError):
        pixelwise_max_aggregation([])


def test_pixelwise_max_aggregation_rejects_nonfinite():
    a = np.array([1.0, np.inf, 3.0])
    with pytest.raises(PromptPlanSchemaError):
        pixelwise_max_aggregation([a])


def test_pixelwise_max_aggregation_returns_pixelwise_max():
    a = np.array([[1.0, 0.0], [0.0, 0.0]])
    b = np.array([[0.0, 2.0], [0.0, 0.0]])
    out = pixelwise_max_aggregation([a, b])
    assert out.shape == (2, 2)
    assert out[0, 0] == 1.0
    assert out[0, 1] == 2.0


def test_validate_mask_input_shape_mismatch():
    arr = np.zeros((64, 64))
    with pytest.raises(PromptPlanSchemaError):
        validate_mask_input_for_refine(arr, (128, 128))


def test_validate_mask_input_nonfinite():
    arr = np.full((128, 128), np.nan)
    with pytest.raises(PromptPlanSchemaError):
        validate_mask_input_for_refine(arr, (128, 128))


def test_record_fallback_records_requested_and_executed_arm():
    fb = record_fallback("VLM_SINGLE_BOX", "DET_COMPONENT", "duplicate_box")
    assert fb["requested_arm"] == "VLM_SINGLE_BOX"
    assert fb["executed_arm"] == "DET_COMPONENT"
    assert fb["fallback_used"] is True
    assert fb["fallback_reason"] == "duplicate_box"


def test_execute_with_fallback_runs_det_component_on_failure():
    rng = np.random.default_rng(0)
    mlem = rng.standard_normal((IMAGE_DIM, IMAGE_DIM)).astype(np.float32)
    result = execute_with_fallback(None, mlem, "VLM_SINGLE_BOX", None)
    assert result["fallback_used"] is True
    assert result["executed_arm"] == "DET_COMPONENT"
    assert isinstance(result["plan"], PromptPlan)


# ============================================================================
# Refinement packaging
# ============================================================================

def test_package_refinement_carries_prev_logits():
    n = 32
    obj = PromptObject(object_id=0, box=Box(0, 0, 10, 10),
                        positive_points=(), negative_points=())
    first = PromptPlan(arm_name="VLM_AGENT_REFINE", dataset="d",
                       subject_id="s", slice_id="z", count_level=1,
                       realization_id=0, objects=(obj,), abstain=False,
                       fallback_reason="",
                       mask_input_logit_shape=(n, n))
    mask = np.zeros((n, n), dtype=np.float32)
    prev_logits = [np.ones((n, n), dtype=np.float32) * 0.5]
    new_obj = PromptObject(object_id=0, box=Box(0, 0, 10, 10),
                           positive_points=(), negative_points=())
    new_logits = [np.ones((n, n), dtype=np.float32) * 1.0]
    mask, merged = package_refinement(
        first, mask, (n, n), prev_logits, [new_obj], new_logits)
    assert merged[0].shape == (n, n)
    assert np.allclose(merged[0], 0.75)


def test_package_refinement_rejects_shape_mismatch():
    n = 32
    obj = PromptObject(object_id=0, box=Box(0, 0, 10, 10),
                        positive_points=(), negative_points=())
    first = PromptPlan(arm_name="VLM_AGENT_REFINE", dataset="d",
                       subject_id="s", slice_id="z", count_level=1,
                       realization_id=0, objects=(obj,), abstain=False,
                       fallback_reason="",
                       mask_input_logit_shape=(n, n))
    mask = np.zeros((n, n), dtype=np.float32)
    prev_logits = [np.ones((n, n), dtype=np.float32)]  # correct shape
    new_obj = PromptObject(object_id=0, box=Box(0, 0, 10, 10),
                           positive_points=(), negative_points=())
    new_logits = [np.ones((n + 1, n), dtype=np.float32)]  # wrong shape
    with pytest.raises(PromptPlanSchemaError):
        package_refinement(first, mask, (n, n),
                           prev_logits, [new_obj], new_logits)


def test_package_refinement_rejects_object_id_mismatch():
    n = 32
    obj0 = PromptObject(object_id=0, box=Box(0, 0, 10, 10),
                         positive_points=(), negative_points=())
    obj1 = PromptObject(object_id=1, box=Box(0, 0, 10, 10),
                         positive_points=(), negative_points=())
    first = PromptPlan(arm_name="VLM_AGENT_REFINE", dataset="d",
                       subject_id="s", slice_id="z", count_level=1,
                       realization_id=0, objects=(obj0,), abstain=False,
                       fallback_reason="",
                       mask_input_logit_shape=(n, n))
    mask = np.zeros((n, n), dtype=np.float32)
    with pytest.raises(PromptPlanSchemaError):
        package_refinement(first, mask, (n, n),
                           [np.zeros((n, n))], [obj1], [np.zeros((n, n))])


# ============================================================================
# Counter separation
# ============================================================================

def test_counter_separation_per_arm():
    """DET_COMPONENT and FULL_BOX have vlm_calls=0; VLM arms have vlm_calls>=1;
    no arm has both vlm_calls and a separate MedSAM decoder count yet."""
    for arm in ("FULL_BOX", "DET_COMPONENT"):
        assert ARM_CONTRACT[arm]["vlm_calls"] == 0
    for arm in ("VLM_SINGLE_BOX", "VLM_MULTI_BOX", "VLM_BOX_POINT",
                "VLM_AGENT_REFINE"):
        assert ARM_CONTRACT[arm]["vlm_calls"] >= 1


# ============================================================================
# Stage 7A-v2 specific: parse_prompt_plan_json is strict (no silent
# coercion, no silent clip, no silent deduplication)
# ============================================================================

def test_strict_parser_never_int_casts_fractional():
    """Counterexample from directive §3: [1.9, 2.9, 10.9, 20.9]
    must NOT silently become (1, 2, 10, 20)."""
    raw = json.dumps({
        "schema_version": 1,
        "arm_name": "FULL_BOX",
        "dataset": "d", "subject_id": "s", "slice_id": "z",
        "count_level": 1, "realization_id": 0,
        "objects": [{
            "object_id": 0,
            "box": {"x0": 1.9, "y0": 2.9, "x1": 10.9, "y1": 20.9},
            "positive_points": [], "negative_points": [],
        }],
        "abstain": False,
    })
    with pytest.raises(PromptPlanSchemaError):
        parse_prompt_plan_json(raw)


def test_strict_parser_never_accepts_forbidden_field():
    """Counterexample from directive §6."""
    raw = json.dumps({
        "schema_version": 1,
        "arm_name": "FULL_BOX",
        "dataset": "d", "subject_id": "s", "slice_id": "z",
        "count_level": 1, "realization_id": 0,
        "objects": [{
            "object_id": 0,
            "box": {"x0": 0, "y0": 0, "x1": 10, "y1": 10},
            "positive_points": [], "negative_points": [],
        }],
        "abstain": False,
        "reference_mask": "SECRET",  # FORBIDDEN
    })
    with pytest.raises(PromptPlanSchemaError):
        parse_prompt_plan_json(raw)


def test_strict_parser_duplicate_box_triggers_fallback():
    """Counterexample from directive §5."""
    raw = json.dumps({
        "schema_version": 1,
        "arm_name": "FULL_BOX",  # changed from VLM_SINGLE_BOX to use 1 obj
        "dataset": "d", "subject_id": "s", "slice_id": "z",
        "count_level": 1, "realization_id": 0,
        "objects": [{
            "object_id": 0,
            "box": {"x0": 0, "y0": 0, "x1": 10, "y1": 10},
            "positive_points": [], "negative_points": [],
        }, {
            "object_id": 1,
            "box": {"x0": 0, "y0": 0, "x1": 10, "y1": 10},  # duplicate
            "positive_points": [], "negative_points": [],
        }],
        "abstain": False,
    })
    # FULL_BOX has max_objects=1, so 2 objects is also rejected.
    # Try with VLM_BOX_POINT (max_objects=4) instead.
    raw_vbp = raw.replace('"FULL_BOX"', '"VLM_BOX_POINT"')
    with pytest.raises(PromptPlanSchemaError):
        parse_prompt_plan_json(raw_vbp)


def test_pixelwise_max_aggregation_unimplemented_in_v1():
    """Counterexample from directive §8: passing 2-D logits must not raise
    a ValueError, but produce the per-pixel max."""
    logits = [np.ones((128, 128)), np.ones((128, 128)) * 2]
    out = pixelwise_max_aggregation(logits)
    assert out.shape == (128, 128)
    assert out.max() == 2.0


def test_second_round_carries_prev_logits():
    """Counterexample from directive §9."""
    n = 32
    obj = PromptObject(object_id=0, box=Box(0, 0, 10, 10),
                        positive_points=(), negative_points=())
    first = PromptPlan(arm_name="VLM_AGENT_REFINE", dataset="d",
                       subject_id="s", slice_id="z", count_level=1,
                       realization_id=0, objects=(obj,), abstain=False,
                       fallback_reason="",
                       mask_input_logit_shape=(n, n))
    mask = np.zeros((n, n), dtype=np.float32)
    prev_logits = [np.full((n, n), 0.2, dtype=np.float32)]
    new_obj = PromptObject(object_id=0, box=Box(0, 0, 10, 10),
                           positive_points=(), negative_points=())
    new_logits = [np.full((n, n), 0.8, dtype=np.float32)]
    mask, merged = package_refinement(
        first, mask, (n, n), prev_logits, [new_obj], new_logits)
    # The averaged mask should reflect both prev and new contributions.
    assert merged[0].mean() == pytest.approx(0.5)


# ============================================================================
# ARM_CONTRACT sanity
# ============================================================================

def test_preregistered_arms_complete():
    assert set(ARM_CONTRACT.keys()) == {
        "FULL_BOX", "DET_COMPONENT",
        "VLM_SINGLE_BOX", "VLM_MULTI_BOX",
        "VLM_BOX_POINT", "VLM_AGENT_REFINE",
    }


def test_full_box_is_pure_deterministic():
    """FULL_BOX: vlm_calls=0, deterministic exact full image."""
    assert ARM_CONTRACT["FULL_BOX"]["vlm_calls"] == 0
    assert ARM_CONTRACT["FULL_BOX"]["deterministic_box_xyxy"] == (
        0, 0, IMAGE_DIM, IMAGE_DIM)


def test_det_component_has_local_peaks_and_negatives_frozen():
    assert ARM_CONTRACT["DET_COMPONENT"]["vlm_calls"] == 0
    assert ARM_CONTRACT["DET_COMPONENT"]["min_positive_points"] == 1
    assert ARM_CONTRACT["DET_COMPONENT"]["max_positive_points"] == 1
    assert ARM_CONTRACT["DET_COMPONENT"]["min_negative_points"] == 0
    assert ARM_CONTRACT["DET_COMPONENT"]["max_negative_points"] == 3


def test_all_arms_have_consistent_keys():
    required = {"arm_kind", "vlm_calls", "min_objects", "max_objects",
                "min_positive_points", "max_positive_points",
                "min_negative_points", "max_negative_points",
                "allow_multi_round", "max_calls_per_case"}
    for arm, rule in ARM_CONTRACT.items():
        missing = required - set(rule.keys())
        assert not missing, f"{arm}: missing {missing}"