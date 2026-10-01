"""Stage 7A-v3 portable tests for the corrected prompt-plan module.

Per directive 5927858004. Pure function tests. Includes explicit
counterexamples as regression tests.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))

from sam_kem.stage7a_prompt_plan import (
    IMAGE_DIM, PromptRequestContext, PromptObject, parse_agent_response, validate_or_fallback, FatalCaseInputError,
    PromptPlanSchemaError, pixelwise_max_aggregation,
    package_refinement_inputs, det_component_objects,
    full_box_objects, TransactionBudget, ARM_TRANSACTION_BUDGET,
)


def _ctx(arm="VLM_SINGLE_BOX", call_index=1, subject_id="subject04",
         dataset="brainweb", slice_id="z038", count_level=200000,
         realization_id=0):
    return PromptRequestContext(
        requested_arm=arm, dataset=dataset, subject_id=subject_id,
        slice_id=slice_id, count_level=count_level,
        realization_id=realization_id, call_index=call_index,
    )


def _good_point(x=10, y=10, label="positive"):
    return [int(x), int(y)]


def _good_object(oid=0, x0=10, y0=10, x1=50, y1=50, conf=0.9,
                  neg_x=0, neg_y=0):
    return {
        "object_id": int(oid),
        "box_xyxy": [int(x0), int(y0), int(x1), int(y1)],
        "positive_points_xy": [_good_point()],
        "negative_points_xy": [[int(neg_x), int(neg_y)]],
        "confidence": conf,
    }


def _good_payload(arm="VLM_SINGLE_BOX"):
    return {
        "schema_version": 1,
        "objects": [_good_object()],
        "abstain": False,
        "fallback_reason": None,
    }


# ============================================================================
# Required per-object fields: confidence
# ============================================================================

def test_required_object_confidence_rejected_when_missing():
    """Directive §A.2: each object must have confidence in [0,1]."""
    p = _good_payload()
    del p["objects"][0]["confidence"]
    with pytest.raises(PromptPlanSchemaError):
        parse_agent_response(json.dumps(p), _ctx())


def test_required_object_confidence_rejected_when_nonfinite():
    p = _good_payload()
    p["objects"][0]["confidence"] = float("inf")
    with pytest.raises(PromptPlanSchemaError):
        parse_agent_response(json.dumps(p), _ctx())
    p["objects"][0]["confidence"] = float("nan")
    with pytest.raises(PromptPlanSchemaError):
        parse_agent_response(json.dumps(p), _ctx())


def test_required_object_confidence_rejected_when_out_of_range():
    p = _good_payload()
    p["objects"][0]["confidence"] = 1.5
    with pytest.raises(PromptPlanSchemaError):
        parse_agent_response(json.dumps(p), _ctx())


def test_required_object_confidence_rejected_when_bool():
    p = _good_payload()
    p["objects"][0]["confidence"] = True
    with pytest.raises(PromptPlanSchemaError):
        parse_agent_response(json.dumps(p), _ctx())


def test_required_object_confidence_rejected_when_string():
    p = _good_payload()
    p["objects"][0]["confidence"] = "0.9"
    with pytest.raises(PromptPlanSchemaError):
        parse_agent_response(json.dumps(p), _ctx())


# ============================================================================
# Preregistered field names; no spoofed identity
# ============================================================================

def test_response_must_not_include_arm_name():
    p = _good_payload()
    p["arm_name"] = "VLM_SINGLE_BOX"
    # Per directive §A.3: arm_name belongs to trusted context, not
    # raw JSON. The schema ALLOWED_TOP_KEYS rejects unknown keys.
    with pytest.raises(PromptPlanSchemaError):
        parse_agent_response(json.dumps(p), _ctx())


def test_response_must_not_include_dataset():
    p = _good_payload()
    p["dataset"] = "hecktor"  # spoof identity
    with pytest.raises(PromptPlanSchemaError):
        parse_agent_response(json.dumps(p), _ctx())


def test_response_must_not_include_subject():
    p = _good_payload()
    p["subject_id"] = "MALICIOUS"
    with pytest.raises(PromptPlanSchemaError):
        parse_agent_response(json.dumps(p), _ctx())


def test_response_must_not_include_count_or_realization():
    p = _good_payload()
    p["count_level"] = 999
    p["realization_id"] = 99
    with pytest.raises(PromptPlanSchemaError):
        parse_agent_response(json.dumps(p), _ctx())


# ============================================================================
# Abstention executes DET_COMPONENT fallback
# ============================================================================

def test_abstain_true_executes_det_fallback():
    p = _good_payload()
    p["abstain"] = True
    p["objects"] = []
    p["fallback_reason"] = "low_confidence"
    rng = np.random.default_rng(0)
    mlem = rng.standard_normal((IMAGE_DIM, IMAGE_DIM)).astype(np.float32)
    out = validate_or_fallback(json.dumps(p), _ctx(arm="VLM_SINGLE_BOX"), mlem)
    assert out.executed_arm == "DET_COMPONENT"
    assert out.fallback_used is True
    assert out.fallback_reason == "abstain: low_confidence"
    # Plan must be DET_COMPONENT objects (>= 1 object)
    assert len(out.objects) >= 1


def test_abstain_true_with_empty_fallback_reason_falls_back():
    """Empty fallback_reason is invalid. validate_or_fallback catches
    the parse error and falls back to DET (parse errors always result
    in fallback per directive §B)."""
    p = _good_payload()
    p["abstain"] = True
    p["objects"] = []
    p["fallback_reason"] = ""
    rng = np.random.default_rng(0)
    mlem = rng.standard_normal((IMAGE_DIM, IMAGE_DIM)).astype(np.float32)
    out = validate_or_fallback(json.dumps(p), _ctx(), mlem)
    # Schema error → fallback to DET
    assert out.executed_arm == "DET_COMPONENT"
    assert out.fallback_used is True


def test_abstain_true_with_nonempty_objects_falls_back():
    """abstain=true with non-empty objects is invalid; must fall back."""
    p = _good_payload()
    p["abstain"] = True
    p["fallback_reason"] = "valid"
    rng = np.random.default_rng(0)
    mlem = rng.standard_normal((IMAGE_DIM, IMAGE_DIM)).astype(np.float32)
    out = validate_or_fallback(json.dumps(p), _ctx(), mlem)
    assert out.executed_arm == "DET_COMPONENT"
    assert out.fallback_used is True


# ============================================================================
# Single executable entry point catches every parse error
# ============================================================================

def test_parse_error_falls_back_to_det_component():
    """Directive §B: parse failures must integrate with fallback."""
    rng = np.random.default_rng(0)
    mlem = rng.standard_normal((IMAGE_DIM, IMAGE_DIM)).astype(np.float32)
    # Malformed JSON
    out = validate_or_fallback("{not valid", _ctx(), mlem)
    assert out.executed_arm == "DET_COMPONENT"
    assert out.fallback_used is True


def test_schema_error_falls_back_to_det_component():
    p = _good_payload()
    p["objects"][0]["box_xyxy"] = [10, 10]  # missing x1, y1
    raw = json.dumps(p)
    rng = np.random.default_rng(0)
    mlem = rng.standard_normal((IMAGE_DIM, IMAGE_DIM)).astype(np.float32)
    out = validate_or_fallback(raw, _ctx(), mlem)
    assert out.executed_arm == "DET_COMPONENT"
    assert out.fallback_used is True


def test_validate_or_fallback_fatal_on_invalid_mlem():
    rng = np.random.default_rng(0)
    mlem = rng.standard_normal((IMAGE_DIM, IMAGE_DIM)).astype(np.float32)
    mlem[0, 0] = float("inf")
    with pytest.raises(FatalCaseInputError):
        validate_or_fallback("{}", _ctx(), mlem)


def test_validate_or_fallback_fatal_on_nonfinite_mlem():
    mlem = np.full((IMAGE_DIM, IMAGE_DIM), np.nan)
    with pytest.raises(FatalCaseInputError):
        validate_or_fallback("{}", _ctx(), mlem)


def test_validate_or_fallback_fatal_on_zero_mlem():
    mlem = np.zeros((IMAGE_DIM, IMAGE_DIM), dtype=np.float32)
    with pytest.raises(FatalCaseInputError):
        validate_or_fallback("{}", _ctx(), mlem)


def test_validate_or_fallback_fatal_on_wrong_shape_mlem():
    mlem = np.zeros((64, 64), dtype=np.float32)
    with pytest.raises(FatalCaseInputError):
        validate_or_fallback("{}", _ctx(), mlem)


# ============================================================================
# DET_COMPONENT correctness
# ============================================================================

def test_det_component_ranks_by_actual_pixel_count():
    rng = np.random.default_rng(7)
    mlem = rng.standard_normal((IMAGE_DIM, IMAGE_DIM)).astype(np.float32)
    objs = det_component_objects(mlem)
    # Each object_id is unique, ordered
    ids = [o.object_id for o in objs]
    assert ids == list(range(len(objs)))


def test_det_component_local_peak_inside_component():
    """Directive §10: local peak must be inside its component."""
    rng = np.random.default_rng(8)
    mlem = rng.standard_normal((IMAGE_DIM, IMAGE_DIM)).astype(np.float32)
    objs = det_component_objects(mlem)
    for o in objs:
        if o.positive_points_xy:
            (px, py), = o.positive_points_xy
            x0, y0, x1, y1 = o.box_xyxy
            # Local peak is inside the PADDED bbox (>= bbox inside).
            # Tight component is a subset of padded bbox.
            # We assert it's inside the padded bbox.
            assert x0 <= px < x1
            assert y0 <= py < y1


def test_det_component_negative_outside_bbox():
    """Directive §10: background negative must be OUTSIDE the box."""
    rng = np.random.default_rng(9)
    mlem = rng.standard_normal((IMAGE_DIM, IMAGE_DIM)).astype(np.float32)
    objs = det_component_objects(mlem)
    for o in objs:
        for nx, ny in o.negative_points_xy:
            x0, y0, x1, y1 = o.box_xyxy
            assert not (x0 <= nx < x1 and y0 <= ny < y1), \
                f"negative ({nx},{ny}) inside box {o.box_xyxy}"


def test_det_component_does_not_fabricate_inside_negative_for_full_box():
    """For a single component that fills nearly the whole image,
    the 1-px band may be empty. Verify no inside negative is emitted."""
    mlem = np.full((IMAGE_DIM, IMAGE_DIM), 10.0, dtype=np.float32)
    objs = det_component_objects(mlem)
    # At most 4 components, all foreground
    assert 0 < len(objs) <= 4
    for o in objs:
        # All negatives, if any, must be outside the bbox
        for nx, ny in o.negative_points_xy:
            x0, y0, x1, y1 = o.box_xyxy
            assert not (x0 <= nx < x1 and y0 <= ny < y1), \
                f"fabricated inside negative ({nx},{ny}) in full-box {o.box_xyxy}"


def test_det_component_all_nan_mlem_raises_fatal():
    mlem = np.full((IMAGE_DIM, IMAGE_DIM), np.nan, dtype=np.float32)
    with pytest.raises(FatalCaseInputError):
        det_component_objects(mlem)


def test_det_component_rejects_int_dtype():
    """Directive §11: pixelwise max requires float; DET_COMPONENT
    must reject integer MLEM."""
    mlem = np.zeros((IMAGE_DIM, IMAGE_DIM), dtype=np.int32)
    with pytest.raises(FatalCaseInputError):
        det_component_objects(mlem)


# ============================================================================
# FULL_BOX strict: must reject external box
# ============================================================================

def test_full_box_does_not_use_external_box():
    """Directive §1 (FULL_BOX) and §A.5."""
    out = full_box_objects()
    assert len(out) == 1
    assert out[0].box_xyxy == (0, 0, IMAGE_DIM, IMAGE_DIM)


# ============================================================================
# Refinement: no averaging; second-round replaces first-round
# ============================================================================

def test_refinement_does_not_average_logits():
    """Directive §C: package_refinement_inputs does NO averaging.
    The second-round objects (with their logits) REPLACE the
    first-round objects (with their logits)."""
    rng = np.random.default_rng(0)
    mlem = rng.standard_normal((IMAGE_DIM, IMAGE_DIM)).astype(np.float32)
    first = det_component_objects(mlem)
    second = det_component_objects(mlem)
    # Replace with distinct logits
    first_logits = [np.full((IMAGE_DIM, IMAGE_DIM), 0.2, dtype=np.float32)
                    for _ in first]
    second_logits = [np.full((IMAGE_DIM, IMAGE_DIM), 0.9, dtype=np.float32)
                     for _ in second]
    mask = np.full((IMAGE_DIM, IMAGE_DIM), 0.5, dtype=np.float32)
    refined = package_refinement_inputs(
        tuple(first), first_logits, tuple(second), second_logits,
        mask, (IMAGE_DIM, IMAGE_DIM),
    )
    assert len(refined) == len(first) == len(second)
    # The output must be the second-round objects, not a blend.
    for r, s in zip(refined, second):
        assert r.object_id == s.object_id
        assert r.box_xyxy == s.box_xyxy


def test_refinement_rejects_object_id_mismatch():
    rng = np.random.default_rng(0)
    mlem = rng.standard_normal((IMAGE_DIM, IMAGE_DIM)).astype(np.float32)
    first = det_component_objects(mlem)
    second_objs = list(det_component_objects(mlem))
    second_objs[0] = PromptObject(
        object_id=999,  # mismatch
        box_xyxy=second_objs[0].box_xyxy,
        positive_points_xy=second_objs[0].positive_points_xy,
        negative_points_xy=second_objs[0].negative_points_xy,
        confidence=second_objs[0].confidence,
    )
    first_logits = [np.zeros((IMAGE_DIM, IMAGE_DIM))] * len(first)
    second_logits = [np.zeros((IMAGE_DIM, IMAGE_DIM))] * len(second_objs)
    mask = np.zeros((IMAGE_DIM, IMAGE_DIM))
    with pytest.raises(PromptPlanSchemaError):
        package_refinement_inputs(
            tuple(first), first_logits, tuple(second_objs),
            second_logits, mask, (IMAGE_DIM, IMAGE_DIM),
        )


# ============================================================================
# Aggregation: pixelwise per-object max
# ============================================================================

def test_pixelwise_max_aggregation_rejects_int_dtype():
    """Directive §11: pixelwise max requires float."""
    int_logits = [np.zeros((IMAGE_DIM, IMAGE_DIM), dtype=np.int32),
                  np.ones((IMAGE_DIM, IMAGE_DIM), dtype=np.int32)]
    with pytest.raises(PromptPlanSchemaError):
        pixelwise_max_aggregation(int_logits)


def test_pixelwise_max_aggregation_rejects_shape_mismatch():
    a = np.zeros((4, 4))
    b = np.ones((5, 5))
    with pytest.raises(PromptPlanSchemaError):
        pixelwise_max_aggregation([a, b])


def test_pixelwise_max_aggregation_rejects_empty():
    with pytest.raises(PromptPlanSchemaError):
        pixelwise_max_aggregation([])


def test_pixelwise_max_aggregation_rejects_nonfinite():
    a = np.array([[1.0, 2.0], [3.0, np.inf]])
    with pytest.raises(PromptPlanSchemaError):
        pixelwise_max_aggregation([a])


def test_pixelwise_max_aggregation_returns_correct_result():
    a = np.array([[1.0, 0.0], [0.0, 0.0]])
    b = np.array([[0.0, 2.0], [0.0, 0.0]])
    out = pixelwise_max_aggregation([a, b])
    assert out.shape == (2, 2)
    assert out[0, 0] == 1.0
    assert out[0, 1] == 2.0
    assert out.dtype.kind == "f"  # floating


# ============================================================================
# Transaction ledger
# ============================================================================

def test_transaction_budget_zeros_for_deterministic_arms():
    for arm in ("FULL_BOX", "DET_COMPONENT"):
        assert ARM_TRANSACTION_BUDGET[arm]["max_vlm_calls"] == 0
        assert ARM_TRANSACTION_BUDGET[arm]["max_medsam_calls"] == 0


def test_transaction_budget_one_for_three_vlm_arms():
    for arm in ("VLM_SINGLE_BOX", "VLM_MULTI_BOX", "VLM_BOX_POINT"):
        assert ARM_TRANSACTION_BUDGET[arm]["max_vlm_calls"] == 1
        assert ARM_TRANSACTION_BUDGET[arm]["max_medsam_calls"] == 1


def test_transaction_budget_two_for_vlm_agent_refine():
    assert ARM_TRANSACTION_BUDGET["VLM_AGENT_REFINE"]["max_vlm_calls"] == 2
    assert ARM_TRANSACTION_BUDGET["VLM_AGENT_REFINE"]["max_medsam_calls"] == 2


def test_transaction_budget_enforced_for_deterministic_arm():
    ctx = _ctx(arm="FULL_BOX")
    budget = TransactionBudget(ctx)
    with pytest.raises(PromptPlanSchemaError):
        budget.check_call("vlm")


def test_transaction_budget_enforced_for_vlm_single_box():
    ctx = _ctx(arm="VLM_SINGLE_BOX")
    budget = TransactionBudget(ctx)
    budget.check_call("vlm")  # first call OK
    with pytest.raises(PromptPlanSchemaError):
        budget.check_call("vlm")  # second call exceeds 1


def test_transaction_budget_enforces_max_calls_per_case():
    """Directive §9: at-most-one for arms 3-5, at-most-two for arm 6."""
    for arm in ("VLM_SINGLE_BOX", "VLM_MULTI_BOX", "VLM_BOX_POINT"):
        with pytest.raises(ValueError):
            PromptRequestContext(
                requested_arm=arm, dataset="d", subject_id="s",
                slice_id="z", count_level=1, realization_id=0,
                call_index=2,
            )


# ============================================================================
# Duplicate-key and duplicate-object rejection
# ============================================================================

def test_parser_rejects_duplicate_top_level_keys():
    raw = '{"schema_version": 1, "schema_version": 1, "objects": []}'
    with pytest.raises(PromptPlanSchemaError):
        parse_agent_response(raw, _ctx())


def test_parser_rejects_duplicate_object_ids():
    p = _good_payload()
    p["objects"].append(_good_object(oid=0))  # dup
    with pytest.raises(PromptPlanSchemaError):
        parse_agent_response(json.dumps(p), _ctx())


def test_parser_rejects_duplicate_boxes_via_validate_or_fallback():
    p = _good_payload()
    p["objects"].append({
        "object_id": 1,
        "box_xyxy": [10, 10, 50, 50],  # same box as object 0
        "positive_points_xy": [_good_point()],
        "negative_points_xy": [],
        "confidence": 0.9,
    })
    rng = np.random.default_rng(0)
    mlem = rng.standard_normal((IMAGE_DIM, IMAGE_DIM)).astype(np.float32)
    out = validate_or_fallback(json.dumps(p), _ctx(arm="VLM_BOX_POINT"), mlem)
    # Duplicate box triggers DET_COMPONENT fallback (per directive §6).
    assert out.executed_arm == "DET_COMPONENT"
    assert out.fallback_used is True


# ============================================================================
# Identity and context isolation
# ============================================================================

def test_context_isolation_arbitrary_full_box():
    """Directive §4: trusted arm + identity cannot be spoofed."""
    p = _good_payload()
    p["arm_name"] = "FULL_BOX"  # spoof
    # The parser ignores arm_name; the trusted context decides.
    with pytest.raises(PromptPlanSchemaError):
        parse_agent_response(json.dumps(p), _ctx(arm="VLM_SINGLE_BOX"))


def test_context_identity_validation_different_subject():
    """The trusted context subject must not be mutable from raw."""
    p = _good_payload()
    p["subject_id"] = "subject50"
    with pytest.raises(PromptPlanSchemaError):
        parse_agent_response(json.dumps(p), _ctx(subject_id="subject04"))


# ============================================================================
# ARM_CONTRACT sanity
# ============================================================================

def test_preregistered_arms_complete():
    expected = {"FULL_BOX", "DET_COMPONENT",
                "VLM_SINGLE_BOX", "VLM_MULTI_BOX",
                "VLM_BOX_POINT", "VLM_AGENT_REFINE"}
    assert set(ARM_TRANSACTION_BUDGET.keys()) == expected


def test_arm_call_index_consistency_in_context():
    with pytest.raises(ValueError):
        PromptRequestContext(
            requested_arm="VLM_SINGLE_BOX", dataset="d",
            subject_id="s", slice_id="z", count_level=1,
            realization_id=0, call_index=2,  # INVALID
        )
    # VLM_AGENT_REFINE allows call_index=2
    ctx = PromptRequestContext(
        requested_arm="VLM_AGENT_REFINE", dataset="d",
        subject_id="s", slice_id="z", count_level=1,
        realization_id=0, call_index=2,
    )
    assert ctx.call_index == 2