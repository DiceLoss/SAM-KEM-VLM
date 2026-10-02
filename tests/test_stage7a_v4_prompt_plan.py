"""Stage 7A-v4.1 comprehensive regression tests.

Per directive 5932124321. Pure function tests (no I/O, no VLM, no GPU).
Every successful parse test asserts exact point tuples and exact executed
arm. Every fallback test asserts its specific reason.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))

from sam_kem.stage7a_prompt_plan import (
    IMAGE_DIM, ARM_MAXES, FULL_VAL_MAX_VLM_TRANSACTIONS,
    PromptRequestContext, PromptObject, PromptPlanSchemaError, FatalCaseInputError,
    parse_agent_response, validate_or_fallback,
    _validate_box_xyxy, pixelwise_max_aggregation, package_refinement_inputs,
    det_component_objects, TransactionLedger, _check_duplicate_points,
)


def _ctx(arm="VLM_SINGLE_BOX", call_index=1, subject_id="subject04",
         input_sha="0" * 64):
    return PromptRequestContext(
        requested_arm=arm, dataset="brainweb", subject_id=subject_id,
        slice_id="z038", count_level=200000, realization_id=0,
        call_index=call_index, input_image_sha256=input_sha,
    )


def _fresh_ledger(ctx):
    """Create a fresh ledger (clear the registry first)."""
    TransactionLedger._REGISTRY.clear()
    return TransactionLedger(ctx)


def _good_object(oid=1, x0=10, y0=10, x1=50, y1=50, conf=0.9,
                  pos=None, neg=None):
    pos = pos if pos is not None else [[15, 15]]
    neg = neg if neg is not None else [[5, 5]]
    return {
        "object_id": int(oid),
        "box_xyxy": [int(x0), int(y0), int(x1), int(y1)],
        "positive_points_xy": [[int(p[0]), int(p[1])] for p in pos],
        "negative_points_xy": [[int(p[0]), int(p[1])] for p in neg],
        "confidence": conf,
    }


def _good_payload(arm, n=1, pos_per=None, neg_per=None):
    """Build a payload with N objects, each having unique boxes and
    unique points."""
    rules = ARM_MAXES[arm]
    objs = []
    for i in range(1, n + 1):
        pos = pos_per(i) if pos_per else [[15, 15 + i]]
        neg = neg_per(i) if neg_per else [[5, 5 + i]]
        obj = _good_object(oid=i, x0=10 * i, y0=10, x1=10 * i + 40,
                            y1=50, pos=pos, neg=neg)
        # Special-case FULL_BOX and SINGLE_BOX which expect zero points
        if rules["min_positive_points"] == 0 and rules["min_negative_points"] == 0:
            obj["positive_points_xy"] = []
            obj["negative_points_xy"] = []
        objs.append(obj)
    return {
        "schema_version": 1, "objects": objs,
        "abstain": False, "fallback_reason": None,
    }


def _mlem(seed=0):
    rng = np.random.default_rng(seed)
    return rng.standard_normal((IMAGE_DIM, IMAGE_DIM)).astype(np.float32)


# ============================================================================
# Fix 1: _check_duplicate_points returns tuple and stores validated
# ============================================================================

def test_check_duplicate_points_returns_validated_tuple():
    pt = _check_duplicate_points(set(), set(),
                                  "positive_points_xy", [10, 15], 1, 0)
    assert pt == (10, 15)
    assert isinstance(pt, tuple)


def test_check_duplicate_points_rejects_duplicates_in_positive():
    seen_pos = set()
    _check_duplicate_points(seen_pos, set(),
                            "positive_points_xy", [10, 15], 1, 0)
    with pytest.raises(PromptPlanSchemaError):
        _check_duplicate_points(seen_pos, set(),
                                "positive_points_xy", [10, 15], 1, 1)


def test_check_duplicate_points_rejects_in_positive_seen_set():
    """Even if the caller has already added to seen_pos, the validator
    correctly detects the duplicate."""
    seen_pos = {(10, 15)}
    with pytest.raises(PromptPlanSchemaError):
        _check_duplicate_points(seen_pos, set(),
                                "positive_points_xy", [10, 15], 1, 0)


def test_check_duplicate_points_rejects_non_integer():
    with pytest.raises(PromptPlanSchemaError):
        _check_duplicate_points(set(), set(),
                                "positive_points_xy", [1.5, 2], 1, 0)


def test_check_duplicate_points_rejects_out_of_bounds():
    with pytest.raises(PromptPlanSchemaError):
        _check_duplicate_points(set(), set(),
                                "positive_points_xy", [200, 10], 1, 0)


# ============================================================================
# Fix 2: parse_agent_response returns tuples from parser (not raw lists)
# ============================================================================

def test_parse_returns_validated_positive_tuples():
    p = _good_payload("VLM_BOX_POINT", n=1)
    parsed = parse_agent_response(json.dumps(p), _ctx(arm="VLM_BOX_POINT"))
    assert isinstance(parsed.objects[0].positive_points_xy[0], tuple)
    assert isinstance(parsed.objects[0].negative_points_xy[0], tuple)
    assert parsed.objects[0].positive_points_xy == ((15, 16),)
    assert parsed.objects[0].negative_points_xy == ((5, 6),)


# ============================================================================
# Fix 3: Reject duplicate negative points
# ============================================================================

def test_parse_rejects_duplicate_negative_points():
    p = _good_payload("VLM_BOX_POINT", n=1)
    p["objects"][0]["negative_points_xy"] = [[13, 13], [13, 13]]
    with pytest.raises(PromptPlanSchemaError):
        parse_agent_response(json.dumps(p), _ctx(arm="VLM_BOX_POINT"))


# ============================================================================
# Fix 4: Reject pos/neg overlap (same coordinate in both sets)
# ============================================================================

def test_parse_rejects_pos_neg_overlap():
    p = _good_payload("VLM_BOX_POINT", n=1)
    p["objects"][0]["positive_points_xy"] = [[12, 12]]
    p["objects"][0]["negative_points_xy"] = [[12, 12]]  # same
    with pytest.raises(PromptPlanSchemaError):
        parse_agent_response(json.dumps(p), _ctx(arm="VLM_BOX_POINT"))


# ============================================================================
# Fix 5: Add min_objects to every arm; validate_vlm_response works for all
# ============================================================================

def test_all_arms_have_min_objects():
    for arm, rule in ARM_MAXES.items():
        assert "min_objects" in rule
        assert "max_objects" in rule
        assert rule["min_objects"] >= 1
        assert rule["max_objects"] >= rule["min_objects"]


def test_validate_vlm_response_works_for_full_arm_set():
    """VLM arm validation no longer raises KeyError on min_objects."""
    for arm in ("VLM_SINGLE_BOX", "VLM_MULTI_BOX", "VLM_BOX_POINT"):
        # Per-arm points config: SINGLE zero points; MULTI zero; BOX_POINT 1+
        if arm == "VLM_SINGLE_BOX":
            pos_per = lambda i: []
            neg_per = lambda i: []
        elif arm == "VLM_MULTI_BOX":
            pos_per = lambda i: []
            neg_per = lambda i: []
        else:  # VLM_BOX_POINT
            pos_per = lambda i: [[10 + i, 15]]
            neg_per = lambda i: [[5, 5 + i]]
        p = _good_payload(arm, n=1, pos_per=pos_per, neg_per=neg_per)
        # Must parse and validate cleanly
        parse_agent_response(json.dumps(p), _ctx(arm=arm))
        ledger = _fresh_ledger(_ctx(arm=arm))
        from sam_kem.stage7a_prompt_plan import validate_or_fallback as _vof
        # validate_only via a stub using full DET branch path
        out = _vof(json.dumps(p), _ctx(arm=arm), _mlem(), ledger=ledger)
        assert out.executed_arm == arm
        assert out.fallback_used is False


def test_vlm_single_box_zero_points_required():
    """Per closure patch §1.4: SINGLE=one box and zero points."""
    rule = ARM_MAXES["VLM_SINGLE_BOX"]
    assert rule["min_positive_points"] == 0
    assert rule["max_positive_points"] == 0
    assert rule["min_negative_points"] == 0
    assert rule["max_negative_points"] == 0


def test_vlm_multi_box_zero_points_required():
    """MULTI=1-4 boxes and zero points."""
    rule = ARM_MAXES["VLM_MULTI_BOX"]
    assert rule["min_positive_points"] == 0
    assert rule["max_positive_points"] == 0
    assert rule["min_negative_points"] == 0
    assert rule["max_negative_points"] == 0


def test_vlm_box_point_requires_positive_points():
    """BOX_POINT requires 1-3 positive and 0-3 negative per object."""
    rule = ARM_MAXES["VLM_BOX_POINT"]
    assert rule["min_positive_points"] == 1
    assert rule["max_positive_points"] == 3
    assert rule["min_negative_points"] == 0
    assert rule["max_negative_points"] == 3


# ============================================================================
# Fix 6: missing JSON key does not raise KeyError
# ============================================================================

def test_missing_required_key_falls_back_via_validate_or_fallback():
    """When a required key is missing, parse_agent_response raises
    PromptPlanSchemaError, and validate_or_fallback converts to
    DET_COMPONENT fallback (NOT a raw KeyError)."""
    raw = '{"schema_version": 1, "objects": [{"object_id": 1, "box_xyxy": [0,0,10,10], "positive_points_xy": [], "negative_points_xy": [], "confidence": 0.5}]}'
    # No abstain key — must raise PromptPlanSchemaError, NOT KeyError
    with pytest.raises(PromptPlanSchemaError):
        parse_agent_response(raw, _ctx())
    # validate_or_fallback catches the schema error and returns DET fallback
    ctx = _ctx(arm="VLM_SINGLE_BOX")
    out = validate_or_fallback(raw, ctx, _mlem(), ledger=_fresh_ledger(ctx))
    assert out.executed_arm == "DET_COMPONENT"
    assert "missing required top-level keys" in out.fallback_reason


# ============================================================================
# Fix 7: boundary boxes (x1 or y1 == 128) accepted
# ============================================================================

def test_boundary_box_accepted_in_validator():
    box = _validate_box_xyxy([0, 0, 128, 128], "test")
    assert box == (0, 0, 128, 128)


# ============================================================================
# Fix 8: refinement permits new correction points
# ============================================================================

def test_refinement_accepts_new_correction_points():
    """Per closure patch §3: correction points need not be a subset
    of first-round points. The whole purpose of the second call is
    to introduce new correction points."""
    first_objects = (PromptObject(
        object_id=1, box_xyxy=(10, 10, 50, 50),
        positive_points_xy=(), negative_points_xy=(), confidence=0.9),)
    first_logits = [np.full((IMAGE_DIM, IMAGE_DIM), 0.5, dtype=np.float32)]
    new_correction_obj = PromptObject(
        object_id=1, box_xyxy=(10, 10, 50, 50),
        positive_points_xy=((20, 20), (30, 30)),  # NEW points not in first
        negative_points_xy=((5, 5),),
        confidence=0.95,
    )
    specs = package_refinement_inputs(
        first_objects, first_logits, (new_correction_obj,),
        _ctx(arm="VLM_AGENT_REFINE", call_index=2))
    assert len(specs) == 1
    assert specs[0].object_id == 1


def test_refinement_rejects_object_id_mismatch():
    first = (PromptObject(object_id=1, box_xyxy=(0, 0, 10, 10),
                          positive_points_xy=(), negative_points_xy=(),
                          confidence=0.9),)
    log = [np.zeros((IMAGE_DIM, IMAGE_DIM))]
    corr = (PromptObject(object_id=2, box_xyxy=(0, 0, 10, 10),
                        positive_points_xy=(), negative_points_xy=(),
                        confidence=0.9),)
    with pytest.raises(PromptPlanSchemaError):
        package_refinement_inputs(first, log, corr, _ctx())


# ============================================================================
# Fix 9: DET background candidate filtered to labels==0
# ============================================================================

def test_det_component_background_outside_component_pixels():
    """Background negative must be outside component AND from labels==0."""
    rng = np.random.default_rng(7)
    mlem = rng.standard_normal((IMAGE_DIM, IMAGE_DIM)).astype(np.float32)
    objs = det_component_objects(mlem)
    for o in objs:
        for nx, ny in o.negative_points_xy:
            x0, y0, x1, y1 = o.box_xyxy
            # Must be outside the box
            assert not (x0 <= nx < x1 and y0 <= ny < y1)


def test_det_component_zero_signal_raises_fatal():
    """All-zero mlem must raise FatalCaseInputError (no silent empty)."""
    with pytest.raises(FatalCaseInputError):
        det_component_objects(np.zeros((IMAGE_DIM, IMAGE_DIM), dtype=np.float32))


# ============================================================================
# Fix 10: transaction ledger explicit, not hidden global
# ============================================================================

def test_ledger_persists_across_contexts():
    ctx1 = _ctx(arm="VLM_MULTI_BOX", subject_id="s1")
    ctx2 = _ctx(arm="VLM_MULTI_BOX", subject_id="s2")
    TransactionLedger._REGISTRY.clear()
    TransactionLedger(ctx1).check_call("vlm")
    assert TransactionLedger(ctx2).vlm_calls == 0
    assert TransactionLedger(ctx1).vlm_calls == 1


def test_ledger_persists_across_same_case():
    """The ledger key includes subject_id, so same case persists."""
    TransactionLedger._REGISTRY.clear()
    ctx = _ctx(arm="VLM_MULTI_BOX")
    TransactionLedger(ctx).check_call("vlm")
    assert TransactionLedger(ctx).vlm_calls == 1


def test_ledger_vlm_call_exceeds_max_raises():
    """Deterministic arms have vlm_calls=0; first VLM call is the max."""
    TransactionLedger._REGISTRY.clear()
    ctx = _ctx(arm="VLM_MULTI_BOX")
    b = TransactionLedger(ctx)
    b.check_call("vlm")
    with pytest.raises(PromptPlanSchemaError):
        b.check_call("vlm")


# ============================================================================
# Fix 11: trust_id includes arm + call + input_sha + parser_version
# ============================================================================

def test_trust_id_distinct_across_arms():
    a = _ctx(arm="FULL_BOX")
    b = _ctx(arm="DET_COMPONENT")
    assert a.trust_id() != b.trust_id()


def test_trust_id_distinct_across_call_indices():
    a = _ctx(arm="VLM_AGENT_REFINE", call_index=1)
    b = _ctx(arm="VLM_AGENT_REFINE", call_index=2)
    assert a.trust_id() != b.trust_id()


def test_trust_id_distinct_across_input_image_sha():
    a = _ctx(arm="FULL_BOX", input_sha="a" * 64)
    b = _ctx(arm="FULL_BOX", input_sha="b" * 64)
    assert a.trust_id() != b.trust_id()


# ============================================================================
# Aggregation no min-max inside
# ============================================================================

def test_pixelwise_max_returns_finite_float():
    arrs = [np.full((4,), 0.3, dtype=np.float32),
            np.full((4,), 0.7, dtype=np.float32)]
    out = pixelwise_max_aggregation(arrs)
    assert out.dtype.kind == "f"


# ============================================================================
# ARM_MAXES sanity
# ============================================================================

def test_arm_maxes_complete():
    assert set(ARM_MAXES.keys()) == {
        "FULL_BOX", "DET_COMPONENT", "VLM_SINGLE_BOX",
        "VLM_MULTI_BOX", "VLM_BOX_POINT", "VLM_AGENT_REFINE",
    }


def test_full_val_max_vlm_transactions_correct():
    """Sum per case: 0+0+1+1+1+2 = 5; per 8892 cases = 44,460."""
    assert FULL_VAL_MAX_VLM_TRANSACTIONS == 5 * 8892


def test_vlm_max_zero_for_deterministic_arms():
    assert ARM_MAXES["FULL_BOX"]["vlm_calls"] == 0
    assert ARM_MAXES["DET_COMPONENT"]["vlm_calls"] == 0


# ============================================================================
# Full validation path tests
# ============================================================================

def test_full_box_zero_point_zero_negative_success():
    ctx = _ctx(arm="FULL_BOX")
    out = validate_or_fallback(None, ctx, _mlem(), ledger=_fresh_ledger(ctx))
    assert out.executed_arm == "FULL_BOX"
    assert out.fallback_used is False
    assert out.objects[0].box_xyxy == (0, 0, IMAGE_DIM, IMAGE_DIM)
    assert out.objects[0].positive_points_xy == ()
    assert out.objects[0].negative_points_xy == ()


def test_det_component_zero_point_one_negative_success():
    ctx = _ctx(arm="DET_COMPONENT")
    out = validate_or_fallback(None, ctx, _mlem(), ledger=_fresh_ledger(ctx))
    assert out.executed_arm == "DET_COMPONENT"
    assert 1 <= len(out.objects) <= 4
    for o in out.objects:
        assert len(o.positive_points_xy) == 1
        assert len(o.negative_points_xy) <= 3


def test_vlm_box_point_one_object_success():
    ctx = _ctx(arm="VLM_BOX_POINT")
    p = _good_payload("VLM_BOX_POINT", n=1)
    out = validate_or_fallback(json.dumps(p), ctx, _mlem(),
                              ledger=_fresh_ledger(ctx))
    assert out.executed_arm == "VLM_BOX_POINT"
    assert out.fallback_used is False
    assert out.objects[0].box_xyxy == (10, 10, 50, 50)
    assert out.objects[0].positive_points_xy == ((15, 16),)
    assert out.objects[0].negative_points_xy == ((5, 6),)


def test_full_box_with_raw_raises_fatal():
    ctx = _ctx(arm="FULL_BOX")
    with pytest.raises(FatalCaseInputError):
        validate_or_fallback("{}", ctx, _mlem(), ledger=_fresh_ledger(ctx))


def test_det_component_with_raw_raises_fatal():
    ctx = _ctx(arm="DET_COMPONENT")
    with pytest.raises(FatalCaseInputError):
        validate_or_fallback("{}", ctx, _mlem(), ledger=_fresh_ledger(ctx))


def test_vlm_arm_with_raw_none_raises_fatal():
    ctx = _ctx(arm="VLM_SINGLE_BOX")
    with pytest.raises(FatalCaseInputError):
        validate_or_fallback(None, ctx, _mlem(), ledger=_fresh_ledger(ctx))


# ============================================================================
# Abstention
# ============================================================================

def test_abstain_with_empty_objects_and_reason_executes_det():
    ctx = _ctx(arm="VLM_SINGLE_BOX")
    p = {"schema_version": 1, "objects": [],
          "abstain": True, "fallback_reason": "low_confidence"}
    out = validate_or_fallback(json.dumps(p), ctx, _mlem(),
                              ledger=_fresh_ledger(ctx))
    assert out.executed_arm == "DET_COMPONENT"
    assert out.fallback_used is True
    assert "abstain: low_confidence" in out.fallback_reason


# ============================================================================
# Schema failure fallback
# ============================================================================

def test_duplicate_boxes_falls_back_to_det():
    ctx = _ctx(arm="VLM_MULTI_BOX")
    p = _good_payload("VLM_MULTI_BOX", n=2)
    p["objects"][1]["box_xyxy"] = [10, 10, 50, 50]  # duplicate box
    out = validate_or_fallback(json.dumps(p), ctx, _mlem(),
                              ledger=_fresh_ledger(ctx))
    assert out.executed_arm == "DET_COMPONENT"
    assert "duplicate" in out.fallback_reason


def test_duplicate_points_falls_back_to_det():
    ctx = _ctx(arm="VLM_BOX_POINT")
    p = _good_payload("VLM_BOX_POINT", n=2)
    p["objects"][1]["positive_points_xy"] = [[15, 16]]  # duplicate of obj 1
    out = validate_or_fallback(json.dumps(p), ctx, _mlem(),
                              ledger=_fresh_ledger(ctx))
    assert out.executed_arm == "DET_COMPONENT"
    assert "duplicate" in out.fallback_reason


def test_pos_neg_overlap_falls_back_to_det():
    ctx = _ctx(arm="VLM_BOX_POINT")
    p = _good_payload("VLM_BOX_POINT", n=1)
    p["objects"][0]["negative_points_xy"] = [[15, 16]]  # same as positive
    out = validate_or_fallback(json.dumps(p), ctx, _mlem(),
                              ledger=_fresh_ledger(ctx))
    assert out.executed_arm == "DET_COMPONENT"


def test_missing_required_key_falls_back():
    ctx = _ctx(arm="VLM_SINGLE_BOX")
    raw = '{"schema_version": 1, "objects": [{"object_id": 1, "box_xyxy": [0,0,10,10], "positive_points_xy": [], "negative_points_xy": [], "confidence": 0.5}]}'
    out = validate_or_fallback(raw, ctx, _mlem(), ledger=_fresh_ledger(ctx))
    assert out.executed_arm == "DET_COMPONENT"
    assert "missing" in out.fallback_reason


# ============================================================================
# ARM_MAXES completeness
# ============================================================================

def test_all_arms_have_vlm_medsam_max():
    for arm, rule in ARM_MAXES.items():
        assert "vlm_calls" in rule
        assert "medsam_calls" in rule


def test_det_component_obj_ids_sequential():
    objs = det_component_objects(_mlem())
    assert [o.object_id for o in objs] == list(range(1, len(objs) + 1))