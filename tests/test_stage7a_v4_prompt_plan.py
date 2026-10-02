"""Stage 7A-v4.2 comprehensive regression tests.

Per directive 5945647996. Pure function tests (no I/O, no VLM, no GPU).
Tests cover:
- parser strictness (strict bool abstain, cross-object pos/neg overlap both
  directions, malformed raw → schema fallback)
- context validation (call_index range, SHA-256 format, dataset/arm)
- explicit LedgerStore (no class-global, atomic consume, resumable
  serialization)
- round-1/round-2 continuity through explicit serialization
- fallback accounting (VLM counted on malformed/abstained, MedSAM counted
  by actual decoder objects including fallback)
- per-arm MedSAM maximum enforcement
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
    FULL_VAL_MAX_MEDSAM_PER_CASE, ALLOWED_DATASETS,
    PromptRequestContext, PromptPlanSchemaError, FatalCaseInputError,
    parse_agent_response, validate_or_fallback,
    pixelwise_max_aggregation, det_component_objects, LedgerStore, LedgerKey, TransactionLedger,
)


def _ctx(arm="VLM_SINGLE_BOX", call_index=1, subject_id="subject04",
         input_sha="0" * 64, dataset="brainweb", count_level=200000,
         realization_id=0):
    return PromptRequestContext(
        requested_arm=arm, dataset=dataset, subject_id=subject_id,
        slice_id="z038", count_level=count_level, realization_id=realization_id,
        call_index=call_index, input_image_sha256=input_sha,
    )


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


def _fresh_store():
    return LedgerStore()


def _key_for(ctx):
    return LedgerKey(
        dataset=ctx.dataset, subject_id=ctx.subject_id, slice_id=ctx.slice_id,
        count_level=ctx.count_level, realization_id=ctx.realization_id,
        requested_arm=ctx.requested_arm,
    )


# ============================================================================
# Fix 1: strict bool abstain (no 0/1/null/string coercion)
# ============================================================================

@pytest.mark.parametrize("bad_abst", [0, 1, None, "false", "true", [], 0.0, 1.0])
def test_strict_bool_abstain_rejects_non_bool(bad_abst):
    raw = {
        "schema_version": 1, "objects": [],
        "abstain": bad_abst, "fallback_reason": "x",
    }
    with pytest.raises(PromptPlanSchemaError):
        parse_agent_response(json.dumps(raw), _ctx())


def test_strict_bool_abstain_true_accepted():
    raw = {
        "schema_version": 1, "objects": [],
        "abstain": True, "fallback_reason": "low_confidence",
    }
    parsed = parse_agent_response(json.dumps(raw), _ctx())
    assert parsed.abstain is True


def test_strict_bool_abstain_false_accepted():
    p = _good_payload("VLM_SINGLE_BOX", n=1)
    parsed = parse_agent_response(json.dumps(p), _ctx())
    assert parsed.abstain is False


# ============================================================================
# Fix 2: cross-object pos/neg overlap both orderings
# ============================================================================

def test_cross_object_pos_to_neg_overlap_rejected():
    """obj1.pos=[(2,2)]; obj2.neg=[(2,2)] → REJECT."""
    p = _good_payload("VLM_BOX_POINT", n=2)
    p["objects"][0]["positive_points_xy"] = [[2, 2]]
    p["objects"][0]["negative_points_xy"] = []
    p["objects"][1]["positive_points_xy"] = [[10, 10]]
    p["objects"][1]["negative_points_xy"] = [[2, 2]]
    with pytest.raises(PromptPlanSchemaError) as ei:
        parse_agent_response(json.dumps(p), _ctx())
    assert "duplicates a positive point" in str(ei.value).lower() or "negative point" in str(ei.value).lower()


def test_cross_object_neg_to_pos_overlap_rejected():
    """obj1.neg=[(2,2)]; obj2.pos=[(2,2)] → REJECT."""
    p = _good_payload("VLM_BOX_POINT", n=2)
    p["objects"][0]["positive_points_xy"] = [[10, 10]]
    p["objects"][0]["negative_points_xy"] = [[2, 2]]
    p["objects"][1]["positive_points_xy"] = [[2, 2]]
    p["objects"][1]["negative_points_xy"] = []
    with pytest.raises(PromptPlanSchemaError):
        parse_agent_response(json.dumps(p), _ctx())


def test_cross_object_pos_to_pos_duplicate_rejected():
    p = _good_payload("VLM_BOX_POINT", n=2)
    p["objects"][1]["positive_points_xy"] = [[15, 16]]  # same as obj 1
    with pytest.raises(PromptPlanSchemaError):
        parse_agent_response(json.dumps(p), _ctx())


def test_cross_object_neg_to_neg_duplicate_rejected():
    p = _good_payload("VLM_BOX_POINT", n=2)
    p["objects"][1]["negative_points_xy"] = [[5, 6]]
    with pytest.raises(PromptPlanSchemaError):
        parse_agent_response(json.dumps(p), _ctx())


# ============================================================================
# Fix 3: context validation (call_index, SHA, dataset/arm)
# ============================================================================

def test_context_rejects_unsupported_dataset():
    with pytest.raises(ValueError):
        PromptRequestContext(
            requested_arm="FULL_BOX", dataset="x", subject_id="s",
            slice_id="z", count_level=200000, realization_id=0,
            call_index=1, input_image_sha256="0" * 64,
        )


def test_context_rejects_unsupported_arm():
    with pytest.raises(ValueError):
        PromptRequestContext(
            requested_arm="FOO", dataset="brainweb", subject_id="s",
            slice_id="z", count_level=200000, realization_id=0,
            call_index=1, input_image_sha256="0" * 64,
        )


@pytest.mark.parametrize("bad_call", [0, -1, 2, 3, True, False, "1", 1.5])
def test_context_rejects_bad_call_index(bad_call):
    with pytest.raises(ValueError):
        PromptRequestContext(
            requested_arm="FULL_BOX", dataset="brainweb",
            subject_id="s", slice_id="z",
            count_level=200000, realization_id=0,
            call_index=bad_call, input_image_sha256="0" * 64,
        )


def test_context_rejects_malformed_sha():
    with pytest.raises(ValueError):
        PromptRequestContext(
            requested_arm="FULL_BOX", dataset="brainweb",
            subject_id="s", slice_id="z",
            count_level=200000, realization_id=0,
            call_index=1, input_image_sha256="X" * 64,
        )


def test_context_rejects_short_sha():
    with pytest.raises(ValueError):
        PromptRequestContext(
            requested_arm="FULL_BOX", dataset="brainweb",
            subject_id="s", slice_id="z",
            count_level=200000, realization_id=0,
            call_index=1, input_image_sha256="0" * 63,
        )


def test_context_rejects_uppercase_sha():
    with pytest.raises(ValueError):
        PromptRequestContext(
            requested_arm="FULL_BOX", dataset="brainweb",
            subject_id="s", slice_id="z",
            count_level=200000, realization_id=0,
            call_index=1, input_image_sha256="A" * 64,
        )


def test_context_rejects_bool_count_level():
    with pytest.raises(ValueError):
        PromptRequestContext(
            requested_arm="FULL_BOX", dataset="brainweb",
            subject_id="s", slice_id="z",
            count_level=True, realization_id=0,
            call_index=1, input_image_sha256="0" * 64,
        )


def test_context_rejects_bool_realization_id():
    with pytest.raises(ValueError):
        PromptRequestContext(
            requested_arm="FULL_BOX", dataset="brainweb",
            subject_id="s", slice_id="z",
            count_level=200000, realization_id=False,
            call_index=1, input_image_sha256="0" * 64,
        )


def test_context_accepts_call_index_2_for_refine():
    ctx = PromptRequestContext(
        requested_arm="VLM_AGENT_REFINE", dataset="brainweb",
        subject_id="s", slice_id="z",
        count_level=200000, realization_id=0,
        call_index=2, input_image_sha256="0" * 64,
    )
    assert ctx.call_index == 2


# ============================================================================
# Fix 4: ledger explicit + atomic + resumable serialization
# ============================================================================

def test_ledger_no_class_global_state():
    """There must be no class/global mutable registry."""
    # No TransactionLedger._REGISTRY attribute exists any more.
    store1 = LedgerStore()
    store2 = LedgerStore()
    ctx = _ctx()
    TransactionLedger(store1, _key_for(ctx)).consume_vlm()
    # store2 should be empty.
    assert store2.get(_key_for(ctx)) is None
    assert store1.get(_key_for(ctx)).vlm_count == 1


def test_ledger_state_isolation_per_case():
    """Two cases with different keys are isolated."""
    store = LedgerStore()
    ctx_a = _ctx(subject_id="a")
    ctx_b = _ctx(subject_id="b")
    TransactionLedger(store, _key_for(ctx_a)).consume_vlm()
    assert store.get(_key_for(ctx_a)).vlm_count == 1
    assert store.get(_key_for(ctx_b)) is None


def test_ledger_atomic_over_budget_rejection():
    """VLM_AGENT_REFINE vlm_calls=2; third consume must raise."""
    store = LedgerStore()
    ctx = _ctx(arm="VLM_AGENT_REFINE")
    ledger = TransactionLedger(store, _key_for(ctx))
    ledger.consume_vlm()
    ledger.consume_vlm()
    with pytest.raises(PromptPlanSchemaError):
        ledger.consume_vlm()


def test_ledger_medsam_atomic_over_budget_rejection():
    """VLM_MULTI_BOX medsam=4; consume 5 must raise."""
    store = LedgerStore()
    ctx = _ctx(arm="VLM_MULTI_BOX")
    ledger = TransactionLedger(store, _key_for(ctx))
    ledger.consume_medsam_objects(4)
    with pytest.raises(PromptPlanSchemaError):
        ledger.consume_medsam_objects(1)


def test_ledger_medsam_atomic_validates_total_before_mutation():
    """If total exceeds, state must NOT be mutated."""
    store = LedgerStore()
    # DET_COMPONENT has medsam=4 (same as fallback max), so budget = 4.
    ctx = _ctx(arm="DET_COMPONENT")
    ledger = TransactionLedger(store, _key_for(ctx))
    with pytest.raises(PromptPlanSchemaError):
        ledger.consume_medsam_objects(5)
    # State must remain at 0 (no partial mutation).
    assert store.get(_key_for(ctx)).medsam_count == 0


def test_ledger_serialization_round_trip():
    store_a = LedgerStore()
    ctx = _ctx()
    ledger_a = TransactionLedger(store_a, _key_for(ctx))
    ledger_a.consume_vlm()
    ledger_a.consume_medsam_objects(2)
    ledger_a.record.fallback_reasons.append("dummy")
    snap = store_a.to_dict()
    store_b = LedgerStore.from_dict(snap)
    rec_b = store_b.get(_key_for(ctx))
    assert rec_b.vlm_count == 1
    assert rec_b.medsam_count == 2
    assert rec_b.fallback_reasons == ["dummy"]


def test_ledger_round_1_round_2_continuity():
    """VLM_AGENT_REFINE: round 1 and round 2 share the same record via
    explicit serialization."""
    store = LedgerStore()
    ctx_r1 = _ctx(arm="VLM_AGENT_REFINE", call_index=1)
    ctx_r2 = _ctx(arm="VLM_AGENT_REFINE", call_index=2)
    # Same case identity, different call_index (event field, not key field).
    p_r1 = _good_payload("VLM_AGENT_REFINE", n=2)
    p_r2 = _good_payload("VLM_AGENT_REFINE", n=2)
    out1 = validate_or_fallback(json.dumps(p_r1), ctx_r1, _mlem(),
                                 store=store)
    validate_or_fallback(json.dumps(p_r2), ctx_r2, _mlem(),
                        store=store,
                        first_round_objects=out1.objects,
                        first_round_logits=[
                            np.zeros((4, 4), dtype=np.float32)
                            for _ in out1.objects])
    rec = store.get(_key_for(ctx_r1))
    assert rec.vlm_count == 2
    assert rec.medsam_count == 4
    assert len(rec.outcomes) == 2


# ============================================================================
# Fix 5: fallback accounting (VLM + MedSAM counted on every branch)
# ============================================================================

def test_malformed_raw_falls_back_and_counts_vlm():
    """Malformed raw → schema fallback; VLM was consumed (raw counted),
    MedSAM by DET fallback objects (1–4 objects)."""
    store = _fresh_store()
    ctx = _ctx(arm="VLM_BOX_POINT")
    out = validate_or_fallback("{not json", ctx, _mlem(), store=store)
    assert out.executed_arm == "DET_COMPONENT"
    assert out.fallback_used is True
    assert "malformed json" in out.fallback_reason.lower()
    rec = store.get(_key_for(ctx))
    assert rec.vlm_count == 1  # VLM was consumed (raw counted)
    assert 1 <= rec.medsam_count <= 4  # DET MedSAM by objects


def test_abstain_falls_back_and_counts_vlm():
    store = _fresh_store()
    ctx = _ctx(arm="VLM_BOX_POINT")
    raw = {"schema_version": 1, "objects": [], "abstain": True,
           "fallback_reason": "low_conf"}
    out = validate_or_fallback(json.dumps(raw), ctx, _mlem(), store=store)
    assert out.executed_arm == "DET_COMPONENT"
    assert out.fallback_used is True
    assert "abstain" in out.fallback_reason.lower()
    rec = store.get(_key_for(ctx))
    assert rec.vlm_count == 1
    assert 1 <= rec.medsam_count <= 4


def test_vlm_box_point_records_exact_medsam_object_count():
    store = _fresh_store()
    ctx = _ctx(arm="VLM_BOX_POINT")
    p = _good_payload("VLM_BOX_POINT", n=3)
    out = validate_or_fallback(json.dumps(p), ctx, _mlem(), store=store)
    assert out.executed_arm == "VLM_BOX_POINT"
    assert out.medsam_calls_used == 3
    rec = store.get(_key_for(ctx))
    assert rec.vlm_count == 1
    assert rec.medsam_count == 3


def test_vlm_multi_box_records_exact_medsam_object_count():
    store = _fresh_store()
    ctx = _ctx(arm="VLM_MULTI_BOX")
    p = _good_payload("VLM_MULTI_BOX", n=2)
    out = validate_or_fallback(json.dumps(p), ctx, _mlem(), store=store)
    assert out.medsam_calls_used == 2
    rec = store.get(_key_for(ctx))
    assert rec.medsam_count == 2


def test_det_component_counts_medsam_by_objects():
    store = _fresh_store()
    ctx = _ctx(arm="DET_COMPONENT")
    out = validate_or_fallback(None, ctx, _mlem(), store=store)
    assert out.executed_arm == "DET_COMPONENT"
    assert 1 <= out.medsam_calls_used <= 4
    rec = store.get(_key_for(ctx))
    assert rec.vlm_count == 0
    assert rec.medsam_count == out.medsam_calls_used


def test_full_box_counts_medsam_one():
    store = _fresh_store()
    ctx = _ctx(arm="FULL_BOX")
    out = validate_or_fallback(None, ctx, _mlem(), store=store)
    assert out.executed_arm == "FULL_BOX"
    assert out.medsam_calls_used == 1
    rec = store.get(_key_for(ctx))
    assert rec.medsam_count == 1


def test_vlm_single_box_counts_medsam_one():
    store = _fresh_store()
    ctx = _ctx(arm="VLM_SINGLE_BOX")
    p = _good_payload("VLM_SINGLE_BOX", n=1)
    out = validate_or_fallback(json.dumps(p), ctx, _mlem(), store=store)
    assert out.executed_arm == "VLM_SINGLE_BOX"
    assert out.medsam_calls_used == 1


# ============================================================================
# Fix 6: protocol MedSAM maximum map matches ARM_MAXES
# ============================================================================

def test_protocol_medsam_max_matches_arm_records():
    """The authoritative map must equal arm records exactly."""
    expected = {arm: ARM_MAXES[arm]["medsam_calls"] for arm in ARM_MAXES}
    assert FULL_VAL_MAX_MEDSAM_PER_CASE == expected


def test_full_val_max_vlm_transactions_constant():
    assert FULL_VAL_MAX_VLM_TRANSACTIONS == 5 * 8892 == 44460


# ============================================================================
# Fix 7: schema rejection examples
# ============================================================================

def test_missing_required_top_level_key_falls_back():
    raw = '{"schema_version": 1, "objects": []}'
    store = _fresh_store()
    ctx = _ctx(arm="VLM_SINGLE_BOX")
    out = validate_or_fallback(raw, ctx, _mlem(), store=store)
    assert out.executed_arm == "DET_COMPONENT"
    assert "missing required top-level keys" in out.fallback_reason


def test_top_level_extra_key_falls_back():
    raw = json.dumps({
        "schema_version": 1, "objects": [], "abstain": False,
        "fallback_reason": None, "extra_key": 1,
    })
    store = _fresh_store()
    ctx = _ctx(arm="VLM_SINGLE_BOX")
    out = validate_or_fallback(raw, ctx, _mlem(), store=store)
    assert out.executed_arm == "DET_COMPONENT"


def test_non_str_or_bytes_raw_falls_back():
    store = _fresh_store()
    ctx = _ctx(arm="VLM_BOX_POINT")
    # int raw → schema fallback (not crash)
    out = validate_or_fallback(12345, ctx, _mlem(), store=store)
    assert out.executed_arm == "DET_COMPONENT"


# ============================================================================
# Fix 8: refinement tests (per-object DecoderCallSpec)
# ============================================================================

def test_refinement_records_per_object_medsam():
    """Round 1 produces the initial prompt plan (decoder_call_specs empty
    since first_round_objects is None). Round 2 produces decoder specs."""
    store = _fresh_store()
    ctx = _ctx(arm="VLM_AGENT_REFINE", call_index=1)
    p = _good_payload("VLM_AGENT_REFINE", n=2)
    out1 = validate_or_fallback(json.dumps(p), ctx, _mlem(), store=store)
    assert out1.executed_arm == "VLM_AGENT_REFINE"
    # Round 1 has no prior logits yet → no decoder specs.
    assert out1.decoder_call_specs == ()
    assert out1.medsam_calls_used == 2


def test_refinement_round_2_uses_first_round_logits():
    store = _fresh_store()
    ctx1 = _ctx(arm="VLM_AGENT_REFINE", call_index=1)
    p1 = _good_payload("VLM_AGENT_REFINE", n=2)
    out1 = validate_or_fallback(json.dumps(p1), ctx1, _mlem(), store=store)
    logits = [np.full((4, 4), 0.5, dtype=np.float32),
              np.full((4, 4), -0.5, dtype=np.float32)]
    ctx2 = _ctx(arm="VLM_AGENT_REFINE", call_index=2)
    p2 = _good_payload("VLM_AGENT_REFINE", n=2)
    out2 = validate_or_fallback(json.dumps(p2), ctx2, _mlem(), store=store,
                                 first_round_objects=out1.objects,
                                 first_round_logits=logits)
    assert out2.executed_arm == "VLM_AGENT_REFINE"
    assert len(out2.decoder_call_specs) == 2
    rec = store.get(_key_for(ctx1))
    assert rec.vlm_count == 2
    assert rec.medsam_count == 4


# ============================================================================
# Other determinism / DET tests
# ============================================================================

def test_det_component_zero_signal_raises_fatal():
    with pytest.raises(FatalCaseInputError):
        det_component_objects(np.zeros((IMAGE_DIM, IMAGE_DIM),
                                       dtype=np.float32))


def test_vlm_box_point_one_object_success():
    store = _fresh_store()
    ctx = _ctx(arm="VLM_BOX_POINT")
    p = _good_payload("VLM_BOX_POINT", n=1)
    out = validate_or_fallback(json.dumps(p), ctx, _mlem(), store=store)
    assert out.executed_arm == "VLM_BOX_POINT"
    assert out.fallback_used is False


def test_full_box_with_raw_raises_fatal():
    store = _fresh_store()
    ctx = _ctx(arm="FULL_BOX")
    with pytest.raises(FatalCaseInputError):
        validate_or_fallback("{}", ctx, _mlem(), store=store)


def test_vlm_arm_with_raw_none_raises_fatal():
    store = _fresh_store()
    ctx = _ctx(arm="VLM_SINGLE_BOX")
    with pytest.raises(FatalCaseInputError):
        validate_or_fallback(None, ctx, _mlem(), store=store)


def test_pixelwise_max_aggregation_basic():
    a = np.array([[1.0, 0.0], [0.0, 0.0]])
    b = np.array([[0.0, 1.0], [0.0, 0.0]])
    out = pixelwise_max_aggregation([a, b])
    assert np.array_equal(out, np.array([[1.0, 1.0], [0.0, 0.0]]))


def test_pixelwise_max_aggregation_empty_raises():
    with pytest.raises(PromptPlanSchemaError):
        pixelwise_max_aggregation([])


def test_alldatasets_listed():
    assert ALLOWED_DATASETS == frozenset({"brainweb", "hecktor"})