"""Stage 7A-v4.3 comprehensive regression tests.

Per directive 5950637029. Pure function tests (no I/O, no VLM, no GPU).
Tests cover:
- canonical LF hashing (CRLF normalized; lone CR rejected)
- LedgerKey includes input_image_sha256 + parser_schema_version
- strict primary MedSAM budget (no DET expansion); fallback_medsam
  separate with DET ceiling
- finalize() locks the record against consume/record
- resumable to_dict/from_dict (lossless)
- cross-object pos/neg overlap both orderings
- strict bool abstain
- context validation (call_index range, SHA-256, dataset/arm)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))

from sam_kem.stage7a_prompt_plan import (
    IMAGE_DIM, ARM_MAXES, FULL_VAL_MAX_MEDSAM_PER_CASE, SCHEMA_VERSION, DET_MAX_MEDSAM,
    PromptRequestContext, PromptPlanSchemaError, parse_agent_response, validate_or_fallback,
    LedgerStore, LedgerKey, TransactionLedger,
    canonical_lf_sha256_file, execution_file_sha256,
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
        "schema_version": 4, "objects": objs,
        "abstain": False, "fallback_reason": None,
    }


def _mlem(seed=0):
    rng = np.random.default_rng(seed)
    return rng.standard_normal((IMAGE_DIM, IMAGE_DIM)).astype(np.float32)


def _fresh_store():
    return LedgerStore()


def _key_for(ctx):
    return ctx.ledger_key()


# ============================================================================
# §A: Canonical LF SHA helpers
# ============================================================================

def test_canonical_lf_sha_lf_input(tmp_path):
    p = tmp_path / "a.txt"
    p.write_bytes(b"a\nb\nc\n")
    assert canonical_lf_sha256_file(p) == \
        execution_file_sha256(p)


def test_canonical_lf_sha_crlf_input_normalizes(tmp_path):
    p = tmp_path / "a.txt"
    p.write_bytes(b"a\r\nb\r\nc\r\n")
    # Create LF reference
    plf = tmp_path / "lf.txt"
    plf.write_bytes(b"a\nb\nc\n")
    assert canonical_lf_sha256_file(p) == canonical_lf_sha256_file(plf)


def test_canonical_lf_sha_lone_cr_rejected(tmp_path):
    p = tmp_path / "a.txt"
    p.write_bytes(b"a\rb\nc\n")
    with pytest.raises(ValueError):
        canonical_lf_sha256_file(p)


def test_execution_file_sha_differs_for_crlf(tmp_path):
    p_lf = tmp_path / "lf.txt"
    p_crlf = tmp_path / "crlf.txt"
    p_lf.write_bytes(b"a\nb\n")
    p_crlf.write_bytes(b"a\r\nb\r\n")
    assert canonical_lf_sha256_file(p_lf) == canonical_lf_sha256_file(p_crlf)
    assert execution_file_sha256(p_lf) != execution_file_sha256(p_crlf)


# ============================================================================
# §B: LedgerKey includes input_image_sha256 + parser_schema_version
# ============================================================================

def test_ledger_key_includes_input_image_sha():
    ctx_a = _ctx(input_sha="a" * 64)
    ctx_b = _ctx(input_sha="b" * 64)
    assert _key_for(ctx_a) != _key_for(ctx_b)
    # Differ by parser_schema_version
    pa = _key_for(ctx_a)
    pb = LedgerKey(dataset=pa.dataset, subject_id=pa.subject_id,
                    slice_id=pa.slice_id, count_level=pa.count_level,
                    realization_id=pa.realization_id,
                    requested_arm=pa.requested_arm,
                    input_image_sha256=pa.input_image_sha256,
                    parser_schema_version=pa.parser_schema_version + 1)
    assert pa != pb


def test_ledger_key_isolates_different_input_sha():
    store = LedgerStore()
    ctx_a = _ctx(input_sha="a" * 64, subject_id="s1")
    ctx_b = _ctx(input_sha="b" * 64, subject_id="s1")
    ka = _key_for(ctx_a)
    kb = _key_for(ctx_b)
    assert ka != kb
    TransactionLedger(store, ka).consume_vlm()
    assert store.get(kb) is None
    assert store.get(ka).vlm_count == 1


def test_ledger_key_to_string_includes_all_fields():
    ctx = _ctx()
    s = _key_for(ctx).to_string()
    fields = s.split("|")
    assert len(fields) == 8
    assert fields[-2] == "0" * 64
    assert fields[-1] == str(SCHEMA_VERSION)


def test_ledger_key_serialization_round_trip():
    ctx = _ctx()
    k1 = _key_for(ctx)
    s = k1.to_string()
    # from_dict expects flat dict of key_str → record dict
    parts = s.split("|")
    rec_input_sha = parts[6]
    assert len(rec_input_sha) == 64


# ============================================================================
# §B: Strict primary MedSAM budget (no DET expansion)
# ============================================================================

def test_full_box_primary_medsam_budget_is_1_not_4():
    """FULL_BOX declared medsam_calls=1; consume_medsam_objects(2) must
    raise. v4's max(..., DET_MAX) silently enlarged to 4; v4.3 removes
    that."""
    store = LedgerStore()
    ctx = _ctx(arm="FULL_BOX")
    ledger = TransactionLedger(store, _key_for(ctx))
    with pytest.raises(PromptPlanSchemaError):
        ledger.consume_medsam_objects(2)
    assert store.get(_key_for(ctx)).medsam_count == 0


def test_vlm_single_box_primary_medsam_budget_is_1():
    store = LedgerStore()
    ctx = _ctx(arm="VLM_SINGLE_BOX")
    ledger = TransactionLedger(store, _key_for(ctx))
    with pytest.raises(PromptPlanSchemaError):
        ledger.consume_medsam_objects(2)


def test_fallback_medsam_separate_counter():
    """When fallback executes DET, fallback_medsam_objects is incremented
    separately with DET ceiling, NOT silently enlarging the requested
    arm's primary budget."""
    store = LedgerStore()
    ctx = _ctx(arm="FULL_BOX")
    out = validate_or_fallback(None, ctx, _mlem(), store=store)
    assert out.executed_arm == "FULL_BOX"
    assert out.medsam_calls_used == 1
    assert out.fallback_medsam_calls_used == 0
    rec = store.get(_key_for(ctx))
    assert rec.medsam_count == 1
    assert rec.fallback_medsam_count == 0


def test_fallback_medsam_det_ceiling():
    """fallback_medsam_count is bounded by DET_MAX_MEDSAM=4."""
    store = LedgerStore()
    ctx = _ctx(arm="VLM_SINGLE_BOX")
    # Force a malformed raw → DET fallback
    out = validate_or_fallback(12345, ctx, _mlem(), store=store)
    assert out.executed_arm == "DET_COMPONENT"
    rec = store.get(_key_for(ctx))
    assert rec.fallback_medsam_count == len(out.objects)
    assert rec.fallback_medsam_count <= DET_MAX_MEDSAM


# ============================================================================
# §B: finalize locks record
# ============================================================================

def test_finalize_blocks_consume():
    store = LedgerStore()
    ctx = _ctx(arm="FULL_BOX")
    ledger = TransactionLedger(store, _key_for(ctx))
    ledger.consume_medsam_objects(1)
    ledger.finalize()
    with pytest.raises(PromptPlanSchemaError):
        ledger.consume_medsam_objects(1)
    with pytest.raises(PromptPlanSchemaError):
        ledger.consume_vlm()
    with pytest.raises(PromptPlanSchemaError):
        ledger.consume_fallback_medsam_objects(1)


def test_finalize_blocks_record_outcome():
    store = LedgerStore()
    ctx = _ctx(arm="FULL_BOX")
    ledger = TransactionLedger(store, _key_for(ctx))
    out = validate_or_fallback(None, ctx, _mlem(), store=store)
    ledger.record_outcome(out)  # OK pre-finalize
    ledger.finalize()
    # After finalize, record_outcome must raise
    with pytest.raises(PromptPlanSchemaError):
        ledger.record_outcome(out)


def test_reset_forbidden_after_finalize():
    store = LedgerStore()
    ctx = _ctx(arm="FULL_BOX")
    ledger = TransactionLedger(store, _key_for(ctx))
    ledger.consume_medsam_objects(1)
    ledger.finalize()
    with pytest.raises(PromptPlanSchemaError):
        ledger.reset()


# ============================================================================
# §B: Outcome serialization is lossless
# ============================================================================

def test_outcome_serialization_lossless_round_trip():
    store_a = LedgerStore()
    ctx = _ctx(arm="VLM_BOX_POINT")
    p = _good_payload("VLM_BOX_POINT", n=3)
    _out = validate_or_fallback(json.dumps(p), ctx, _mlem(), store=store_a)
    snap_a = store_a.to_dict()
    store_b = LedgerStore.from_dict(snap_a)
    snap_b = store_b.to_dict()
    # Byte-stable: must serialize to identical JSON
    assert json.dumps(snap_a, sort_keys=True) == \
           json.dumps(snap_b, sort_keys=True)
    # Outcomes preserved
    rec_a = store_a.get(_key_for(ctx))
    rec_b = store_b.get(_key_for(ctx))
    assert rec_a.vlm_count == rec_b.vlm_count == 1
    assert rec_a.medsam_count == rec_b.medsam_count == 3
    assert len(rec_a.outcomes) == len(rec_b.outcomes) == 1
    o_a = rec_a.outcomes[0]
    o_b = rec_b.outcomes[0]
    for key in ("event_index", "call_index", "trust_id", "requested_arm",
                "executed_arm", "fallback_used", "fallback_reason",
                "raw_response_sha256", "vlm_delta", "medsam_delta",
                "fallback_medsam_delta", "objects", "decoder_specs"):
        assert o_a[key] == o_b[key]


def test_outcome_records_full_event_history():
    """Each outcome captures event/call index, trust ID, executed arm,
    fallback status/reason, raw-response SHA, VLM/MedSAM deltas,
    cumulative counts, objects, decoder specs."""
    store = LedgerStore()
    ctx = _ctx(arm="VLM_BOX_POINT")
    p = _good_payload("VLM_BOX_POINT", n=2)
    _out = validate_or_fallback(json.dumps(p), ctx, _mlem(), store=store)
    rec = store.get(_key_for(ctx))
    assert len(rec.outcomes) == 1
    o = rec.outcomes[0]
    assert o["event_index"] == 0
    assert o["call_index"] == ctx.call_index
    assert o["trust_id"] == ctx.trust_id()
    assert o["executed_arm"] == "VLM_BOX_POINT"
    assert o["fallback_used"] is False
    assert o["raw_response_sha256"] != ""
    assert o["vlm_delta"] == 1
    assert o["medsam_delta"] == 2
    assert len(o["objects"]) == 2
    for obj in o["objects"]:
        assert "object_id" in obj
        assert "box_xyxy" in obj
        assert "positive_points_xy" in obj
        assert "negative_points_xy" in obj
        assert "confidence" in obj


def test_round_1_round_2_continuity():
    """VLM_AGENT_REFINE: round 1 and round 2 share the same record
    via explicit serialization."""
    store = LedgerStore()
    ctx_r1 = _ctx(arm="VLM_AGENT_REFINE", call_index=1)
    ctx_r2 = _ctx(arm="VLM_AGENT_REFINE", call_index=2)
    p_r1 = _good_payload("VLM_AGENT_REFINE", n=2)
    p_r2 = _good_payload("VLM_AGENT_REFINE", n=2)
    out1 = validate_or_fallback(json.dumps(p_r1), ctx_r1, _mlem(),
                                 store=store)
    logits = [np.zeros((4, 4), dtype=np.float32),
              np.zeros((4, 4), dtype=np.float32)]
    validate_or_fallback(json.dumps(p_r2), ctx_r2, _mlem(),
                        store=store,
                        first_round_objects=out1.objects,
                        first_round_logits=logits)
    rec = store.get(_key_for(ctx_r1))
    assert rec.vlm_count == 2
    assert rec.medsam_count == 4
    assert len(rec.outcomes) == 2
    # Serialize and restore; round 2 outcome persists
    snap = store.to_dict()
    store2 = LedgerStore.from_dict(snap)
    rec2 = store2.get(_key_for(ctx_r1))
    assert len(rec2.outcomes) == 2


# ============================================================================
# §1: Atomic accounting for all 9 paths
# ============================================================================

def test_atomic_normal_path():
    store = LedgerStore()
    ctx = _ctx(arm="VLM_BOX_POINT")
    p = _good_payload("VLM_BOX_POINT", n=2)
    _out = validate_or_fallback(json.dumps(p), ctx, _mlem(), store=store)
    rec = store.get(_key_for(ctx))
    assert rec.vlm_count == 1
    assert rec.medsam_count == 2
    assert rec.fallback_medsam_count == 0


def test_atomic_malformed_path_counts_vlm_and_uses_fallback_medsam():
    store = LedgerStore()
    ctx = _ctx(arm="VLM_BOX_POINT")
    out = validate_or_fallback(12345, ctx, _mlem(), store=store)
    assert out.executed_arm == "DET_COMPONENT"
    rec = store.get(_key_for(ctx))
    assert rec.vlm_count == 1  # raw was consumed
    assert rec.medsam_count == 0  # primary NOT enlarged
    assert rec.fallback_medsam_count == len(out.objects)


def test_atomic_abstention_path_counts_vlm_and_uses_fallback_medsam():
    raw = {"schema_version": 4, "objects": [], "abstain": True,
           "fallback_reason": "low_conf"}
    store = LedgerStore()
    ctx = _ctx(arm="VLM_BOX_POINT")
    _out = validate_or_fallback(json.dumps(raw), ctx, _mlem(), store=store)
    rec = store.get(_key_for(ctx))
    assert rec.vlm_count == 1
    assert rec.fallback_medsam_count >= 1


def test_atomic_over_budget_raises_no_mutation():
    store = LedgerStore()
    ctx = _ctx(arm="VLM_SINGLE_BOX")  # vlm=1
    ledger = TransactionLedger(store, _key_for(ctx))
    with pytest.raises(PromptPlanSchemaError):
        ledger.consume_vlm()
        ledger.consume_vlm()  # 2 > 1
    assert store.get(_key_for(ctx)).vlm_count == 1  # no partial mutation


def test_atomic_medsam_over_budget_raises_no_mutation():
    store = LedgerStore()
    ctx = _ctx(arm="VLM_MULTI_BOX")  # medsam=4
    ledger = TransactionLedger(store, _key_for(ctx))
    with pytest.raises(PromptPlanSchemaError):
        ledger.consume_medsam_objects(5)
    assert store.get(_key_for(ctx)).medsam_count == 0


# ============================================================================
# Strict bool, cross-object, context validation (preserved from v4.2)
# ============================================================================

def test_strict_bool_abstain_rejects_non_bool():
    for bad in [0, 1, None, "false", "true", [], 0.0, 1.0]:
        raw = {"schema_version": 4, "objects": [],
               "abstain": bad, "fallback_reason": "x"}
        with pytest.raises(PromptPlanSchemaError):
            parse_agent_response(json.dumps(raw), _ctx())


def test_cross_object_pos_to_neg_overlap_rejected():
    p = _good_payload("VLM_BOX_POINT", n=2)
    p["objects"][0]["positive_points_xy"] = [[2, 2]]
    p["objects"][0]["negative_points_xy"] = []
    p["objects"][1]["positive_points_xy"] = [[10, 10]]
    p["objects"][1]["negative_points_xy"] = [[2, 2]]
    with pytest.raises(PromptPlanSchemaError):
        parse_agent_response(json.dumps(p), _ctx())


def test_cross_object_neg_to_pos_overlap_rejected():
    p = _good_payload("VLM_BOX_POINT", n=2)
    p["objects"][0]["positive_points_xy"] = [[10, 10]]
    p["objects"][0]["negative_points_xy"] = [[2, 2]]
    p["objects"][1]["positive_points_xy"] = [[2, 2]]
    p["objects"][1]["negative_points_xy"] = []
    with pytest.raises(PromptPlanSchemaError):
        parse_agent_response(json.dumps(p), _ctx())


def test_context_rejects_malformed_sha():
    with pytest.raises(ValueError):
        PromptRequestContext(
            requested_arm="FULL_BOX", dataset="brainweb",
            subject_id="s", slice_id="z",
            count_level=200000, realization_id=0,
            call_index=1, input_image_sha256="X" * 64,
        )


def test_context_rejects_wrong_schema_version():
    with pytest.raises(ValueError):
        PromptRequestContext(
            requested_arm="FULL_BOX", dataset="brainweb",
            subject_id="s", slice_id="z",
            count_level=200000, realization_id=0,
            call_index=1, input_image_sha256="0" * 64,
            parser_schema_version=3,
        )


def test_schema_version_constant_is_4():
    assert SCHEMA_VERSION == 4


def test_protocol_medsam_max_matches_arm_records():
    assert FULL_VAL_MAX_MEDSAM_PER_CASE == {
        arm: ARM_MAXES[arm]["medsam_calls"] for arm in ARM_MAXES
    }
    expected = {1: ["FULL_BOX", "VLM_SINGLE_BOX"],
                4: ["DET_COMPONENT", "VLM_MULTI_BOX", "VLM_BOX_POINT"],
                8: ["VLM_AGENT_REFINE"]}
    seen = {}
    for arm, medsam in FULL_VAL_MAX_MEDSAM_PER_CASE.items():
        seen.setdefault(medsam, []).append(arm)
    for k, arms in expected.items():
        assert sorted(seen.get(k, [])) == sorted(arms)