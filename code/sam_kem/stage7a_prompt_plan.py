"""Stage 7A-v4 closure patch — corrected prompt-plan module.

Per directive 5928360149. All seven defects addressed:

1. Dedicated box validator strict 0<=x0,y0<128, 0<x1,y1<=128.
2. Duplicate boxes/points rejected with valid IDs.
3. Deterministic arms (FULL_BOX, DET_COMPONENT) bypass parsing
   when raw is None; raise FatalCaseInputError otherwise.
4. Public execution path validates case input first; VLM_AGENT_REFINE
   now validated under per-object box+point contract.
5. DET_COMPONENT peak selected from component coordinates only with
   frozen row-major tie-break; deterministic sort keys; internal
   object_ids 1..N.
6. Refinement package: DecoderCallSpec per object; package_refinement_inputs
   takes (first_objects, first_round_logits, correction_objects) only;
   returns specs with each object's first-round float logit as its
   own mask_input.
7. Persistent transaction ledger keyed on case identity + arm; correct
   vlm/medsam maxima; final execution counters remain zero.
"""
from __future__ import annotations

import hashlib
import json
from collections import OrderedDict
from dataclasses import dataclass

import numpy as np


# ============================================================================
# Constants
# ============================================================================

SCHEMA_VERSION = 1
IMAGE_DIM = 128

# Per-arm maxes (corrected per directive §5).
ARM_MAXES = {
    "FULL_BOX":        dict(max_vlm_calls=0, max_medsam_calls=1,
                          max_objects=1, max_negative_points=3,
                          min_positive_points=0, max_positive_points=0,
                          allow_multi_round=False, max_calls_per_case=1),
    "DET_COMPONENT":   dict(max_vlm_calls=0, max_medsam_calls=4,
                          max_objects=4, max_negative_points=3,
                          min_positive_points=1, max_positive_points=1,
                          allow_multi_round=False, max_calls_per_case=1),
    "VLM_SINGLE_BOX":  dict(max_vlm_calls=1, max_medsam_calls=1,
                          max_objects=1, max_negative_points=1,
                          min_positive_points=1, max_positive_points=1,
                          allow_multi_round=False, max_calls_per_case=1),
    "VLM_MULTI_BOX":   dict(max_vlm_calls=1, max_medsam_calls=4,
                          max_objects=4, max_negative_points=3,
                          min_positive_points=0, max_positive_points=0,
                          allow_multi_round=False, max_calls_per_case=1),
    "VLM_BOX_POINT":   dict(max_vlm_calls=1, max_medsam_calls=4,
                          max_objects=4, max_negative_points=3,
                          min_positive_points=1, max_positive_points=3,
                          allow_multi_round=False, max_calls_per_case=1),
    "VLM_AGENT_REFINE": dict(max_vlm_calls=2, max_medsam_calls=8,
                          min_objects=1, max_objects=4,
                          max_negative_points=3,
                          min_positive_points=1, max_positive_points=3,
                          allow_multi_round=True, max_calls_per_case=2),
}

# Max VLM transactions across full validation
# FULL_BOX: 0, DET: 0, SINGLE: 1, MULTI: 1, BOX_POINT: 1, AGENT_REFINE: 2
# Sum per case: 6; per 8892 cases: 6 * 8892 = 53352
FULL_VAL_MAX_VLM_TRANSACTIONS = (
    sum(ARM_MAXES[arm]["max_vlm_calls"] for arm in ARM_MAXES)
    * 8892
)

ALLOWED_OBJECT_KEYS = (
    "object_id", "box_xyxy", "positive_points_xy",
    "negative_points_xy", "confidence",
)

ALLOWED_TOP_KEYS = (
    "schema_version", "objects", "abstain", "fallback_reason",
)


# ============================================================================
# Trusted context
# ============================================================================

@dataclass(frozen=True)
class PromptRequestContext:
    """Immutable; arm/identity/call_index must come from here, never raw."""
    requested_arm: str
    dataset: str
    subject_id: str
    slice_id: str
    count_level: int
    realization_id: int
    call_index: int            # 1 = initial, 2 = correction
    input_image_sha256: str   # trusted caller pre-computes
    parser_schema_version: int = SCHEMA_VERSION

    def trust_id(self) -> str:
        """Full identity hash (per directive §A.5, §8)."""
        canonical = (
            f"{self.requested_arm}|{self.dataset}|{self.subject_id}|"
            f"{self.slice_id}|{self.count_level}|{self.realization_id}|"
            f"{self.call_index}|{self.input_image_sha256}|"
            f"{self.parser_schema_version}")
        return hashlib.sha256(canonical.encode()).hexdigest()

    def __post_init__(self):
        valid_arms = tuple(ARM_MAXES.keys())
        if self.requested_arm not in valid_arms:
            raise ValueError(f"requested_arm must be one of {valid_arms}")
        if self.requested_arm != "VLM_AGENT_REFINE" and self.call_index > 1:
            raise ValueError(
                f"{self.requested_arm} cannot have call_index > 1")
        if self.requested_arm == "VLM_AGENT_REFINE" and self.call_index > 2:
            raise ValueError(
                "VLM_AGENT_REFINE cannot have call_index > 2")


# ============================================================================
# Errors
# ============================================================================

class PromptPlanSchemaError(ValueError):
    pass


class FatalCaseInputError(ValueError):
    """Case input itself invalid (nonfinite mlem, wrong shape, raw for
    deterministic arm, etc.). Never silently replaced by fallback."""


# ============================================================================
# Strict box/point validators
# ============================================================================

def _validate_int(v, label, allow_zero=True):
    if isinstance(v, bool) or not isinstance(v, int):
        raise PromptPlanSchemaError(
            f"{label} must be int (no bool coercion); got {v!r}")
    return v


def _validate_point_xy(p, label):
    """Per-object point strict validation: 0 <= x < 128, 0 <= y < 128."""
    if not isinstance(p, list) or len(p) != 2:
        raise PromptPlanSchemaError(
            f"{label} must be a list of two ints [x, y]; got {p!r}")
    x, y = p
    _validate_int(x, f"{label}[0]"); _validate_int(y, f"{label}[1]")
    if not (0 <= x < IMAGE_DIM):
        raise PromptPlanSchemaError(
            f"{label} x={x} out of range [0, {IMAGE_DIM})")
    if not (0 <= y < IMAGE_DIM):
        raise PromptPlanSchemaError(
            f"{label} y={y} out of range [0, {IMAGE_DIM})")
    return (x, y)


def _validate_box_xyxy(box, label):
    """Strict half-open XYXY: 0 <= x0 < 128; 0 < x1 <= 128 (inclusive upper).
    x0 < x1 and y0 < y1 strict."""
    if not isinstance(box, list) or len(box) != 4:
        raise PromptPlanSchemaError(
            f"{label} must be a list of 4 ints [x0, y0, x1, y1]; got {box!r}")
    x0, y0, x1, y1 = box
    _validate_int(x0, f"{label}[0]"); _validate_int(y0, f"{label}[1]")
    _validate_int(x1, f"{label}[2]"); _validate_int(y1, f"{label}[3]")
    if not (0 <= x0 < IMAGE_DIM):
        raise PromptPlanSchemaError(
            f"{label} x0={x0} out of range [0, {IMAGE_DIM})")
    if not (0 <= y0 < IMAGE_DIM):
        raise PromptPlanSchemaError(
            f"{label} y0={y0} out of range [0, {IMAGE_DIM})")
    if not (0 < x1 <= IMAGE_DIM):
        raise PromptPlanSchemaError(
            f"{label} x1={x1} must be in (0, {IMAGE_DIM}]")
    if not (0 < y1 <= IMAGE_DIM):
        raise PromptPlanSchemaError(
            f"{label} y1={y1} must be in (0, {IMAGE_DIM}]")
    if not (x0 < x1):
        raise PromptPlanSchemaError(f"{label} must satisfy x0 < x1; got {box}")
    if not (y0 < y1):
        raise PromptPlanSchemaError(f"{label} must satisfy y0 < y1; got {box}")
    return (x0, y0, x1, y1)


# ============================================================================
# Parsed object + response model
# ============================================================================

@dataclass(frozen=True)
class PromptObject:
    object_id: int
    box_xyxy: tuple
    positive_points_xy: tuple
    negative_points_xy: tuple
    confidence: float


@dataclass(frozen=True)
class AgentResponse:
    schema_version: int
    objects: tuple
    abstain: bool
    fallback_reason: str
    raw_response_sha256: str


@dataclass(frozen=True)
class DecoderCallSpec:
    """Per-object decoder call spec. mask_input is that object's
    first-round low-resolution logit (finite float, same shape)."""
    object_id: int
    box_xyxy: tuple
    positive_points_xy: tuple
    negative_points_xy: tuple
    mask_input: np.ndarray


@dataclass(frozen=True)
class ValidationOutcome:
    requested_arm: str
    executed_arm: str
    trust_id: str
    raw_response_sha256: str
    fallback_used: bool
    fallback_reason: str
    objects: tuple
    decoder_call_specs: tuple
    vlm_calls_used: int = 0
    medsam_calls_used: int = 0


# ============================================================================
# Strict raw-JSON validator
# ============================================================================

def _check_no_duplicate_keys(pairs):
    seen = {}
    out = OrderedDict()
    for k, v in pairs:
        if k in seen:
            raise PromptPlanSchemaError(f"duplicate JSON key: {k!r}")
        seen[k] = True
        out[k] = v
    return out


def _check_duplicate_boxes(boxes_seen, box, obj_id, partner_label):
    """Per-object-pair duplicate box rejection (IoU == 1.0)."""
    for prev_obj_id, prev_box in boxes_seen:
        if prev_box == box:
            raise PromptPlanSchemaError(
                f"objects[{obj_id}] box duplicates object {prev_obj_id} "
                f"({partner_label} duplicate)")
    boxes_seen.append((obj_id, box))


def _check_duplicate_points(seen_points, label, p, obj_id, idx):
    """Reject exact duplicates within a single point set."""
    if p in seen_points:
        raise PromptPlanSchemaError(
            f"object {obj_id} duplicate {label} point {p} (index {idx})")
    seen_points.append(p)


def _check_neg_pos_overlap(seen_positive, obj_id, neg_pt, idx):
    """Reject the same coordinate appearing as both positive and negative."""
    if neg_pt in seen_positive:
        raise PromptPlanSchemaError(
            f"object {obj_id} negative_point_xy[{idx}]={neg_pt} "
            f"duplicates a positive_point_xy coordinate")


def _parse_object(obj_raw, idx, seen_obj_ids, boxes_seen):
    """Parse one object strictly."""
    if not isinstance(obj_raw, dict):
        raise PromptPlanSchemaError(f"objects[{idx}] must be an object")
    extra = set(obj_raw.keys()) - set(ALLOWED_OBJECT_KEYS)
    if extra:
        raise PromptPlanSchemaError(
            f"objects[{idx}] forbidden/unknown keys: {sorted(extra)}")
    for k in ALLOWED_OBJECT_KEYS:
        if k not in obj_raw:
            raise PromptPlanSchemaError(
                f"objects[{idx}] missing required field: {k}")
    # object_id unique positive int
    oid = obj_raw["object_id"]
    _validate_int(oid, f"objects[{idx}].object_id")
    if oid <= 0:
        raise PromptPlanSchemaError(
            f"objects[{idx}].object_id must be positive; got {oid}")
    if oid in seen_obj_ids:
        raise PromptPlanSchemaError(
            f"objects[{idx}].object_id={oid} duplicates a previous object_id")
    seen_obj_ids.add(oid)
    # box_xyxy
    box = _validate_box_xyxy(obj_raw["box_xyxy"], f"objects[{idx}].box_xyxy")
    _check_duplicate_boxes(boxes_seen, box, oid, "positive")
    # positive_points
    pos_raw = obj_raw["positive_points_xy"]
    if not isinstance(pos_raw, list):
        raise PromptPlanSchemaError(
            f"object {oid} positive_points_xy must be a list")
    seen_pos = []
    pos = []
    for j, p in enumerate(pos_raw):
        pos.append(_check_duplicate_points(seen_pos, "positive", p, oid, j))
    # negative_points
    neg_raw = obj_raw["negative_points_xy"]
    if not isinstance(neg_raw, list):
        raise PromptPlanSchemaError(
            f"object {oid} negative_points_xy must be a list")
    neg = []
    for j, p in enumerate(neg_raw):
        np_ = _validate_point_xy(p, f"object {oid} negative_points_xy[{j}]")
        _check_neg_pos_overlap(seen_pos, oid, np_, j)
        neg.append(np_)
    # confidence
    conf = obj_raw["confidence"]
    if isinstance(conf, bool):
        raise PromptPlanSchemaError(
            f"object {oid}.confidence must not be bool; got {conf!r}")
    if not isinstance(conf, (int, float)):
        raise PromptPlanSchemaError(
            f"object {oid}.confidence must be number; got {type(conf).__name__}")
    if conf != conf or conf in (float("inf"), float("-inf")):
        raise PromptPlanSchemaError(
            f"object {oid}.confidence must be finite; got {conf!r}")
    if not (0.0 <= float(conf) <= 1.0):
        raise PromptPlanSchemaError(
            f"object {oid}.confidence must be in [0, 1]; got {conf}")
    return PromptObject(
        object_id=oid,
        box_xyxy=box,
        positive_points_xy=tuple(pos),
        negative_points_xy=tuple(neg),
        confidence=float(conf),
    )


def parse_agent_response(raw: str | bytes, context: PromptRequestContext,
                         seen_obj_ids=None) -> AgentResponse:
    """Strict raw-JSON validator. Reject duplicate boxes/points/keys.
    Callers can pass seen_obj_ids to thread object IDs across multiple
    parse calls (refinement)."""
    if isinstance(raw, bytes):
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError as e:
            raise PromptPlanSchemaError(f"raw bytes not valid UTF-8: {e}") from e
    elif isinstance(raw, str):
        text = raw
    else:
        raise PromptPlanSchemaError(
            f"raw must be str or bytes; got {type(raw).__name__}")
    raw_sha = hashlib.sha256(
        raw.encode("utf-8") if isinstance(raw, str) else raw
    ).hexdigest()
    try:
        parsed = json.loads(text, object_pairs_hook=_check_no_duplicate_keys)
    except json.JSONDecodeError as e:
        raise PromptPlanSchemaError(f"malformed JSON: {e}") from e
    if not isinstance(parsed, dict):
        raise PromptPlanSchemaError(
            f"top-level JSON must be an object; got {type(parsed).__name__}")
    extra = set(parsed.keys()) - set(ALLOWED_TOP_KEYS)
    if extra:
        raise PromptPlanSchemaError(
            f"top-level forbidden/unknown keys: {sorted(extra)}")
    if seen_obj_ids is None:
        seen_obj_ids = set()
    boxes_seen = []
    # schema_version strict
    sv = parsed["schema_version"]
    _validate_int(sv, "schema_version")
    if sv != SCHEMA_VERSION:
        raise PromptPlanSchemaError(
            f"schema_version must be {SCHEMA_VERSION}; got {sv}")
    # must match context
    if sv != context.parser_schema_version:
        raise PromptPlanSchemaError(
            f"schema_version {sv} != context {context.parser_schema_version}")
    # schema_version must be in expected positions (top-level)
    abstain_raw = parsed["abstain"]
    if not isinstance(abstain_raw, bool):
        raise PromptPlanSchemaError(
            f"abstain must be bool; got {type(abstain_raw).__name__}")
    if abstain_raw:
        objs_raw = parsed["objects"]
        if not isinstance(objs_raw, list) or len(objs_raw) > 0:
            raise PromptPlanSchemaError(
                "abstain=true requires empty objects list")
        fr_raw = parsed["fallback_reason"]
        if not isinstance(fr_raw, str) or len(fr_raw.strip()) == 0:
            raise PromptPlanSchemaError(
                "abstain=true requires nonempty string fallback_reason")
        return AgentResponse(
            schema_version=sv, objects=tuple(), abstain=True,
            fallback_reason=str(fr_raw),
            raw_response_sha256=raw_sha,
        )
    fr_raw = parsed["fallback_reason"]
    if fr_raw is not None:
        raise PromptPlanSchemaError(
            f"non-abstain requires fallback_reason == null; got {fr_raw!r}")
    objs_raw = parsed["objects"]
    if not isinstance(objs_raw, list):
        raise PromptPlanSchemaError("objects must be a list")
    objects = []
    for i, obj_raw in enumerate(objs_raw):
        objects.append(_parse_object(obj_raw, i, seen_obj_ids, boxes_seen))
    return AgentResponse(
        schema_version=sv, objects=tuple(objects), abstain=False,
        fallback_reason="", raw_response_sha256=raw_sha,
    )


# ============================================================================
# Public execution path
# ============================================================================

def _validate_case_input(mlem10: np.ndarray, context: PromptRequestContext):
    if mlem10 is None or not isinstance(mlem10, np.ndarray):
        raise FatalCaseInputError(
            "same_realization_mlem10 is required and must be ndarray")
    if mlem10.shape != (IMAGE_DIM, IMAGE_DIM):
        raise FatalCaseInputError(
            f"same_realization_mlem10 must be {(IMAGE_DIM, IMAGE_DIM)}; "
            f"got {mlem10.shape}")
    if not np.isfinite(mlem10).all():
        raise FatalCaseInputError(
            "same_realization_mlem10 must be fully finite")
    if mlem10.dtype.kind != "f":
        raise FatalCaseInputError(
            f"same_realization_mlem10 must be floating dtype; got {mlem10.dtype}")
    if float(np.abs(mlem10).max()) <= 0.0:
        raise FatalCaseInputError(
            "same_realization_mlem10 has no signal (all-zero); "
            "DET_COMPONENT cannot produce a deterministic plan")


def _validate_vlm_response(parsed: AgentResponse, arm: str,
                            context: PromptRequestContext,
                            seen_obj_ids):
    """Per-arm contract. ARM_MAXES is authoritative."""
    rules = ARM_MAXES[arm]
    n = len(parsed.objects)
    if not (rules["min_objects"] <= n <= rules["max_objects"]):
        raise PromptPlanSchemaError(
            f"{arm}: object count {n} out of range "
            f"[{rules['min_objects']}, {rules['max_objects']}]")
    # Per-object point ranges
    for obj in parsed.objects:
        if len(obj.positive_points_xy) > rules["max_positive_points"]:
            raise PromptPlanSchemaError(
                f"{arm}: object {obj.object_id} positive_points too many")
        if len(obj.positive_points_xy) < rules["min_positive_points"]:
            raise PromptPlanSchemaError(
                f"{arm}: object {obj.object_id} too few positive_points")
        if len(obj.negative_points_xy) > rules["max_negative_points"]:
            raise PromptPlanSchemaError(
                f"{arm}: object {obj.object_id} negative_points too many")
        if len(obj.negative_points_xy) < 0:
            raise PromptPlanSchemaError(
                f"{arm}: negative_points must be non-negative")
    return True


def package_refinement_inputs(
    first_round_objects: tuple, first_round_logits: list,
    correction_objects: tuple,
    context: PromptRequestContext,
) -> list:
    """Public wrapper for the closure-patch refinement packaging.
    Returns one DecoderCallSpec per object, with that object's
    first-round finite float logit as its own mask_input."""
    return _build_decoder_call_specs_impl(
        first_round_objects, first_round_logits,
        correction_objects, context)


def _build_decoder_call_specs_impl(
    first_round_objects: tuple, first_round_logits: list,
    correction_objects: tuple,
    context: PromptRequestContext,
) -> list:
    """Validate mask_input per object; return DecoderCallSpec list."""
    rules = ARM_MAXES[context.requested_arm]
    if len(first_round_objects) != len(first_round_logits):
        raise PromptPlanSchemaError("first_round_objects vs logits mismatch")
    if len(first_round_objects) != len(correction_objects):
        raise PromptPlanSchemaError("correction vs first mismatch")
    # Match by object_id
    by_id = {o.object_id: (o, fl) for o, fl in
             zip(first_round_objects, first_round_logits)}
    specs = []
    for co in correction_objects:
        if co.object_id not in by_id:
            raise PromptPlanSchemaError(
                f"object_id {co.object_id} missing from first round")
        fo, fl = by_id[co.object_id]
        if not isinstance(fl, np.ndarray):
            raise PromptPlanSchemaError(
                f"first_round_logits[{co.object_id}] must be ndarray")
        if fl.dtype.kind != "f":
            raise PromptPlanSchemaError(
                f"first_round_logits[{co.object_id}] must be floating")
        if not np.isfinite(fl).all():
            raise PromptPlanSchemaError(
                f"first_round_logits[{co.object_id}] non-finite")
        # Box equality check
        if fo.box_xyxy != co.box_xyxy:
            raise PromptPlanSchemaError(
                f"object_id {co.object_id}: box changed across rounds "
                f"(first={fo.box_xyxy}, correction={co.box_xyxy})")
        # Correction points check
        for cp in co.positive_points_xy:
            if cp not in fo.positive_points_xy:
                raise PromptPlanSchemaError(
                    f"object_id {co.object_id}: correction positive "
                    f"{cp} not in first round {fo.positive_points_xy}")
        specs.append(DecoderCallSpec(
            object_id=co.object_id,
            box_xyxy=co.box_xyxy,
            positive_points_xy=co.positive_points_xy,
            negative_points_xy=co.negative_points_xy,
            mask_input=fl,
        ))
    if len(specs) > rules["max_medsam_calls"]:
        raise PromptPlanSchemaError(
            f"{context.requested_arm}: exceeds max_medsam_calls={rules['max_medsam_calls']}")
    return specs


def validate_or_fallback(
    raw: str | bytes | None, context: PromptRequestContext,
    same_realization_mlem10: np.ndarray,
    seen_obj_ids: set | None = None,
    previous_objects: tuple | None = None,
    first_round_logits: list | None = None,
) -> ValidationOutcome:
    """Single executable entry point.

    Deterministic arms (FULL_BOX, DET_COMPONENT) require raw is None;
    VLM arms require raw is str/bytes.
    """
    _validate_case_input(same_realization_mlem10, context)
    arm = context.requested_arm

    # Deterministic arms: no raw, no parsing, no VLM
    if arm in ("FULL_BOX", "DET_COMPONENT"):
        if raw is not None:
            raise FatalCaseInputError(
                f"{arm} must have raw=None; got {type(raw).__name__}")
        if arm == "FULL_BOX":
            objs = full_box_objects()
        else:
            objs = tuple(det_component_objects(same_realization_mlem10))
        return ValidationOutcome(
            requested_arm=arm, executed_arm=arm,
            trust_id=context.trust_id(),
            raw_response_sha256="",
            fallback_used=False, fallback_reason="",
            objects=tuple(objs),
            decoder_call_specs=tuple(),
            vlm_calls_used=0,
            medsam_calls_used=len(objs),
        )

    # VLM arms: raw required
    if raw is None:
        raise FatalCaseInputError(
            f"{arm}: VLM arm requires raw response (str or bytes)")
    raw_sha = hashlib.sha256(
        raw.encode("utf-8") if isinstance(raw, str) else raw
    ).hexdigest()
    trust_id = context.trust_id()

    if seen_obj_ids is None:
        seen_obj_ids = set()

    # Parse (may raise)
    try:
        parsed = parse_agent_response(raw, context, seen_obj_ids=seen_obj_ids)
    except PromptPlanSchemaError as e:
        return _fallback_to_det(
            context, raw_sha, str(e), same_realization_mlem10,
            vlm_calls_used=1, medsam_calls_used=0)

    if parsed.abstain:
        return _fallback_to_det(
            context, raw_sha,
            f"abstain: {parsed.fallback_reason}",
            same_realization_mlem10,
            vlm_calls_used=1, medsam_calls_used=0)

    # Validate per-arm contract
    try:
        _validate_vlm_response(parsed, arm, context, seen_obj_ids)
    except PromptPlanSchemaError as e:
        return _fallback_to_det(
            context, raw_sha, str(e), same_realization_mlem10,
            vlm_calls_used=1, medsam_calls_used=0)

    # For arm 6: build decoder call specs from previous + correction
    decoder_specs = tuple()
    medsam_calls_used = 0
    if arm == "VLM_AGENT_REFINE" and previous_objects is not None:
        decoder_specs = tuple(_build_decoder_call_specs_impl(
            previous_objects, first_round_logits,
            parsed.objects, context))
        medsam_calls_used = len(decoder_specs)
    return ValidationOutcome(
        requested_arm=arm, executed_arm=arm, trust_id=trust_id,
        raw_response_sha256=raw_sha, fallback_used=False,
        fallback_reason="", objects=parsed.objects,
        decoder_call_specs=decoder_specs,
        vlm_calls_used=1, medsam_calls_used=medsam_calls_used,
    )


def _fallback_to_det(
    context: PromptRequestContext, raw_sha: str, reason: str,
    mlem10: np.ndarray, vlm_calls_used: int = 1,
    medsam_calls_used: int = 0,
) -> ValidationOutcome:
    """DET_COMPONENT deterministic fallback on any schema failure."""
    det_objs = det_component_objects(mlem10)
    return ValidationOutcome(
        requested_arm=context.requested_arm,
        executed_arm="DET_COMPONENT",
        trust_id=context.trust_id(),
        raw_response_sha256=raw_sha,
        fallback_used=True, fallback_reason=reason,
        objects=tuple(det_objs),
        decoder_call_specs=tuple(),
        vlm_calls_used=vlm_calls_used,
        medsam_calls_used=len(det_objs),
    )


# ============================================================================
# Deterministic builders
# ============================================================================

def _flood_fill_components(mask):
    h, w = mask.shape
    labels = np.zeros((h, w), dtype=np.int32)
    cur = 1
    for sy in range(h):
        for sx in range(w):
            if mask[sy, sx] and labels[sy, sx] == 0:
                stack = [(sy, sx)]
                while stack:
                    y, x = stack.pop()
                    if y < 0 or y >= h or x < 0 or x >= w:
                        continue
                    if not mask[y, x] or labels[y, x] != 0:
                        continue
                    labels[y, x] = cur
                    stack.extend([(y + 1, x), (y - 1, x),
                                  (y, x + 1), (y, x - 1)])
                cur += 1
    return labels


def det_component_objects(mlem10: np.ndarray) -> list:
    """Deterministic DET_COMPONENT planner.

    Per directive §10:
      - Threshold |mlem10| at quantile=0.85.
      - 4-connected flood-fill of foreground mask.
      - Per component: tight padded bbox (padding=2); peak inside
        component coordinates only (not bbox); background candidate
        (argmin) on the 1-px band OUTSIDE padded bbox; if no valid
        background, emit zero negatives.
      - Deterministic sort keys: (-pixel_count, y0, x0, y1, x1).
      - Internal object_ids 1..N.
      - Merge exact duplicate padded boxes deterministically.
      - Strict float dtype; reject nonfinite; reject all-zero (fatal).
    """
    if mlem10.ndim != 2 or mlem10.shape != (IMAGE_DIM, IMAGE_DIM):
        raise FatalCaseInputError(
            f"mlem10 must be {(IMAGE_DIM, IMAGE_DIM)}; got {mlem10.shape}")
    if mlem10.dtype.kind != "f":
        raise FatalCaseInputError(
            f"mlem10 must be floating; got {mlem10.dtype}")
    if not np.isfinite(mlem10).all():
        raise FatalCaseInputError("mlem10 must be fully finite")
    abs_map = np.abs(mlem10.astype(np.float32))
    thr = float(np.quantile(abs_map, 0.85))
    mask = (abs_map >= thr).astype(np.uint8)
    if mask.sum() == 0:
        raise FatalCaseInputError(
            "DET: empty foreground mask; no valid deterministic plan")
    labels = _flood_fill_components(mask)
    raw_components = []
    for k in range(1, int(labels.max()) + 1):
        ys, xs = np.where(labels == k)
        if len(xs) == 0:
            continue
        raw_components.append({
            "pixel_count": int(len(xs)),
            "x0_orig": int(xs.min()),
            "x1_orig": int(xs.max()) + 1,
            "y0_orig": int(ys.min()),
            "y1_orig": int(ys.max()) + 1,
            "ys": ys, "xs": xs,
        })
    # Merge by exact padded bbox (deterministic)
    seen_keys = {}
    for c in raw_components:
        xp0 = max(0, c["x0_orig"] - 2)
        yp0 = max(0, c["y0_orig"] - 2)
        xp1 = min(IMAGE_DIM, c["x1_orig"] + 2)
        yp1 = min(IMAGE_DIM, c["y1_orig"] + 2)
        c["xp0"], c["yp0"], c["xp1"], c["yp1"] = xp0, yp0, xp1, yp1
        k = (xp0, yp0, xp1, yp1)
        if k not in seen_keys:
            seen_keys[k] = c
        else:
            # Merge: sum pixel counts
            seen_keys[k]["pixel_count"] += c["pixel_count"]
            seen_keys[k]["ys"] = np.concatenate([seen_keys[k]["ys"], c["ys"]])
            seen_keys[k]["xs"] = np.concatenate([seen_keys[k]["xs"], c["xs"]])
    components = list(seen_keys.values())
    # Sort by (-pixel_count, y0, x0, y1, x1)
    components.sort(key=lambda c: (-c["pixel_count"], c["yp0"], c["xp0"],
                                   c["yp1"], c["xp1"]))
    components = components[:4]
    # Build PromptObjects with peak inside component coords
    objs = []
    for i, c in enumerate(components[:4]):
        # peak inside component only
        sub = abs_map[c["ys"], c["xs"]]
        i_max = int(np.argmax(sub))
        px = int(c["xs"][i_max])
        py = int(c["ys"][i_max])
        # Background candidates: 1-px band outside padded bbox
        bg_x0 = max(0, c["xp0"] - 1); bg_x1 = min(IMAGE_DIM, c["xp1"] + 1)
        bg_y0 = max(0, c["yp0"] - 1); bg_y1 = min(IMAGE_DIM, c["yp1"] + 1)
        band_mask = np.zeros_like(abs_map, dtype=bool)
        if c["yp0"] > 0:
            band_mask[bg_y0:c["yp0"], bg_x0:bg_x1] = True
        if c["yp1"] < IMAGE_DIM:
            band_mask[c["yp1"]:bg_y1, bg_x0:bg_x1] = True
        if c["xp0"] > 0:
            band_mask[bg_y0:bg_y1, bg_x0:c["xp0"]] = True
        if c["xp1"] < IMAGE_DIM:
            band_mask[bg_y0:bg_y1, c["xp1"]:bg_x1] = True
        negs = ()
        if band_mask.any():
            ys_b, xs_b = np.where(band_mask)
            abs_band = abs_map[ys_b, xs_b]
            i_min = int(np.argmin(abs_band))
            negs = ((int(xs_b[i_min]), int(ys_b[i_min])),)
        objs.append(PromptObject(
            object_id=i + 1,  # 1..N
            box_xyxy=(c["xp0"], c["yp0"], c["xp1"], c["yp1"]),
            positive_points_xy=((px, py),),
            negative_points_xy=negs,
            confidence=1.0,
        ))
    if not objs:
        raise FatalCaseInputError(
            "DET: no components after merge (fatal)")
    return objs


def full_box_objects() -> tuple:
    """FULL_BOX: exact [0, 0, 128, 128]. Always."""
    return (PromptObject(
        object_id=1, box_xyxy=(0, 0, IMAGE_DIM, IMAGE_DIM),
        positive_points_xy=(), negative_points_xy=(), confidence=1.0,
    ),)


# ============================================================================
# Aggregation
# ============================================================================

def pixelwise_max_aggregation(logit_arrays):
    """Per-object logits, same shape, float, finite. No min-max inside."""
    if len(logit_arrays) == 0:
        raise PromptPlanSchemaError(
            "pixelwise_max requires >= 1 logit array")
    ref = logit_arrays[0]
    if ref.dtype.kind != "f":
        raise PromptPlanSchemaError("logits must be float dtype")
    if not np.isfinite(ref).all():
        raise PromptPlanSchemaError("logits must be finite")
    for i, a in enumerate(logit_arrays):
        if a.shape != ref.shape:
            raise PromptPlanSchemaError(
                f"logits[{i}].shape {a.shape} != {ref.shape}")
        if a.dtype.kind != "f":
            raise PromptPlanSchemaError(f"logits[{i}] must be float")
        if not np.isfinite(a).all():
            raise PromptPlanSchemaError(f"logits[{i}] non-finite")
    return np.maximum.reduce(logit_arrays)


# ============================================================================
# Persistent transaction ledger
# ============================================================================

class TransactionBudget:
    """Keyed on (case identity + arm). Persists across call indices for
    the same case (one budget per case)."""

    _LEDGER: dict = {}

    def __init__(self, context: PromptRequestContext):
        self.context = context
        self._key = (context.dataset, context.subject_id, context.slice_id,
                     context.count_level, context.realization_id,
                     context.requested_arm)
        if self._key not in TransactionBudget._LEDGER:
            TransactionBudget._LEDGER[self._key] = {
                "vlm": 0, "medsam": 0, "executed_arm": None,
                "objects": [], "fallback_reason": "",
            }
        self._state = TransactionBudget._LEDGER[self._key]
        self._rules = ARM_MAXES[context.requested_arm]

    @property
    def vlm_calls(self):
        return self._state["vlm"]

    @property
    def medsam_calls(self):
        return self._state["medsam"]

    def check_call(self, kind):
        if kind == "vlm":
            n = self._state["vlm"] + 1
            if n > self._rules["max_vlm_calls"]:
                raise PromptPlanSchemaError(
                    f"{self.context.requested_arm} exceeds max_vlm_calls="
                    f"{self._rules['max_vlm_calls']} (call {n})")
            self._state["vlm"] = n
        elif kind == "medsam":
            n = self._state["medsam"] + 1
            if n > self._rules["max_medsam_calls"]:
                raise PromptPlanSchemaError(
                    f"{self.context.requested_arm} exceeds max_medsam_calls="
                    f"{self._rules['max_medsam_calls']} (call {n})")
            self._state["medsam"] = n
        else:
            raise ValueError(kind)

    def record_outcome(self, outcome):
        self._state["executed_arm"] = outcome.executed_arm
        self._state["objects"] = outcome.objects
        self._state["fallback_reason"] = outcome.fallback_reason


def full_box_strict_objects() -> tuple:
    """Alias for full_box_objects (kept for compatibility)."""
    return full_box_objects()