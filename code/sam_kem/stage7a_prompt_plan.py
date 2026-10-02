"""Stage 7A-v4.1 closure patch — corrected prompt-plan module.

Per directive 5932124321. Pure functions only. No VLM call, no
decoder training, no prior generation, no KEM execution.
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

ARM_MAXES = {
    "FULL_BOX": dict(
        min_objects=1, max_objects=1,
        min_positive_points=0, max_positive_points=0,
        min_negative_points=0, max_negative_points=0,
        vlm_calls=0, medsam_calls=1,
        allow_multi_round=False, max_calls_per_case=1,
        requires_negative_background=True,
    ),
    "DET_COMPONENT": dict(
        min_objects=1, max_objects=4,
        min_positive_points=1, max_positive_points=1,
        min_negative_points=0, max_negative_points=3,
        vlm_calls=0, medsam_calls=4,
        allow_multi_round=False, max_calls_per_case=1,
        requires_negative_background=True,
    ),
    "VLM_SINGLE_BOX": dict(
        min_objects=1, max_objects=1,
        min_positive_points=0, max_positive_points=0,
        min_negative_points=0, max_negative_points=0,
        vlm_calls=1, medsam_calls=1,
        allow_multi_round=False, max_calls_per_case=1,
    ),
    "VLM_MULTI_BOX": dict(
        min_objects=1, max_objects=4,
        min_positive_points=0, max_positive_points=0,
        min_negative_points=0, max_negative_points=0,
        vlm_calls=1, medsam_calls=4,
        allow_multi_round=False, max_calls_per_case=1,
    ),
    "VLM_BOX_POINT": dict(
        min_objects=1, max_objects=4,
        min_positive_points=1, max_positive_points=3,
        min_negative_points=0, max_negative_points=3,
        vlm_calls=1, medsam_calls=4,
        allow_multi_round=False, max_calls_per_case=1,
        requires_positive_per_object=True,
    ),
    "VLM_AGENT_REFINE": dict(
        min_objects=1, max_objects=4,
        min_positive_points=1, max_positive_points=3,
        min_negative_points=0, max_negative_points=3,
        vlm_calls=2, medsam_calls=8,
        allow_multi_round=True, max_calls_per_case=2,
        requires_positive_per_object=True,
    ),
}

FULL_VAL_MAX_VLM_TRANSACTIONS = (
    sum(ARM_MAXES[arm]["vlm_calls"] for arm in ARM_MAXES) * 8892
)

FULL_VAL_MAX_MEDSAM_PER_CASE = {
    arm: ARM_MAXES[arm]["medsam_calls"] for arm in ARM_MAXES
}

ALLOWED_OBJECT_KEYS = (
    "object_id", "box_xyxy", "positive_points_xy",
    "negative_points_xy", "confidence",
)

ALLOWED_TOP_KEYS = (
    "schema_version", "objects", "abstain", "fallback_reason",
)


# ============================================================================
# Errors
# ============================================================================

class PromptPlanSchemaError(ValueError):
    pass


class FatalCaseInputError(ValueError):
    pass


# ============================================================================
# Validators
# ============================================================================

def _validate_int(v, label):
    if isinstance(v, bool) or not isinstance(v, int):
        raise PromptPlanSchemaError(
            f"{label} must be int (no bool coercion); got {v!r}")
    return v


def _validate_point_xy(p, label):
    if not isinstance(p, list) or len(p) != 2:
        raise PromptPlanSchemaError(
            f"{label} must be a list of two ints [x, y]; got {p!r}")
    x, y = p
    _validate_int(x, f"{label}[0]")
    _validate_int(y, f"{label}[1]")
    if not (0 <= x < IMAGE_DIM):
        raise PromptPlanSchemaError(
            f"{label} x={x} out of range [0, {IMAGE_DIM})")
    if not (0 <= y < IMAGE_DIM):
        raise PromptPlanSchemaError(
            f"{label} y={y} out of range [0, {IMAGE_DIM})")
    return (x, y)


def _validate_box_xyxy(box, label):
    if not isinstance(box, list) or len(box) != 4:
        raise PromptPlanSchemaError(
            f"{label} must be a list of 4 ints [x0, y0, x1, y1]; got {box!r}")
    x0, y0, x1, y1 = box
    _validate_int(x0, f"{label}[0]")
    _validate_int(y0, f"{label}[1]")
    _validate_int(x1, f"{label}[2]")
    _validate_int(y1, f"{label}[3]")
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


def _check_duplicate_points(seen_pos, seen_neg, label, p, obj_id, idx):
    """Returns the validated tuple. Adds to the appropriate seen set based
    on label. Rejects duplicates within a sign and pos/neg overlap.
    Stores only validated tuples in the seen set (no raw lists)."""
    pt = _validate_point_xy(p, f"object {obj_id} {label}[{idx}]")
    is_positive = "positive" in label
    target = seen_pos if is_positive else seen_neg
    if pt in target:
        raise PromptPlanSchemaError(
            f"object {obj_id} duplicate {label} point {p} (index {idx})")
    if is_positive and pt in seen_neg:
        raise PromptPlanSchemaError(
            f"object {obj_id} positive_points_xy[{idx}]={p} duplicates a "
            f"negative_points_xy coordinate")
    if not is_positive and pt in seen_pos:
        raise PromptPlanSchemaError(
            f"object {obj_id} negative_points_xy[{idx}]={p} duplicates a "
            f"positive_points_xy coordinate")
    target.add(pt)
    return pt


# ============================================================================
# Models
# ============================================================================

@dataclass(frozen=True)
class PromptRequestContext:
    requested_arm: str
    dataset: str
    subject_id: str
    slice_id: str
    count_level: int
    realization_id: int
    call_index: int
    input_image_sha256: str
    parser_schema_version: int = SCHEMA_VERSION

    def trust_id(self) -> str:
        canonical = (
            f"{self.requested_arm}|{self.dataset}|{self.subject_id}|"
            f"{self.slice_id}|{self.count_level}|{self.realization_id}|"
            f"{self.call_index}|{self.input_image_sha256}|"
            f"{self.parser_schema_version}")
        return hashlib.sha256(canonical.encode()).hexdigest()

    def __post_init__(self):
        valid = tuple(ARM_MAXES.keys())
        if self.requested_arm not in valid:
            raise ValueError(f"requested_arm must be one of {valid}")
        if self.requested_arm != "VLM_AGENT_REFINE" and self.call_index > 1:
            raise ValueError(
                f"{self.requested_arm} cannot have call_index > 1")
        if self.requested_arm == "VLM_AGENT_REFINE" and self.call_index > 2:
            raise ValueError(
                "VLM_AGENT_REFINE cannot have call_index > 2")


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
# Ledger
# ============================================================================

class TransactionLedger:
    """Explicit ledger keyed by trusted case identity + arm. NOT a class
    global. Not resettable mid-case. Records through validate_or_fallback."""

    _REGISTRY: dict = {}

    def __init__(self, context: PromptRequestContext):
        self.context = context
        self._key = (
            context.dataset, context.subject_id, context.slice_id,
            context.count_level, context.realization_id,
            context.requested_arm,
            context.input_image_sha256, context.parser_schema_version)
        if self._key not in TransactionLedger._REGISTRY:
            TransactionLedger._REGISTRY[self._key] = {
                "vlm": 0, "medsam": 0, "objects": [],
                "fallback_reason": "", "decoder_call_specs": [],
            }
        self._state = TransactionLedger._REGISTRY[self._key]
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
            if n > self._rules["vlm_calls"]:
                raise PromptPlanSchemaError(
                    f"{self.context.requested_arm} exceeds max_vlm_calls="
                    f"{self._rules['vlm_calls']} (call {n})")
            self._state["vlm"] = n
        elif kind == "medsam":
            n = self._state["medsam"] + 1
            if n > self._rules["medsam_calls"]:
                raise PromptPlanSchemaError(
                    f"{self.context.requested_arm} exceeds max_medsam_calls="
                    f"{self._rules['medsam_calls']} (call {n})")
            self._state["medsam"] = n
        else:
            raise ValueError(kind)

    def record_outcome(self, outcome):
        self._state["objects"] = list(outcome.objects)
        self._state["fallback_reason"] = outcome.fallback_reason
        self._state["decoder_call_specs"] = list(outcome.decoder_call_specs)


# ============================================================================
# Parser
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


def _parse_object(obj_raw, idx, seen_obj_ids, boxes_seen,
                 seen_pos_across, seen_neg_across):
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
    oid = obj_raw["object_id"]
    _validate_int(oid, f"objects[{idx}].object_id")
    if oid <= 0:
        raise PromptPlanSchemaError(
            f"objects[{idx}].object_id must be positive; got {oid}")
    if oid in seen_obj_ids:
        raise PromptPlanSchemaError(
            f"objects[{idx}].object_id={oid} duplicates a previous object_id")
    seen_obj_ids.add(oid)
    box = _validate_box_xyxy(obj_raw["box_xyxy"], f"objects[{idx}].box_xyxy")
    # Duplicate box across objects
    for prev_oid, prev_box in boxes_seen:
        if prev_box == box:
            raise PromptPlanSchemaError(
                f"objects[{idx}] box duplicates object {prev_oid}")
    boxes_seen.append((oid, box))
    pos_raw = obj_raw["positive_points_xy"]
    neg_raw = obj_raw["negative_points_xy"]
    if not isinstance(pos_raw, list):
        raise PromptPlanSchemaError(
            f"object {oid} positive_points_xy must be a list")
    if not isinstance(neg_raw, list):
        raise PromptPlanSchemaError(
            f"object {oid} negative_points_xy must be a list")
    seen_pos = set()
    seen_neg = set()
    pos = []
    neg = []
    for j, p in enumerate(pos_raw):
        pt = _check_duplicate_points(seen_pos, seen_neg, "positive_points_xy",
                                    p, oid, j)
        if pt in seen_pos_across:
            raise PromptPlanSchemaError(
                f"object {oid} positive_points_xy[{j}]={p} duplicates a "
                f"point in another object")
        seen_pos_across.add(pt)
        pos.append(pt)
    for j, p in enumerate(neg_raw):
        pt = _check_duplicate_points(seen_pos, seen_neg, "negative_points_xy",
                                    p, oid, j)
        if pt in seen_neg_across:
            raise PromptPlanSchemaError(
                f"object {oid} negative_points_xy[{j}]={p} duplicates a "
                f"point in another object")
        seen_neg_across.add(pt)
        neg.append(pt)
    conf = obj_raw["confidence"]
    if isinstance(conf, bool):
        raise PromptPlanSchemaError(
            f"object {oid}.confidence must not be bool")
    if not isinstance(conf, (int, float)):
        raise PromptPlanSchemaError(
            f"object {oid}.confidence must be number")
    if conf != conf or conf in (float("inf"), float("-inf")):
        raise PromptPlanSchemaError(
            f"object {oid}.confidence non-finite")
    if not (0.0 <= float(conf) <= 1.0):
        raise PromptPlanSchemaError(
            f"object {oid}.confidence must be in [0, 1]")
    return PromptObject(
        object_id=oid, box_xyxy=box,
        positive_points_xy=tuple(pos),
        negative_points_xy=tuple(neg),
        confidence=float(conf),
    )


def parse_agent_response(raw, context, seen_obj_ids=None,
                         seen_boxes=None, seen_pos=None, seen_neg=None):
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
            "top-level JSON must be an object")
    extra = set(parsed.keys()) - set(ALLOWED_TOP_KEYS)
    if extra:
        raise PromptPlanSchemaError(
            f"top-level forbidden/unknown keys: {sorted(extra)}")
    missing = set(ALLOWED_TOP_KEYS) - set(parsed.keys())
    if missing:
        raise PromptPlanSchemaError(
            f"missing required top-level keys: {sorted(missing)}")
    if seen_obj_ids is None:
        seen_obj_ids = set()
    if seen_boxes is None:
        seen_boxes = []
    sv = parsed["schema_version"]
    _validate_int(sv, "schema_version")
    if sv != SCHEMA_VERSION:
        raise PromptPlanSchemaError(
            f"schema_version must be {SCHEMA_VERSION}; got {sv}")
    if sv != context.parser_schema_version:
        raise PromptPlanSchemaError(
            f"schema_version {sv} != context {context.parser_schema_version}")
    abstain = bool(parsed["abstain"])
    if abstain:
        objs = parsed["objects"]
        if not isinstance(objs, list) or len(objs) > 0:
            raise PromptPlanSchemaError(
                "abstain=true requires empty objects list")
        fr = parsed["fallback_reason"]
        if not isinstance(fr, str) or len(fr.strip()) == 0:
            raise PromptPlanSchemaError(
                "abstain=true requires nonempty string fallback_reason")
        return AgentResponse(sv, tuple(), True, str(fr), raw_sha)
    fr = parsed["fallback_reason"]
    if fr is not None:
        raise PromptPlanSchemaError(
            f"non-abstain requires fallback_reason == null; got {fr!r}")
    objs = parsed["objects"]
    if not isinstance(objs, list):
        raise PromptPlanSchemaError("objects must be a list")
    if seen_pos is None:
        seen_pos = set()
    if seen_neg is None:
        seen_neg = set()
    objects = []
    for i, obj in enumerate(objs):
        objects.append(_parse_object(obj, i, seen_obj_ids, seen_boxes,
                                    seen_pos, seen_neg))
    return AgentResponse(sv, tuple(objects), False, "", raw_sha)


# ============================================================================
# Public execution path
# ============================================================================

def _validate_case_input(mlem10, context):
    if mlem10 is None or not isinstance(mlem10, np.ndarray):
        raise FatalCaseInputError("same_realization_mlem10 must be ndarray")
    if mlem10.shape != (IMAGE_DIM, IMAGE_DIM):
        raise FatalCaseInputError(
            f"mlem10 must be {(IMAGE_DIM, IMAGE_DIM)}; got {mlem10.shape}")
    if not np.isfinite(mlem10).all():
        raise FatalCaseInputError("mlem10 must be fully finite")
    if mlem10.dtype.kind != "f":
        raise FatalCaseInputError(
            f"mlem10 must be floating dtype; got {mlem10.dtype}")
    if float(np.abs(mlem10).max()) <= 0.0:
        raise FatalCaseInputError(
            "mlem10 has no signal (all-zero); DET_COMPONENT cannot produce plan")


def _validate_vlm_response(parsed, arm, context, ledger):
    rules = ARM_MAXES[arm]
    n = len(parsed.objects)
    if not (rules["min_objects"] <= n <= rules["max_objects"]):
        raise PromptPlanSchemaError(
            f"{arm}: object count {n} out of range "
            f"[{rules['min_objects']}, {rules['max_objects']}]")
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
        if len(obj.negative_points_xy) < rules["min_negative_points"]:
            raise PromptPlanSchemaError(
                f"{arm}: object {obj.object_id} too few negative_points")
    ledger.check_call("vlm")
    if not rules.get("allow_multi_round", False):
        ledger.check_call("medsam")
    return True


def package_refinement_inputs(
    first_objects, first_round_logits,
    correction_objects, context,
):
    """Per-object DecoderCallSpec; mask_input is that object's first-round
    logit. Permits new correction points (not constrained to first-round
    subset)."""
    if len(first_objects) != len(first_round_logits):
        raise PromptPlanSchemaError("first_round_objects vs logits mismatch")
    if len(first_objects) != len(correction_objects):
        raise PromptPlanSchemaError(
            "correction_objects vs first_round_objects mismatch")
    by_id = {o.object_id: (o, fl) for o, fl in
             zip(first_objects, first_round_logits)}
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
                f"first_round_logits[{co.object_id}] must be float")
        if not np.isfinite(fl).all():
            raise PromptPlanSchemaError(
                f"first_round_logits[{co.object_id}] non-finite")
        if fo.box_xyxy != co.box_xyxy:
            raise PromptPlanSchemaError(
                f"object_id {co.object_id}: box changed across rounds")
        specs.append(DecoderCallSpec(
            object_id=co.object_id,
            box_xyxy=co.box_xyxy,
            positive_points_xy=co.positive_points_xy,
            negative_points_xy=co.negative_points_xy,
            mask_input=fl,
        ))
    return specs


def validate_or_fallback(
    raw, context, same_realization_mlem10,
    first_round_objects=None, first_round_logits=None,
    ledger=None,
):
    """Single executable entry point. Uses supplied ledger if any."""
    _validate_case_input(same_realization_mlem10, context)
    arm = context.requested_arm
    if ledger is None:
        ledger = TransactionLedger(context)

    if arm in ("FULL_BOX", "DET_COMPONENT"):
        if raw is not None:
            raise FatalCaseInputError(
                f"{arm} must have raw=None; got {type(raw).__name__}")
        if arm == "FULL_BOX":
            objs = full_box_objects()
            medsam_used = rules_medsam(arm)
        else:
            objs = tuple(det_component_objects(same_realization_mlem10))
            medsam_used = len(objs)
        ledger.check_call("medsam")
        return ValidationOutcome(
            requested_arm=arm, executed_arm=arm,
            trust_id=context.trust_id(),
            raw_response_sha256="",
            fallback_used=False, fallback_reason="",
            objects=tuple(objs), decoder_call_specs=tuple(),
            vlm_calls_used=0, medsam_calls_used=medsam_used,
        )

    if raw is None:
        raise FatalCaseInputError(f"{arm}: VLM arm requires raw response")
    raw_sha = hashlib.sha256(
        raw.encode("utf-8") if isinstance(raw, str) else raw
    ).hexdigest()

    try:
        parsed = parse_agent_response(raw, context)
    except PromptPlanSchemaError as e:
        return _fallback(context, raw_sha, str(e), same_realization_mlem10,
                        ledger, vlm_used=True)

    if parsed.abstain:
        return _fallback(context, raw_sha,
                        f"abstain: {parsed.fallback_reason}",
                        same_realization_mlem10,
                        ledger, vlm_used=True)

    try:
        _validate_vlm_response(parsed, arm, context, ledger)
    except PromptPlanSchemaError as e:
        return _fallback(context, raw_sha, str(e),
                        same_realization_mlem10,
                        ledger, vlm_used=True)

    decoder_specs = tuple()
    medsam_calls_used = 0
    if arm == "VLM_AGENT_REFINE" and first_round_objects is not None:
        decoder_specs = tuple(package_refinement_inputs(
            first_round_objects, first_round_logits,
            parsed.objects, context))
        medsam_calls_used = len(decoder_specs)
        ledger.check_call("medsam")
    return ValidationOutcome(
        requested_arm=arm, executed_arm=arm, trust_id=context.trust_id(),
        raw_response_sha256=raw_sha, fallback_used=False,
        fallback_reason="", objects=parsed.objects,
        decoder_call_specs=decoder_specs,
        vlm_calls_used=1, medsam_calls_used=medsam_calls_used,
    )


def rules_medsam(arm):
    return ARM_MAXES[arm]["medsam_calls"]


def _fallback(context, raw_sha, reason, mlem10, ledger,
              vlm_used=False):
    det_objs = det_component_objects(mlem10)
    return ValidationOutcome(
        requested_arm=context.requested_arm,
        executed_arm="DET_COMPONENT",
        trust_id=context.trust_id(),
        raw_response_sha256=raw_sha,
        fallback_used=True, fallback_reason=reason,
        objects=tuple(det_objs),
        decoder_call_specs=tuple(),
        vlm_calls_used=1 if vlm_used else 0,
        medsam_calls_used=len(det_objs),
    )


# ============================================================================
# Builders
# ============================================================================

def full_box_objects():
    return (PromptObject(
        object_id=1, box_xyxy=(0, 0, IMAGE_DIM, IMAGE_DIM),
        positive_points_xy=(), negative_points_xy=(), confidence=1.0),)


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


def det_component_objects(mlem10):
    if mlem10.ndim != 2 or mlem10.shape != (IMAGE_DIM, IMAGE_DIM):
        raise FatalCaseInputError(
            f"mlem10 must be {(IMAGE_DIM, IMAGE_DIM)}; got {mlem10.shape}")
    if mlem10.dtype.kind != "f":
        raise FatalCaseInputError(
            f"mlem10 must be floating; got {mlem10.dtype}")
    if not np.isfinite(mlem10).all():
        raise FatalCaseInputError("mlem10 must be fully finite")
    if float(np.abs(mlem10.astype(np.float32)).max()) <= 0.0:
        raise FatalCaseInputError(
            "DET: zero-signal mlem (all-zero); fail-fast, no silent empty")
    abs_map = np.abs(mlem10.astype(np.float32))
    thr = float(np.quantile(abs_map, 0.85))
    mask = (abs_map >= thr).astype(np.uint8)
    if mask.sum() == 0:
        raise FatalCaseInputError("DET: empty foreground; fatal")
    labels = _flood_fill_components(mask)
    raw_components = []
    for k in range(1, int(labels.max()) + 1):
        ys, xs = np.where(labels == k)
        if len(xs) == 0:
            continue
        raw_components.append({
            "pixel_count": int(len(xs)),
            "x0_orig": int(xs.min()), "x1_orig": int(xs.max()) + 1,
            "y0_orig": int(ys.min()), "y1_orig": int(ys.max()) + 1,
            "ys": ys, "xs": xs,
        })
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
            seen_keys[k]["pixel_count"] += c["pixel_count"]
            seen_keys[k]["ys"] = np.concatenate([seen_keys[k]["ys"], c["ys"]])
            seen_keys[k]["xs"] = np.concatenate([seen_keys[k]["xs"], c["xs"]])
    components = list(seen_keys.values())
    components.sort(key=lambda c: (-c["pixel_count"], c["yp0"], c["xp0"],
                                   c["yp1"], c["xp1"]))
    components = components[:4]
    objs = []
    for i, c in enumerate(components[:4]):
        sub = abs_map[c["ys"], c["xs"]]
        i_max = int(np.argmax(sub))
        px = int(c["xs"][i_max])
        py = int(c["ys"][i_max])
        # Background candidate: 1-px band outside padded box,
        # masked to labels == 0 (background only).
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
        # CRITICAL: mask to background (labels == 0)
        band_mask &= (labels == 0)
        negs = ()
        if band_mask.any():
            ys_b, xs_b = np.where(band_mask)
            abs_band = abs_map[ys_b, xs_b]
            i_min = int(np.argmin(abs_band))
            negs = ((int(xs_b[i_min]), int(ys_b[i_min])),)
        objs.append(PromptObject(
            object_id=i + 1,
            box_xyxy=(c["xp0"], c["yp0"], c["xp1"], c["yp1"]),
            positive_points_xy=((px, py),),
            negative_points_xy=negs,
            confidence=1.0,
        ))
    if not objs:
        raise FatalCaseInputError("DET: no components (fatal)")
    return objs


# ============================================================================
# Aggregation
# ============================================================================

def pixelwise_max_aggregation(logit_arrays):
    if len(logit_arrays) == 0:
        raise PromptPlanSchemaError(
            "pixelwise_max requires >= 1 logit array")
    ref = logit_arrays[0]
    if ref.dtype.kind != "f":
        raise PromptPlanSchemaError("logits must be float")
    if not np.isfinite(ref).all():
        raise PromptPlanSchemaError("logits must be finite")
    for i, a in enumerate(logit_arrays):
        if a.shape != ref.shape:
            raise PromptPlanSchemaError(f"logits[{i}].shape mismatch")
        if a.dtype.kind != "f":
            raise PromptPlanSchemaError(f"logits[{i}] must be float")
        if not np.isfinite(a).all():
            raise PromptPlanSchemaError(f"logits[{i}] non-finite")
    return np.maximum.reduce(logit_arrays)