"""Stage 7A-v3 — corrected prompt-plan with strict trusted-context
isolation from untrusted VLM JSON, single validate-or-fallback entry
point, correct DET_COMPONENT (pixel-count ranking), correct
refinement semantics (no averaging), and full adversarial test surface.

Per directive 5927858004. Pure functions only (no I/O, no GPU,
no VLM, no decoder). The agent JSON contract is enforced strictly;
trusted identity (dataset, subject, slice, count, realization,
arm) is held in an immutable PromptRequestContext and cannot be
spoofed by VLM output.
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

# Per-object ALLOWED keys (exact match; any other key raises).
ALLOWED_OBJECT_KEYS = (
    "object_id", "box_xyxy", "positive_points_xy",
    "negative_points_xy", "confidence",
)

# Top-level ALLOWED keys in the raw agent response.
ALLOWED_TOP_KEYS = (
    "schema_version", "objects", "abstain", "fallback_reason",
)

# Top-level REQUIRED keys in the raw agent response.
REQUIRED_TOP_KEYS = ALLOWED_TOP_KEYS


# ============================================================================
# Trusted execution context (immutable; NEVER derived from agent JSON)
# ============================================================================

@dataclass(frozen=True)
class PromptRequestContext:
    """Trusted execution identity. Built by the trusted caller; cannot
    be set or modified by the VLM response. All validation routes
    required to use this exact context."""
    requested_arm: str            # FULL_BOX | DET_COMPONENT | VLM_SINGLE_BOX | VLM_MULTI_BOX | VLM_BOX_POINT | VLM_AGENT_REFINE
    dataset: str
    subject_id: str
    slice_id: str
    count_level: int
    realization_id: int
    call_index: int              # 1 for first call, 2 for refinement

    def __post_init__(self):
        # Validate arm enum
        valid_arms = ("FULL_BOX", "DET_COMPONENT", "VLM_SINGLE_BOX",
                      "VLM_MULTI_BOX", "VLM_BOX_POINT", "VLM_AGENT_REFINE")
        if self.requested_arm not in valid_arms:
            raise ValueError(f"requested_arm must be one of {valid_arms}; "
                              f"got {self.requested_arm!r}")
        # arm + call_index consistency
        if self.requested_arm != "VLM_AGENT_REFINE" and self.call_index > 1:
            raise ValueError(
                f"arm {self.requested_arm} cannot have call_index > 1")
        if self.requested_arm == "VLM_AGENT_REFINE" and self.call_index > 2:
            raise ValueError(
                "VLM_AGENT_REFINE cannot have call_index > 2")


# ============================================================================
# Object model (strict)
# ============================================================================

@dataclass(frozen=True)
class PromptObject:
    """One arm output object. Strict: only integer coords and finite
    confidence in [0, 1]. Per-object, no per-arm special cases."""
    object_id: int
    box_xyxy: tuple[int, int, int, int]    # x0, y0, x1, y1
    positive_points_xy: tuple[tuple[int, int], ...]
    negative_points_xy: tuple[tuple[int, int], ...]
    confidence: float

    def __post_init__(self):
        if isinstance(self.box_xyxy, bool) or not isinstance(
                self.box_xyxy, (tuple, list)) or len(self.box_xyxy) != 4:
            raise PromptPlanSchemaError(
                f"box_xyxy must be tuple/list of 4 ints; got {self.box_xyxy!r}")
        x0, y0, x1, y1 = self.box_xyxy
        for k, v in (("x0", x0), ("y0", y0), ("x1", x1), ("y1", y1)):
            if isinstance(v, bool) or not isinstance(v, int):
                raise PromptPlanSchemaError(
                    f"box_xyxy[{k}] must be int (no coercion); got {v!r}")
        if not (0 <= x0 < x1 <= IMAGE_DIM):
            raise PromptPlanSchemaError(
                f"box_xyxy ({x0},{y0},{x1},{y1}) must satisfy "
                f"0 <= x0 < x1 <= {IMAGE_DIM}")
        if not (0 <= y0 < y1 <= IMAGE_DIM):
            raise PromptPlanSchemaError(
                f"box_xyxy ({x0},{y0},{x1},{y1}) must satisfy "
                f"0 <= y0 < y1 <= {IMAGE_DIM}")
        for pts, kind in ((self.positive_points_xy, "positive"),
                          (self.negative_points_xy, "negative")):
            for p in pts:
                if (not isinstance(p, (tuple, list))) or len(p) != 2:
                    raise PromptPlanSchemaError(
                        f"{kind} point must be [x, y]; got {p!r}")
                if isinstance(p[0], bool) or not isinstance(p[0], int):
                    raise PromptPlanSchemaError(
                        f"{kind} point x must be int; got {p[0]!r}")
                if isinstance(p[1], bool) or not isinstance(p[1], int):
                    raise PromptPlanSchemaError(
                        f"{kind} point y must be int; got {p[1]!r}")
                if not (0 <= p[0] < IMAGE_DIM):
                    raise PromptPlanSchemaError(
                        f"{kind} point x={p[0]} out of range [0, {IMAGE_DIM})")
                if not (0 <= p[1] < IMAGE_DIM):
                    raise PromptPlanSchemaError(
                        f"{kind} point y={p[1]} out of range [0, {IMAGE_DIM})")
        if not isinstance(self.confidence, (int, float)):
            raise PromptPlanSchemaError(
                f"confidence must be number; got {type(self.confidence).__name__}")
        if isinstance(self.confidence, bool):
            raise PromptPlanSchemaError(
                f"confidence must not be bool; got {self.confidence!r}")
        if not (0.0 <= self.confidence <= 1.0):
            raise PromptPlanSchemaError(
                f"confidence must be in [0, 1]; got {self.confidence}")
        if self.confidence != self.confidence or self.confidence in (
                float("inf"), float("-inf")):
            raise PromptPlanSchemaError(
                f"confidence must be finite; got {self.confidence}")


@dataclass(frozen=True)
class AgentResponse:
    """Strict validated raw agent response."""
    schema_version: int
    objects: tuple[PromptObject, ...]
    abstain: bool
    fallback_reason: str  # "" if not abstaining
    raw_response_sha256: str


# ============================================================================
# Errors
# ============================================================================

class PromptPlanSchemaError(ValueError):
    pass


class FatalCaseInputError(ValueError):
    """Raised when the case input itself is invalid (nonfinite mlem,
    wrong shape, etc.). NEVER silently replaced by a fallback."""


# ============================================================================
# Strict raw-JSON validator
# ============================================================================

def _check_no_duplicate_keys(pairs):
    """object_pairs_hook for json.loads."""
    seen = {}
    out = OrderedDict()
    for k, v in pairs:
        if k in seen:
            raise PromptPlanSchemaError(f"duplicate JSON key: {k!r}")
        seen[k] = True
        out[k] = v
    return out


def _validate_xy(point_obj, label):
    if not isinstance(point_obj, list) or len(point_obj) != 2:
        raise PromptPlanSchemaError(
            f"{label} must be a list of two ints [x, y]; got {point_obj!r}")
    if (isinstance(point_obj[0], bool) or not isinstance(point_obj[0], int) or
            isinstance(point_obj[1], bool) or not isinstance(point_obj[1], int)):
        raise PromptPlanSchemaError(
            f"{label} coords must be int (no coercion); got {point_obj!r}")
    x, y = point_obj
    if not (0 <= x < IMAGE_DIM):
        raise PromptPlanSchemaError(
            f"{label} x={x} out of range [0, {IMAGE_DIM})")
    if not (0 <= y < IMAGE_DIM):
        raise PromptPlanSchemaError(
            f"{label} y={y} out of range [0, {IMAGE_DIM})")
    return (x, y)


def _parse_object(obj_raw, idx, seen_obj_ids):
    """Parse one object from raw dict into a PromptObject."""
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
    if isinstance(oid, bool) or not isinstance(oid, int):
        raise PromptPlanSchemaError(
            f"objects[{idx}].object_id must be int; got {oid!r}")
    if oid <= 0:
        raise PromptPlanSchemaError(
            f"objects[{idx}].object_id must be positive; got {oid}")
    if oid in seen_obj_ids:
        raise PromptPlanSchemaError(
            f"objects[{idx}].object_id={oid} duplicates a previous object_id")
    seen_obj_ids.add(oid)
    # box_xyxy list of 4 ints
    box_raw = obj_raw["box_xyxy"]
    if not isinstance(box_raw, list) or len(box_raw) != 4:
        raise PromptPlanSchemaError(
            f"objects[{idx}].box_xyxy must be a list of 4 ints; got {box_raw!r}")
    box = tuple(_validate_xy([box_raw[0], box_raw[1]], f"objects[{idx}].box_xyxy[0:2]") +
               _validate_xy([box_raw[2], box_raw[3]], f"objects[{idx}].box_xyxy[2:4]"))
    # points
    pos = tuple(_validate_xy(p, f"objects[{idx}].positive_points_xy[{i}]")
                for i, p in enumerate(obj_raw["positive_points_xy"]))
    neg = tuple(_validate_xy(p, f"objects[{idx}].negative_points_xy[{i}]")
                for i, p in enumerate(obj_raw["negative_points_xy"]))
    # confidence
    conf = obj_raw["confidence"]
    if isinstance(conf, bool):
        raise PromptPlanSchemaError(
            f"objects[{idx}].confidence must not be bool; got {conf!r}")
    if not isinstance(conf, (int, float)):
        raise PromptPlanSchemaError(
            f"objects[{idx}].confidence must be number; got {type(conf).__name__}")
    if conf != conf or conf in (float("inf"), float("-inf")):
        raise PromptPlanSchemaError(
            f"objects[{idx}].confidence must be finite; got {conf!r}")
    if not (0.0 <= float(conf) <= 1.0):
        raise PromptPlanSchemaError(
            f"objects[{idx}].confidence must be in [0, 1]; got {conf}")
    return PromptObject(
        object_id=oid,
        box_xyxy=box,
        positive_points_xy=pos,
        negative_points_xy=neg,
        confidence=float(conf),
    )


def parse_agent_response(raw: str | bytes, context: PromptRequestContext) -> AgentResponse:
    """Strict raw-JSON validator. NEVER reads case identity from raw JSON
    (it comes from context). NEVER silently repairs bad data."""
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
    # raw response SHA (independent of validation outcome)
    raw_sha = hashlib.sha256(raw if isinstance(raw, bytes)
                              else raw.encode("utf-8")).hexdigest()
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
    missing = set(REQUIRED_TOP_KEYS) - set(parsed.keys())
    if missing:
        raise PromptPlanSchemaError(f"missing top-level keys: {sorted(missing)}")
    # schema_version must be int (no coercion)
    sv = parsed["schema_version"]
    if isinstance(sv, bool) or not isinstance(sv, int):
        raise PromptPlanSchemaError(
            f"schema_version must be int; got {type(sv).__name__}")
    if sv != SCHEMA_VERSION:
        raise PromptPlanSchemaError(
            f"schema_version must be {SCHEMA_VERSION}; got {sv}")
    abstain_raw = parsed["abstain"]
    if not isinstance(abstain_raw, bool):
        raise PromptPlanSchemaError(
            f"abstain must be bool; got {type(abstain_raw).__name__}")
    abstain = bool(abstain_raw)
    if abstain:
        # abstention contract: objects must be empty, fallback_reason nonempty
        objs_raw = parsed["objects"]
        if not isinstance(objs_raw, list) or len(objs_raw) > 0:
            raise PromptPlanSchemaError(
                "abstain=true requires empty objects list")
        fr_raw = parsed["fallback_reason"]
        if not isinstance(fr_raw, str) or len(fr_raw.strip()) == 0:
            raise PromptPlanSchemaError(
                "abstain=true requires nonempty string fallback_reason")
        return AgentResponse(
            schema_version=sv,
            objects=tuple(),
            abstain=True,
            fallback_reason=str(fr_raw),
            raw_response_sha256=raw_sha,
        )
    # Non-abstain: fallback_reason must be JSON null (= None)
    fr_raw = parsed["fallback_reason"]
    if fr_raw is not None:
        raise PromptPlanSchemaError(
            f"non-abstain requires fallback_reason == null; got {fr_raw!r}")
    objs_raw = parsed["objects"]
    if not isinstance(objs_raw, list):
        raise PromptPlanSchemaError("objects must be a list")
    objects = []
    seen = set()
    for i, obj_raw in enumerate(objs_raw):
        objects.append(_parse_object(obj_raw, i, seen))
    return AgentResponse(
        schema_version=sv,
        objects=tuple(objects),
        abstain=False,
        fallback_reason="",
        raw_response_sha256=raw_sha,
    )


# ============================================================================
# Single executable validate-or-fallback entry point
# ============================================================================

@dataclass(frozen=True)
class ValidationOutcome:
    requested_arm: str
    executed_arm: str
    trust_id: str                  # canonical hash of context identity
    raw_response_sha256: str       # "" if not parsed
    fallback_used: bool
    fallback_reason: str
    objects: tuple[PromptObject, ...]


def validate_or_fallback(
    raw: str | bytes | None,
    context: PromptRequestContext,
    same_realization_mlem10: np.ndarray | None,
) -> ValidationOutcome:
    """The single public entry point. Returns a typed outcome that binds
    requested arm, executed arm, trust identity, fallback flag/reason,
    raw-response SHA, and a valid plan.

    abstention, invalid input, or non-finite mlem all flow through here.
    The trusted case identity (dataset/subject/slice/count/realization)
    comes from `context` only, never from raw JSON.
    """
    trust_id = hashlib.sha256(
        f"{context.dataset}|{context.subject_id}|{context.slice_id}|"
        f"{context.count_level}|{context.realization_id}".encode()
    ).hexdigest()

    if same_realization_mlem10 is None or not isinstance(
            same_realization_mlem10, np.ndarray):
        raise FatalCaseInputError(
            "same_realization_mlem10 is required and must be ndarray")
    if same_realization_mlem10.shape != (IMAGE_DIM, IMAGE_DIM):
        raise FatalCaseInputError(
            f"same_realization_mlem10 must be {(IMAGE_DIM, IMAGE_DIM)}; "
            f"got {same_realization_mlem10.shape}")
    if not np.isfinite(same_realization_mlem10).all():
        raise FatalCaseInputError(
            "same_realization_mlem10 must be fully finite")
    # Detect trivial (constant-zero / no-signal) input
    if float(np.abs(same_realization_mlem10).max()) <= 0.0:
        raise FatalCaseInputError(
            "same_realization_mlem10 has no signal (all-zero); "
            "DET_COMPONENT cannot produce a deterministic plan")
    if same_realization_mlem10.dtype.kind != "f":
        raise FatalCaseInputError(
            f"same_realization_mlem10 must be floating dtype; got "
            f"{same_realization_mlem10.dtype}")

    raw_sha = hashlib.sha256(
        raw.encode("utf-8") if isinstance(raw, str) else raw
    ).hexdigest()

    # Parse the raw response (may raise PromptPlanSchemaError)
    try:
        parsed = parse_agent_response(raw, context)
    except PromptPlanSchemaError as e:
        return _fallback_to_det(context, trust_id, raw, str(e),
                                same_realization_mlem10,
                                parsed_response_sha=raw_sha)

    arm = context.requested_arm

    # Deterministic arms: never parse a response; build directly.
    if arm in ("FULL_BOX", "DET_COMPONENT"):
        objs = (full_box_objects()
                if arm == "FULL_BOX"
                else tuple(det_component_objects(same_realization_mlem10)))
        return ValidationOutcome(
            requested_arm=arm, executed_arm=arm, trust_id=trust_id,
            raw_response_sha256=raw_sha,
            fallback_used=False, fallback_reason="",
            objects=tuple(objs),
        )

    # VLM arms: abstention → fallback to DET (per directive §A.4).
    if parsed.abstain:
        return _fallback_to_det(
            context, trust_id, raw,
            f"abstain: {parsed.fallback_reason}",
            same_realization_mlem10,
            parsed_response_sha=raw_sha,
        )

    # Validate against the requested arm contract
    ok = _validate_vlm_response(parsed, arm, context)
    if ok is None:
        return _fallback_to_det(context, trust_id, raw,
                                f"{arm}_validation_failed",
                                same_realization_mlem10,
                                parsed_response_sha=raw_sha)
    return ValidationOutcome(
        requested_arm=arm, executed_arm=arm, trust_id=trust_id,
        raw_response_sha256=raw_sha,
        fallback_used=False, fallback_reason="",
        objects=parsed.objects,
    )


def _fallback_to_det(
    context: PromptRequestContext,
    trust_id: str,
    raw,
    reason: str,
    same_realization_mlem10: np.ndarray,
    parsed_response_sha: str = "",
) -> ValidationOutcome:
    """abstention / invalid input -> deterministic DET_COMPONENT fallback."""
    try:
        det_objs = det_component_objects(same_realization_mlem10)
    except FatalCaseInputError:
        # Re-raise if even DET_COMPONENT cannot run (e.g. zero-all image).
        raise
    raw_sha = ""
    if raw is not None:
        try:
            raw_sha = hashlib.sha256(raw if isinstance(raw, bytes)
                                      else raw.encode("utf-8")).hexdigest()
        except Exception:
            raw_sha = ""
    return ValidationOutcome(
        requested_arm=context.requested_arm,
        executed_arm="DET_COMPONENT",
        trust_id=trust_id,
        raw_response_sha256=raw_sha or parsed_response_sha,
        fallback_used=True,
        fallback_reason=reason,
        objects=tuple(det_objs),
    )


def _validate_vlm_response(parsed: AgentResponse, arm: str,
                            context: PromptRequestContext):
    """Enforce per-VLM-arm contract on the parsed response. Return None
    if any contract is violated (caller should fallback to DET_COMPONENT)."""
    rules = {
        "VLM_SINGLE_BOX": dict(min_objects=1, max_objects=1,
                               min_positive_points=0, max_positive_points=0,
                               min_negative_points=0, max_negative_points=0),
        "VLM_MULTI_BOX": dict(min_objects=1, max_objects=4,
                              min_positive_points=0, max_positive_points=0,
                              min_negative_points=0, max_negative_points=0),
        "VLM_BOX_POINT": dict(min_objects=1, max_objects=4,
                               min_positive_points=1, max_positive_points=3,
                               min_negative_points=0, max_negative_points=3),
        # VLM_AGENT_REFINE has its own refinement path; not validated here.
    }
    if arm not in rules:
        return None
    r = rules[arm]
    n = len(parsed.objects)
    if not (r["min_objects"] <= n <= r["max_objects"]):
        return None
    for obj in parsed.objects:
        if len(obj.positive_points_xy) > r["max_positive_points"]:
            return None
        if len(obj.positive_points_xy) < r["min_positive_points"]:
            return None
        if len(obj.negative_points_xy) > r["max_negative_points"]:
            return None
        if len(obj.negative_points_xy) < r["min_negative_points"]:
            return None
    return True


# ============================================================================
# DET_COMPONENT deterministic planner (corrected per directive §10)
# ============================================================================

def _flood_fill_components(mask: np.ndarray) -> np.ndarray:
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
                    stack.extend([(y + 1, x), (y - 1, x), (y, x + 1), (y, x - 1)])
                cur += 1
    return labels


def det_component_objects(mlem10: np.ndarray) -> list[PromptObject]:
    """Corrected DET_COMPONENT planner:
      - Ranks components by ACTUAL PIXEL COUNT (component area in pixels),
        not bounding-box area.
      - Local peak is INSIDE the component (not just bbox).
      - Background negatives are OUTSIDE the bbox (1-px band ring).
      - Reject non-finite / wrong-shaped / all-zero / all-NaN inputs via
        FatalCaseInputError (no silent empty fallback for valid finite inputs).
      - If no valid background location exists, emit zero negatives for
        that component.
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
            "DET_COMPONENT: empty foreground mask after threshold; "
            "no valid deterministic plan")
    labels = _flood_fill_components(mask)
    components = []
    for k in range(1, int(labels.max()) + 1):
        ys, xs = np.where(labels == k)
        if len(xs) == 0:
            continue
        pixel_count = int(len(xs))
        x0 = int(xs.min()); x1 = int(xs.max()) + 1
        y0 = int(ys.min()); y1 = int(ys.max()) + 1
        # pad to bbox
        xp0 = max(0, x0 - 2); yp0 = max(0, y0 - 2)
        xp1 = min(IMAGE_DIM, x1 + 2); yp1 = min(IMAGE_DIM, y1 + 2)
        if xp0 >= xp1 or yp0 >= yp1:
            continue
        # local peak INSIDE the component (not the bbox)
        sub = abs_map[y0:y1, x0:x1]
        py, px = np.unravel_index(np.argmax(sub), sub.shape)
        peak = (int(x0 + px), int(y0 + py))
        # background negative: argmin outside padded bbox
        bg_x0 = max(0, xp0 - 1); bg_x1 = min(IMAGE_DIM, xp1 + 1)
        bg_y0 = max(0, yp0 - 1); bg_y1 = min(IMAGE_DIM, yp1 + 1)
        band_mask = np.zeros_like(abs_map, dtype=bool)
        if yp0 > 0:
            band_mask[bg_y0:yp0, bg_x0:bg_x1] = True
        if yp1 < IMAGE_DIM:
            band_mask[yp1:bg_y1, bg_x0:bg_x1] = True
        if xp0 > 0:
            band_mask[bg_y0:bg_y1, bg_x0:xp0] = True
        if xp1 < IMAGE_DIM:
            band_mask[bg_y0:bg_y1, xp1:bg_x1] = True
        if not band_mask.any():
            # No valid background location; emit zero negatives (NOT
            # an inside point). Per directive.
            negs = ()
        else:
            ys_b, xs_b = np.where(band_mask)
            abs_band = abs_map[ys_b, xs_b]
            i_min = int(np.argmin(abs_band))
            neg = (int(xs_b[i_min]), int(ys_b[i_min]))
            negs = (neg,)
        components.append((pixel_count, PromptObject(
            object_id=len(components),
            box_xyxy=(xp0, yp0, xp1, yp1),
            positive_points_xy=(peak,),
            negative_points_xy=negs,
            confidence=1.0,
        )))
    if not components:
        raise FatalCaseInputError(
            "DET_COMPONENT: no valid components produced")
    # Deterministic tie-break: rank by pixel_count desc, then by
    # smaller x0+y0 asc, then by larger padded area desc.
    components.sort(key=lambda t: (-t[0], t[1].box_xyxy[0] + t[1].box_xyxy[1],
                                  -((t[1].box_xyxy[2] - t[1].box_xyxy[0]) *
                                    (t[1].box_xyxy[3] - t[1].box_xyxy[1]))))
    components = components[:4]
    # Re-id after sorting/drop
    return [
        PromptObject(
            object_id=i,
            box_xyxy=o.box_xyxy,
            positive_points_xy=o.positive_points_xy,
            negative_points_xy=o.negative_points_xy,
            confidence=o.confidence,
        ) for i, (_, o) in enumerate(components)
    ]


# ============================================================================
# Deterministic builders (no parse, no VLM)
# ============================================================================

def full_box_objects() -> tuple[PromptObject, ...]:
    """FULL_BOX: exact [0, 0, 128, 128] full image. No parse, no VLM."""
    return (PromptObject(
        object_id=0,
        box_xyxy=(0, 0, IMAGE_DIM, IMAGE_DIM),
        positive_points_xy=(),
        negative_points_xy=(),
        confidence=1.0,
    ),)


# ============================================================================
# Aggregation: pixelwise per-object max (NO numerical averaging)
# ============================================================================

def pixelwise_max_aggregation(logit_arrays: list[np.ndarray]) -> np.ndarray:
    """Validate same shape, floating dtype, all finite, nonempty;
    return np.maximum.reduce(...). NO min-max inside; NO averaging."""
    if len(logit_arrays) == 0:
        raise PromptPlanSchemaError(
            "pixelwise_max_aggregation requires >= 1 logit array")
    ref_shape = logit_arrays[0].shape
    for i, a in enumerate(logit_arrays):
        if a.shape != ref_shape:
            raise PromptPlanSchemaError(
                f"logits[{i}].shape {a.shape} != reference {ref_shape}")
        if a.dtype.kind != "f":
            raise PromptPlanSchemaError(
                f"logits[{i}] must be floating; got {a.dtype}")
        if not np.isfinite(a).all():
            raise PromptPlanSchemaError(
                f"logits[{i}] must be fully finite")
    return np.maximum.reduce(logit_arrays)


# ============================================================================
# Refinement packaging: NO averaging, only second-pass objects
# ============================================================================

def package_refinement_inputs(
    first_round_objects: tuple[PromptObject, ...],
    first_round_logits: list[np.ndarray],
    second_round_objects: tuple[PromptObject, ...],
    second_round_logits: list[np.ndarray],
    mask_input: np.ndarray,
    mask_input_shape: tuple[int, int],
) -> list[PromptObject]:
    """Package the decoder-call inputs for VLM_AGENT_REFINE's correction
    pass. NO numerical averaging, no logit blending, no min-max.

    Returns one decoder-call object per first-round object_id: the
    second-round corrected box and its corrected points, plus the
    verified mask_input.

    The final multi-object prior is computed separately via
    pixelwise_max_aggregation(second_round_object_logits).
    """
    if len(first_round_objects) != len(second_round_objects):
        raise PromptPlanSchemaError(
            f"first_round_objects ({len(first_round_objects)}) != "
            f"second_round_objects ({len(second_round_objects)})")
    if len(first_round_logits) != len(first_round_objects):
        raise PromptPlanSchemaError("first_round_logits length mismatch")
    if len(second_round_logits) != len(second_round_objects):
        raise PromptPlanSchemaError("second_round_logits length mismatch")
    # Validate mask_input strictly
    if not isinstance(mask_input, np.ndarray):
        raise PromptPlanSchemaError(
            f"mask_input must be ndarray; got {type(mask_input).__name__}")
    if mask_input.shape != mask_input_shape:
        raise PromptPlanSchemaError(
            f"mask_input.shape {mask_input.shape} != expected {mask_input_shape}")
    if mask_input.dtype.kind != "f":
        raise PromptPlanSchemaError(
            f"mask_input must be floating; got {mask_input.dtype}")
    if not np.isfinite(mask_input).all():
        raise PromptPlanSchemaError("mask_input must be fully finite")
    # Verify identity alignment (object_id matched by index, not averaged)
    out = []
    for i, (fr, sr, fl, sl) in enumerate(
            zip(first_round_objects, second_round_objects,
                first_round_logits, second_round_logits)):
        if fr.object_id != sr.object_id:
            raise PromptPlanSchemaError(
                f"object_id mismatch at index {i}: "
                f"first={fr.object_id} second={sr.object_id}")
        if fl.shape != sl.shape:
            raise PromptPlanSchemaError(
                f"object {fr.object_id}: first-round logits shape "
                f"{fl.shape} != second-round shape {sl.shape}")
        # DO NOT AVERAGE; the second-pass object's logits REPLACE the
        # first-pass logits per MedSAM2 mask_input semantics.
        # Return the second-round object as-is (validated).
        out.append(sr)
    return out


# ============================================================================
# Per-arm transaction budget enforcement
# ============================================================================

ARM_TRANSACTION_BUDGET = {
    "FULL_BOX":        dict(max_vlm_calls=0, max_medsam_calls=0,
                              max_rounds=0),
    "DET_COMPONENT":   dict(max_vlm_calls=0, max_medsam_calls=0,
                              max_rounds=0),
    "VLM_SINGLE_BOX":  dict(max_vlm_calls=1, max_medsam_calls=1,
                              max_rounds=1),
    "VLM_MULTI_BOX":   dict(max_vlm_calls=1, max_medsam_calls=1,
                              max_rounds=1),
    "VLM_BOX_POINT":   dict(max_vlm_calls=1, max_medsam_calls=1,
                              max_rounds=1),
    "VLM_AGENT_REFINE": dict(max_vlm_calls=2, max_medsam_calls=2,
                              max_rounds=2),
}


class TransactionBudget:
    """Trusted-side enforcement; called by the executor (NOT by the VLM)."""
    def __init__(self, context: PromptRequestContext):
        self.context = context
        self.vlm_calls = 0
        self.medsam_calls = 0
        self.budget = ARM_TRANSACTION_BUDGET[context.requested_arm]
        self.max_rounds = context.call_index  # enforced externally

    def check_call(self, kind: str) -> None:
        """kind: 'vlm' or 'medsam'. Raises if budget exceeded."""
        if kind == "vlm":
            if self.vlm_calls + 1 > self.budget["max_vlm_calls"]:
                raise PromptPlanSchemaError(
                    f"{self.context.requested_arm} exceeds max_vlm_calls="
                    f"{self.budget['max_vlm_calls']} (call "
                    f"{self.vlm_calls + 1})")
            self.vlm_calls += 1
        elif kind == "medsam":
            if self.medsam_calls + 1 > self.budget["max_medsam_calls"]:
                raise PromptPlanSchemaError(
                    f"{self.context.requested_arm} exceeds max_medsam_calls="
                    f"{self.budget['max_medsam_calls']} (call "
                    f"{self.medsam_calls + 1})")
            self.medsam_calls += 1
        else:
            raise ValueError(f"unknown call kind: {kind}")


# ============================================================================
# FULL_BOX strict (reject external boxes)
# ============================================================================

def full_box_strict_objects() -> tuple[PromptObject, ...]:
    """FULL_BOX must NOT accept any external box. Always emits the
    deterministic exact full image."""
    return full_box_objects()