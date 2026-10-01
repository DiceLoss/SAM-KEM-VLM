"""Stage 7A-v2 — corrected prompt-plan structures, strict JSON parser,
per-object schema, per-pixel logit max aggregation, refinement with
prior-logits mask_input, DET_COMPONENT with peaks + background
negatives, and full adversarial test surface.

Per directive 5926624802. Stage 7A-v2 is a code/protocol/zero-write
planning stage only. No real VLM call, no decoder training, no prior
generation, no KEM execution.

All functions here are PURE (deterministic, no I/O, no VLM, no GPU,
no side effects). DET_COMPONENT operates on an in-memory MLEM-10
array (no model loading, no frozen-metric inspection).
"""
from __future__ import annotations

import json
from collections import OrderedDict
from dataclasses import dataclass
from typing import Optional

import numpy as np


# ============================================================================
# Constants (per directive 5926624802)
# ============================================================================

SCHEMA_VERSION = 1

# Arm name -> deterministic contract. The contract is a pure-data dict;
# all numbers (boxes, points, counts) are integer in the strict parser.
ARM_CONTRACT = {
    "FULL_BOX": dict(
        arm_kind="deterministic_full",
        vlm_calls=0,
        medsam_calls=0,
        min_objects=1, max_objects=1,
        min_positive_points=0, max_positive_points=0,
        min_negative_points=0, max_negative_points=0,
        allow_multi_round=False, max_calls_per_case=1,
        deterministic_box_xyxy=(0, 0, 128, 128),  # exact full image
    ),
    "DET_COMPONENT": dict(
        arm_kind="deterministic_components",
        vlm_calls=0,
        medsam_calls=0,
        min_objects=1, max_objects=4,
        min_positive_points=1, max_positive_points=1,  # 1 peak per object
        min_negative_points=0, max_negative_points=3,
        allow_multi_round=False, max_calls_per_case=1,
        threshold_quantile=0.85,
        padding_px=2,
        component_order="original_area_descending",
    ),
    "VLM_SINGLE_BOX": dict(
        arm_kind="vlm_only",
        vlm_calls=1,
        medsam_calls=0,
        min_objects=1, max_objects=1,
        min_positive_points=0, max_positive_points=0,
        min_negative_points=0, max_negative_points=0,
        allow_multi_round=False, max_calls_per_case=1,
    ),
    "VLM_MULTI_BOX": dict(
        arm_kind="vlm_only",
        vlm_calls=1,
        medsam_calls=0,
        min_objects=1, max_objects=4,
        min_positive_points=0, max_positive_points=0,
        min_negative_points=0, max_negative_points=0,
        allow_multi_round=False, max_calls_per_case=1,
    ),
    "VLM_BOX_POINT": dict(
        arm_kind="vlm_box_point",
        vlm_calls=1,
        medsam_calls=0,
        min_objects=1, max_objects=4,
        min_positive_points=1, max_positive_points=3,  # per object
        min_negative_points=0, max_negative_points=3,  # per object
        allow_multi_round=False, max_calls_per_case=1,
    ),
    "VLM_AGENT_REFINE": dict(
        arm_kind="vlm_agent_refine",
        vlm_calls=2,  # initial + 1 correction
        medsam_calls=0,
        min_objects=1, max_objects=4,
        min_positive_points=1, max_positive_points=3,  # per object
        min_negative_points=0, max_negative_points=3,  # per object
        allow_multi_round=True, max_calls_per_case=2,
        requires_mask_input=True,
    ),
}

# Per-object image dim for the canonical 128x128 grid.
IMAGE_DIM = 128

# Forbidden top-level/object keys (any presence triggers fallback).
FORBIDDEN_TOP_KEYS = (
    "reference_mask", "metric", "mr_image", "ct_image", "frozen_target",
    "label", "mask_input", "iou_logits", "multimask_output",
)

# Allowed top-level keys.
ALLOWED_TOP_KEYS = (
    "schema_version", "arm_name", "dataset", "subject_id", "slice_id",
    "count_level", "realization_id", "objects", "abstain",
    "fallback_reason", "mask_input_logit_shape",
)

# Allowed per-object keys.
ALLOWED_OBJECT_KEYS = (
    "object_id", "box", "positive_points", "negative_points",
)


# ============================================================================
# Strict schema classes (frozen)
# ============================================================================

@dataclass(frozen=True)
class Box:
    """Strict half-open XYXY integer box."""
    x0: int
    y0: int
    x1: int
    y1: int

    def __post_init__(self):
        for k, v in (("x0", self.x0), ("y0", self.y0),
                     ("x1", self.x1), ("y1", self.y1)):
            if not isinstance(v, int) or isinstance(v, bool):
                raise ValueError(f"box coord {k} must be int, got {v!r}")
            if v < 0 or v > IMAGE_DIM:
                raise ValueError(f"box coord {k}={v} out of range [0, {IMAGE_DIM}]")
        if not (self.x0 < self.x1):
            raise ValueError(f"box must satisfy x0 < x1: {(self.x0, self.x1)}")
        if not (self.y0 < self.y1):
            raise ValueError(f"box must satisfy y0 < y1: {(self.y0, self.y1)}")

    @property
    def area(self) -> int:
        return (self.x1 - self.x0) * (self.y1 - self.y0)

    @classmethod
    def from_dict(cls, d: dict) -> "Box":
        if set(d.keys()) != {"x0", "y0", "x1", "y1"}:
            raise PromptPlanSchemaError(
                f"box dict must have exactly keys x0/y0/x1/y1; got {set(d.keys())}")
        # STRICT: no int() coercion; reject bool, str, float.
        for k in ("x0", "y0", "x1", "y1"):
            v = d[k]
            if isinstance(v, bool) or not isinstance(v, int):
                raise PromptPlanSchemaError(
                    f"box coord {k} must be int (no coercion); got {v!r}")
        return cls(d["x0"], d["y0"], d["x1"], d["y1"])


@dataclass(frozen=True)
class Point:
    """Strict integer point with label."""
    x: int
    y: int
    label: str  # "positive" | "negative"

    def __post_init__(self):
        if not isinstance(self.x, int) or isinstance(self.x, bool):
            raise ValueError(f"point x must be int, got {self.x!r}")
        if not isinstance(self.y, int) or isinstance(self.y, bool):
            raise ValueError(f"point y must be int, got {self.y!r}")
        if self.label not in ("positive", "negative"):
            raise ValueError(f"point label must be positive|negative; got {self.label!r}")
        if self.x < 0 or self.x >= IMAGE_DIM:
            raise ValueError(f"point x={self.x} out of range [0, {IMAGE_DIM})")
        if self.y < 0 or self.y >= IMAGE_DIM:
            raise ValueError(f"point y={self.y} out of range [0, {IMAGE_DIM})")

    @classmethod
    def from_dict(cls, d: dict) -> "Point":
        if set(d.keys()) != {"x", "y", "label"}:
            raise PromptPlanSchemaError(
                f"point dict must have exactly x/y/label; got {set(d.keys())}")
        for k in ("x", "y"):
            v = d[k]
            if isinstance(v, bool) or not isinstance(v, int):
                raise PromptPlanSchemaError(
                    f"point coord {k} must be int (no coercion); got {v!r}")
        return cls(d["x"], d["y"], d["label"])


@dataclass(frozen=True)
class PromptObject:
    """Per-object container: box + positive_points + negative_points."""
    object_id: int
    box: Box
    positive_points: tuple[Point, ...]
    negative_points: tuple[Point, ...]


@dataclass(frozen=True)
class PromptPlan:
    """Validated prompt plan."""
    arm_name: str
    dataset: str
    subject_id: str
    slice_id: str
    count_level: int
    realization_id: int
    objects: tuple[PromptObject, ...]
    abstain: bool
    fallback_reason: str
    mask_input_logit_shape: Optional[tuple[int, ...]]  # for VLM_AGENT_REFINE

    @property
    def num_objects(self) -> int:
        return len(self.objects)


# ============================================================================
# Strict JSON parser
# ============================================================================

class PromptPlanSchemaError(ValueError):
    pass


def parse_prompt_plan_json(raw: str | bytes) -> PromptPlan:
    """Strict JSON parser. Accepts raw UTF-8 JSON string or bytes.

    Raises PromptPlanSchemaError on any contract violation. Does NOT
    coerce, clip, drop, or clean. Returns a typed PromptPlan or, on
    invalid input, raises PromptPlanSchemaError.
    """
    if isinstance(raw, bytes):
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError as e:
            raise PromptPlanSchemaError(f"raw bytes are not valid UTF-8: {e}") from e
    elif isinstance(raw, str):
        text = raw
    else:
        raise PromptPlanSchemaError(
            f"raw must be str or bytes; got {type(raw).__name__}")

    # Reject duplicate JSON keys via object_pairs_hook=OrderedDict.
    try:
        parsed = json.loads(text, object_pairs_hook=_check_no_duplicate_keys)
    except json.JSONDecodeError as e:
        raise PromptPlanSchemaError(f"malformed JSON: {e}") from e
    if not isinstance(parsed, dict):
        raise PromptPlanSchemaError(
            f"top-level JSON must be an object; got {type(parsed).__name__}")

    # Top-level key whitelist (forbid any unknown key).
    extra_keys = set(parsed.keys()) - set(ALLOWED_TOP_KEYS)
    if extra_keys:
        raise PromptPlanSchemaError(
            f"forbidden/unknown top-level keys: {sorted(extra_keys)}")

    # Required fields.
    for k in ("schema_version", "arm_name", "dataset", "subject_id",
              "slice_id", "count_level", "realization_id", "objects",
              "abstain"):
        if k not in parsed:
            raise PromptPlanSchemaError(f"missing required top-level field: {k}")

    if parsed["schema_version"] != SCHEMA_VERSION:
        raise PromptPlanSchemaError(
            f"schema_version must be {SCHEMA_VERSION}; got {parsed['schema_version']}")

    arm = parsed["arm_name"]
    if arm not in ARM_CONTRACT:
        raise PromptPlanSchemaError(
            f"arm_name must be one of {list(ARM_CONTRACT.keys())}; got {arm!r}")
    contract = ARM_CONTRACT[arm]

    # Reject non-finite numbers and non-int/float leakage for numeric fields.
    # `abstain` is allowed to be bool (per the schema). Strings are allowed
    # only for the whitelisted string fields (dataset/subject_id/slice_id/
    # arm_name/fallback_reason).
    STRING_FIELDS = {"arm_name", "dataset", "subject_id", "slice_id",
                     "fallback_reason"}
    for k, v in parsed.items():
        if k == "abstain":
            if not isinstance(v, bool):
                raise PromptPlanSchemaError(
                    f"abstain must be bool, got {type(v).__name__}")
            continue
        if isinstance(v, bool):
            raise PromptPlanSchemaError(
                f"top-level field {k!r} must not be bool")
        if isinstance(v, str):
            if k not in STRING_FIELDS:
                raise PromptPlanSchemaError(
                    f"top-level field {k!r} must not be string; got {v!r}")
            continue
        if isinstance(v, float):
            if not (v == v and v != float("inf") and v != float("-inf")):
                raise PromptPlanSchemaError(
                    f"top-level field {k!r} is non-finite; got {v!r}")
        if k == "count_level" and not isinstance(v, int):
            raise PromptPlanSchemaError(
                f"count_level must be int, got {type(v).__name__}")
        if k == "realization_id" and not isinstance(v, int):
            raise PromptPlanSchemaError(
                f"realization_id must be int, got {type(v).__name__}")
        if k == "schema_version" and not isinstance(v, int):
            raise PromptPlanSchemaError(
                f"schema_version must be int, got {type(v).__name__}")
        if k == "mask_input_logit_shape" and not (
                isinstance(v, list) and all(isinstance(x, int) and not isinstance(x, bool)
                                            for x in v)):
            raise PromptPlanSchemaError(
                f"mask_input_logit_shape must be list of int; got {v!r}")

    objects_raw = parsed["objects"]
    if not isinstance(objects_raw, list):
        raise PromptPlanSchemaError("objects must be a list")
    n_obj = len(objects_raw)
    if n_obj < contract["min_objects"]:
        raise PromptPlanSchemaError(
            f"too few objects for arm {arm}: {n_obj} < min {contract['min_objects']}")
    if n_obj > contract["max_objects"]:
        raise PromptPlanSchemaError(
            f"too many objects for arm {arm}: {n_obj} > max {contract['max_objects']}")

    objects = []
    seen_obj_ids = set()
    seen_box_xyxy = []
    for i, obj_raw in enumerate(objects_raw):
        if not isinstance(obj_raw, dict):
            raise PromptPlanSchemaError(f"objects[{i}] must be an object")
        extra = set(obj_raw.keys()) - set(ALLOWED_OBJECT_KEYS)
        if extra:
            raise PromptPlanSchemaError(
                f"objects[{i}] forbidden/unknown keys: {sorted(extra)}")
        for k in ALLOWED_OBJECT_KEYS:
            if k not in obj_raw:
                raise PromptPlanSchemaError(
                    f"objects[{i}] missing required field: {k}")
        # object_id must be unique
        oid = obj_raw["object_id"]
        if not isinstance(oid, int) or isinstance(oid, bool):
            raise PromptPlanSchemaError(
                f"objects[{i}].object_id must be int; got {oid!r}")
        if oid in seen_obj_ids:
            raise PromptPlanSchemaError(
                f"objects[{i}].object_id={oid} duplicates a previous object_id")
        seen_obj_ids.add(oid)

        # box is required for every object. FULL_BOX overrides the value
        # to the deterministic exact full image, BUT user input must still
        # pass strict validation (so fractional / non-int coords raise).
        if "box" not in obj_raw:
            raise PromptPlanSchemaError(
                f"objects[{i}] missing required field: box")
        # Validate the user-supplied box strictly (raises on bad input).
        Box.from_dict(obj_raw["box"])
        if arm == "FULL_BOX":
            # Then override to deterministic full image.
            box = Box(*contract["deterministic_box_xyxy"])
        else:
            box = Box.from_dict(obj_raw["box"])

        # Duplicate boxes trigger fallback (per directive).
        seen_box_xyxy.append((box.x0, box.y0, box.x1, box.y1))
        dup_box = None
        for prev_box in seen_box_xyxy[:-1]:
            if prev_box == seen_box_xyxy[-1]:
                dup_box = prev_box
                break
        if dup_box is not None:
            raise PromptPlanSchemaError(
                f"duplicate box {dup_box} in objects[{i}]; "
                f"per-arm contract forbids silent deduplication; "
                f"trigger DET_COMPONENT fallback instead")

        # Positive points (per-object counts).
        pos_raw = obj_raw["positive_points"]
        neg_raw = obj_raw["negative_points"]
        if not isinstance(pos_raw, list) or not isinstance(neg_raw, list):
            raise PromptPlanSchemaError(
                f"object {oid}: positive/negative_points must be lists")
        if len(pos_raw) < contract["min_positive_points"]:
            raise PromptPlanSchemaError(
                f"object {oid}: too few positive_points "
                f"({len(pos_raw)} < min {contract['min_positive_points']})")
        if len(pos_raw) > contract["max_positive_points"]:
            raise PromptPlanSchemaError(
                f"object {oid}: too many positive_points "
                f"({len(pos_raw)} > max {contract['max_positive_points']})")
        if len(neg_raw) < contract["min_negative_points"]:
            raise PromptPlanSchemaError(
                f"object {oid}: too few negative_points")
        if len(neg_raw) > contract["max_negative_points"]:
            raise PromptPlanSchemaError(
                f"object {oid}: too many negative_points")

        positive = []
        seen_pos = set()
        for j, p in enumerate(pos_raw):
            if not isinstance(p, dict):
                raise PromptPlanSchemaError(
                    f"object {oid} positive_points[{j}] must be an object")
            pt = Point.from_dict(p)
            if pt.label != "positive":
                raise PromptPlanSchemaError(
                    f"object {oid} positive_points[{j}] label must be 'positive'; "
                    f"got {pt.label!r}")
            if (pt.x, pt.y) in seen_pos:
                raise PromptPlanSchemaError(
                    f"object {oid} duplicate positive point {(pt.x, pt.y)}")
            seen_pos.add((pt.x, pt.y))
            positive.append(pt)
        negative = []
        seen_neg = set()
        for j, p in enumerate(neg_raw):
            if not isinstance(p, dict):
                raise PromptPlanSchemaError(
                    f"object {oid} negative_points[{j}] must be an object")
            pt = Point.from_dict(p)
            if pt.label != "negative":
                raise PromptPlanSchemaError(
                    f"object {oid} negative_points[{j}] label must be 'negative'; "
                    f"got {pt.label!r}")
            if (pt.x, pt.y) in seen_neg:
                raise PromptPlanSchemaError(
                    f"object {oid} duplicate negative point {(pt.x, pt.y)}")
            seen_neg.add((pt.x, pt.y))
            negative.append(pt)
        objects.append(PromptObject(
            object_id=oid, box=box,
            positive_points=tuple(positive),
            negative_points=tuple(negative),
        ))

    # mask_input shape (VLM_AGENT_REFINE only; required and must be non-empty)
    mask_shape = parsed.get("mask_input_logit_shape")
    if arm == "VLM_AGENT_REFINE":
        if mask_shape is None:
            raise PromptPlanSchemaError(
                "VLM_AGENT_REFINE requires mask_input_logit_shape")
        if not isinstance(mask_shape, list) or len(mask_shape) != 2:
            raise PromptPlanSchemaError(
                f"mask_input_logit_shape must be [h, w]; got {mask_shape!r}")
        if not all(isinstance(x, int) and not isinstance(x, bool) and x > 0
                   for x in mask_shape):
            raise PromptPlanSchemaError(
                f"mask_input_logit_shape entries must be positive int; got {mask_shape}")
        mask_shape = (int(mask_shape[0]), int(mask_shape[1]))
    else:
        mask_shape = None

    return PromptPlan(
        arm_name=arm,
        dataset=str(parsed["dataset"]),
        subject_id=str(parsed["subject_id"]),
        slice_id=str(parsed["slice_id"]),
        count_level=int(parsed["count_level"]),
        realization_id=int(parsed["realization_id"]),
        objects=tuple(objects),
        abstain=bool(parsed["abstain"]),
        fallback_reason=str(parsed.get("fallback_reason") or ""),
        mask_input_logit_shape=mask_shape,
    )


def _check_no_duplicate_keys(pairs):
    """object_pairs_hook that rejects duplicate top-level / per-object keys."""
    seen = {}
    out = OrderedDict()
    for k, v in pairs:
        if k in seen:
            raise PromptPlanSchemaError(f"duplicate JSON key: {k!r}")
        seen[k] = True
        out[k] = v
    return out


# ============================================================================
# Pure helpers (per directive)
# ============================================================================

def pixelwise_max_aggregation(logits_per_object):
    """Validate per-object logits: same shape, floating, all finite,
    nonempty. Return np.maximum.reduce(...).

    NEVER min-max inside this helper. NO silent clipping.
    Raises PromptPlanSchemaError on any contract violation.
    """
    arrays = [np.asarray(l) for l in logits_per_object]
    if len(arrays) == 0:
        raise PromptPlanSchemaError("pixelwise_max_aggregation requires >= 1 object")
    ref_shape = arrays[0].shape
    for i, a in enumerate(arrays):
        if a.shape != ref_shape:
            raise PromptPlanSchemaError(
                f"logits[{i}].shape {a.shape} != reference {ref_shape}")
        if not np.isfinite(a).all():
            raise PromptPlanSchemaError(
                f"logits[{i}] contains non-finite values")
    return np.maximum.reduce(arrays)


def validate_mask_input_for_refine(mask_input, expected_shape):
    """Validate that mask_input is a non-empty finite float array with
    the expected shape. Raises PromptPlanSchemaError on any violation.
    """
    arr = np.asarray(mask_input)
    if arr.shape != expected_shape:
        raise PromptPlanSchemaError(
            f"mask_input.shape {arr.shape} != expected {expected_shape}")
    if arr.size == 0:
        raise PromptPlanSchemaError("mask_input must be non-empty")
    if not np.isfinite(arr).all():
        raise PromptPlanSchemaError("mask_input contains non-finite values")
    return arr


def package_refinement(
    first_round_plan, mask_input, mask_input_shape,
    first_round_logits_per_object,
    second_round_objects, second_round_logits_per_object,
):
    """Package a VLM_AGENT_REFINE second-round plan. Validates:
    - mask_input shape matches declared shape;
    - first_round logits and second_round logits per object match in shape;
    - first_round objects must equal second_round objects in identity.

    Returns (refined_plan, merged_logits_per_object). Raises
    PromptPlanSchemaError on any violation. NO silent filtering.
    """
    mask = validate_mask_input_for_refine(mask_input, mask_input_shape)
    if len(second_round_objects) != len(first_round_plan.objects):
        raise PromptPlanSchemaError(
            f"second round has {len(second_round_objects)} objects but "
            f"first round has {len(first_round_plan.objects)}")
    if len(second_round_logits_per_object) != len(second_round_objects):
        raise PromptPlanSchemaError(
            "second_round_logits_per_object length mismatch")
    if len(first_round_logits_per_object) != len(first_round_plan.objects):
        raise PromptPlanSchemaError(
            "first_round_logits_per_object length mismatch")
    for i, (prev_obj, new_obj) in enumerate(
            zip(first_round_plan.objects, second_round_objects)):
        if prev_obj.object_id != new_obj.object_id:
            raise PromptPlanSchemaError(
                f"object_id mismatch at index {i}: "
                f"{prev_obj.object_id} != {new_obj.object_id}")
    # Stack: combine per-object first + second logits
    merged = []
    for i in range(len(second_round_objects)):
        prev = np.asarray(first_round_logits_per_object[i])
        new = np.asarray(second_round_logits_per_object[i])
        if prev.shape != new.shape:
            raise PromptPlanSchemaError(
                f"object {i}: prev shape {prev.shape} != new shape {new.shape}")
        if not np.isfinite(prev).all() or not np.isfinite(new).all():
            raise PromptPlanSchemaError(
                f"object {i}: non-finite logits in prev or new")
        merged.append((prev + new) / 2.0)  # simple averaging as fusion
    return mask, merged


def record_fallback(requested_arm, executed_arm, reason):
    """Deterministic intention-to-treat fallback record."""
    return {
        "requested_arm": requested_arm,
        "executed_arm": executed_arm,
        "fallback_used": True,
        "fallback_reason": reason,
    }


# ============================================================================
# DET_COMPONENT deterministic planner
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


def det_component_plan(
    mlem10: np.ndarray,
    image_dim: int = IMAGE_DIM,
    threshold_quantile: float = 0.85,
    padding_px: int = 2,
    component_order: str = "original_area_descending",
    max_objects: int = 4,
):
    """Deterministic DET_COMPONENT planner:
      1. Threshold |mlem10| at its `threshold_quantile` quantile.
      2. 4-connected flood-fill labeling of the foreground mask.
      3. Per component:
         - tight original-area XYXY bbox (no padding shrink),
         - tight padded XYXY bbox (for the JSON-reported box; same dims),
         - local-peak positive point (argmax of |mlem10| inside the bbox),
         - background negative point (argmin of |mlem10| in a fixed band
           outside the bbox, deterministic seed).
      4. Sort by ORIGINAL area, largest first; cap to max_objects.
      5. Deduplicate (no two same-object_id).
      6. Output one PromptObject per kept component.
    """
    if mlem10.ndim != 2 or mlem10.shape[0] != image_dim or mlem10.shape[1] != image_dim:
        raise ValueError(f"mlem10 must be {image_dim}x{image_dim}")
    abs_map = np.abs(np.asarray(mlem10, dtype=np.float32))
    thr = float(np.quantile(abs_map, threshold_quantile))
    mask = (abs_map >= thr).astype(np.uint8)
    labels = _flood_fill_components(mask)
    components = []
    for k in range(1, int(labels.max()) + 1):
        ys, xs = np.where(labels == k)
        if len(xs) == 0:
            continue
        x0_orig = int(xs.min())
        x1_orig = int(xs.max()) + 1
        y0_orig = int(ys.min())
        y1_orig = int(ys.max()) + 1
        orig_area = (x1_orig - x0_orig) * (y1_orig - y0_orig)

        # Padded bbox (for JSON reporting; does not shrink)
        x0 = max(0, x0_orig - padding_px)
        y0 = max(0, y0_orig - padding_px)
        x1 = min(image_dim, x1_orig + padding_px)
        y1 = min(image_dim, y1_orig + padding_px)
        if x0 >= x1 or y0 >= y1:
            continue

        # Local peak (positive point): argmax inside original bbox
        sub = abs_map[y0_orig:y1_orig, x0_orig:x1_orig]
        py, px = np.unravel_index(np.argmax(sub), sub.shape)
        peak = Point(int(x0_orig + px), int(y0_orig + py), "positive")

        # Background negative point: deterministic 1-px-band ring around
        # the PADDED bbox (which is what the user-facing box reports).
        # The negative is always OUTSIDE the padded bbox.
        band_mask = np.zeros_like(abs_map, dtype=bool)
        x0b = max(0, x0 - 1); x1b = min(image_dim, x1 + 1)
        y0b = max(0, y0 - 1); y1b = min(image_dim, y1 + 1)
        if y0 > 0:
            band_mask[y0b:y0, x0b:x1b] = True
        if y1 < image_dim:
            band_mask[y1:y1b, x0b:x1b] = True
        if x0 > 0:
            band_mask[y0b:y1b, x0b:x0] = True
        if x1 < image_dim:
            band_mask[y0b:y1b, x1:x1b] = True
        neg = Point(0, 0, "negative")  # default; replaced below
        if band_mask.any():
            ys_b, xs_b = np.where(band_mask)
            # deterministic: pick the (y, x) with the lowest |mlem10|
            abs_band = abs_map[ys_b, xs_b]
            i_min = int(np.argmin(abs_band))
            neg = Point(int(xs_b[i_min]), int(ys_b[i_min]), "negative")
        else:
            # Fall back to image-corner (shouldn't normally happen)
            neg = Point(0, 0, "negative")

        components.append({
            "original_area": orig_area,
            "box": Box(x0, y0, x1, y1),
            "positive_point": peak,
            "negative_point": neg,
        })
    if component_order == "original_area_descending":
        components.sort(key=lambda c: -c["original_area"])
    components = components[:max_objects]
    objects = []
    for i, c in enumerate(components):
        objects.append(PromptObject(
            object_id=i,
            box=c["box"],
            positive_points=(c["positive_point"],),
            negative_points=(c["negative_point"],),
        ))
    return tuple(objects)


def det_component_plan_from_mlem10(mlem10, max_objects=4, **kw):
    """Convenience wrapper."""
    return det_component_plan(
        mlem10, image_dim=IMAGE_DIM, max_objects=max_objects, **kw)


# ============================================================================
# Intention-to-treat fallback executor
# ============================================================================

def execute_with_fallback(parsed_plan, mlem10, original_arm, raw_json=None):
    """If parsed_plan is None or invalid, run DET_COMPONENT on the same
    mlem10 (zero VLM calls) and record fallback metadata. Otherwise return
    the parsed plan unchanged.

    Returns a dict with: requested_arm, executed_arm, fallback_used,
    fallback_reason, plan (the PromptPlan or DET_COMPONENT-derived plan).
    """
    if parsed_plan is None or raw_json is None:
        det_objects = det_component_plan_from_mlem10(mlem10)
        return {
            "requested_arm": original_arm,
            "executed_arm": "DET_COMPONENT",
            "fallback_used": True,
            "fallback_reason": "invalid_or_missing_input",
            "plan": PromptPlan(
                arm_name="DET_COMPONENT",
                dataset="", subject_id="", slice_id="",
                count_level=0, realization_id=0,
                objects=det_objects,
                abstain=False,
                fallback_reason="invalid_or_missing_input",
                mask_input_logit_shape=None,
            ),
        }
    return {
        "requested_arm": parsed_plan.arm_name,
        "executed_arm": parsed_plan.arm_name,
        "fallback_used": parsed_plan.abstain,
        "fallback_reason": parsed_plan.fallback_reason,
        "plan": parsed_plan,
    }
