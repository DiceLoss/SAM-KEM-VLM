"""Stage 7A — prompt-plan structures, schema validation, DET_COMPONENT
deterministic planner, and pure helper functions.

Per directive 5924162430 + 5925510360. Stage 7A is a
code/protocol/zero-write planning stage only. No real VLM call, no
decoder training, no prior generation, no KEM execution.

All functions here are PURE (deterministic, no I/O, no VLM, no GPU,
no side effects) except ``det_component_boxes_from_mlem10`` which
operates on an in-memory MLEM-10 array (no model loading, no
frozen-metric inspection).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


# ============================================================================
# Schema (per protocol stage7a_v1.yaml)
# ============================================================================

PREREGISTERED_ARMS = (
    "FULL_BOX",
    "DET_COMPONENT",
    "VLM_SINGLE_BOX",
    "VLM_MULTI_BOX",
    "VLM_BOX_POINT",
    "VLM_AGENT_REFINE",
)

ARM_KIND = {
    "FULL_BOX": "qwen_only",
    "DET_COMPONENT": "det_component",
    "VLM_SINGLE_BOX": "vlm_only",
    "VLM_MULTI_BOX": "vlm_only",
    "VLM_BOX_POINT": "vlm_box_point",
    "VLM_AGENT_REFINE": "vlm_agent_refine",
}

ARM_RULES = {
    "FULL_BOX": dict(max_objects=4, min_positive_points=0, max_positive_points=0,
                    min_negative_points=0, max_negative_points=0,
                    allow_multi_round=False, max_calls_per_case=1,
                    require_box_plus_points=False),
    "DET_COMPONENT": dict(max_objects=4, min_positive_points=0, max_positive_points=0,
                         min_negative_points=0, max_negative_points=0,
                         allow_multi_round=False, max_calls_per_case=1,
                         require_box_plus_points=False,
                         threshold_quantile=0.85, padding_px=2,
                         component_order="area_descending"),
    "VLM_SINGLE_BOX": dict(max_objects=1, min_positive_points=1, max_positive_points=1,
                          min_negative_points=0, max_negative_points=1,
                          allow_multi_round=False, max_calls_per_case=1,
                          require_box_plus_points=False),
    "VLM_MULTI_BOX": dict(max_objects=4, min_positive_points=1, max_positive_points=3,
                         min_negative_points=0, max_negative_points=3,
                         allow_multi_round=False, max_calls_per_case=1,
                         require_box_plus_points=False),
    "VLM_BOX_POINT": dict(max_objects=1, min_positive_points=1, max_positive_points=3,
                         min_negative_points=0, max_negative_points=3,
                         allow_multi_round=False, max_calls_per_case=1,
                         require_box_plus_points=True),
    "VLM_AGENT_REFINE": dict(max_objects=4, min_positive_points=1, max_positive_points=3,
                            min_negative_points=0, max_negative_points=3,
                            allow_multi_round=True, max_calls_per_case=2,
                            require_box_plus_points=False),
}

FORBIDDEN_RESPONSE_FIELDS = (
    "reference_mask", "metric", "mr_image", "ct_image", "frozen_target",
)


# ============================================================================
# Dataclasses (pure, hashable-friendly)
# ============================================================================

@dataclass(frozen=True)
class Point:
    x: int
    y: int
    label: str  # "positive" | "negative"

    def __post_init__(self):
        if not isinstance(self.x, int) or not isinstance(self.y, int):
            raise ValueError(f"point coords must be int: ({self.x!r}, {self.y!r})")
        if self.label not in ("positive", "negative"):
            raise ValueError(f"point label must be positive|negative, got {self.label!r}")


@dataclass(frozen=True)
class Box:
    """XYXY integer box. 0 <= xy < image_dim; xy0 < xy1; xy2 < xy3."""
    x0: int
    y0: int
    x1: int
    y1: int

    def __post_init__(self):
        for v in (self.x0, self.y0, self.x1, self.y1):
            if not isinstance(v, int):
                raise ValueError(f"box coords must be int: {(self.x0, self.y0, self.x1, self.y1)}")
        if self.x0 >= self.x1 or self.y0 >= self.y1:
            raise ValueError(f"box must satisfy x0<x1 and y0<y1: {(self.x0, self.y0, self.x1, self.y1)}")

    @classmethod
    def from_xyxy(cls, x0: int, y0: int, x1: int, y1: int) -> "Box":
        return cls(int(x0), int(y0), int(x1), int(y1))

    def validate_in_bounds(self, image_dim: int) -> None:
        if image_dim <= 0:
            raise ValueError(f"image_dim must be positive: {image_dim}")
        if not (0 <= self.x0 < self.x1 <= image_dim):
            raise ValueError(f"x out of bounds [0, {image_dim}]: {(self.x0, self.x1)}")
        if not (0 <= self.y0 < self.y1 <= image_dim):
            raise ValueError(f"y out of bounds [0, {image_dim}]: {(self.y0, self.y1)}")

    def iou(self, other: "Box") -> float:
        ix0 = max(self.x0, other.x0)
        iy0 = max(self.y0, other.y0)
        ix1 = min(self.x1, other.x1)
        iy1 = min(self.y1, other.y1)
        iw = max(0, ix1 - ix0)
        ih = max(0, iy1 - iy0)
        inter = iw * ih
        a1 = (self.x1 - self.x0) * (self.y1 - self.y0)
        a2 = (other.x1 - other.x0) * (other.y1 - other.y0)
        union = a1 + a2 - inter
        if union <= 0:
            return 0.0
        return inter / union


@dataclass(frozen=True)
class Detection:
    box: Box
    confidence: float  # finite, [0, 1]

    def __post_init__(self):
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError(f"confidence out of [0, 1]: {self.confidence}")
        # NaN / inf rejected by float comparison.
        if self.confidence != self.confidence or self.confidence in (float("inf"), float("-inf")):
            raise ValueError(f"confidence must be finite: {self.confidence}")


@dataclass(frozen=True)
class PromptPlan:
    """Per-arm prompt plan. Boxes and points are deterministic."""
    arm_name: str
    dataset: str
    subject_id: str
    slice_id: str
    count_level: int
    realization_id: int
    boxes: tuple[Box, ...]
    positive_points: tuple[Point, ...]
    negative_points: tuple[Point, ...]
    confidence: float
    fallback_used: bool = False
    fallback_reason: str = ""

    def __post_init__(self):
        if self.arm_name not in ARM_RULES:
            raise ValueError(f"unknown arm: {self.arm_name}")
        for b in self.boxes:
            b.validate_in_bounds(128)  # canonical 128x128; caller may re-validate
        for p in self.positive_points:
            if p.label != "positive":
                raise ValueError("positive_points label must be positive")
        for p in self.negative_points:
            if p.label != "negative":
                raise ValueError("negative_points label must be negative")
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError(f"confidence must be in [0, 1]: {self.confidence}")


# ============================================================================
# Pure helpers
# ============================================================================

def validate_box_xyxy(box: Box, image_dim: int = 128) -> None:
    """Validate XYXY box against the canonical 128x128 image grid."""
    box.validate_in_bounds(image_dim)


def reject_duplicate_boxes(boxes: Iterable[Box]) -> list[Box]:
    """Reject exact duplicates (IoU == 1.0) preserving first occurrence."""
    out = []
    seen = []
    for b in boxes:
        is_dup = False
        for s in seen:
            if b.iou(s) >= 1.0:
                is_dup = True
                break
        if not is_dup:
            out.append(b)
            seen.append(b)
    return out


def reject_out_of_bounds(boxes: Iterable[Box], image_dim: int) -> list[Box]:
    """Reject any box whose coordinates fall outside [0, image_dim]."""
    out = []
    for b in boxes:
        try:
            b.validate_in_bounds(image_dim)
            out.append(b)
        except ValueError:
            continue
    return out


def reject_non_finite(boxes: Iterable[Box]) -> list[Box]:
    """Reject boxes with NaN/inf coordinates. Coords are int so this is a no-op
    but kept for API completeness and consistency with confidence checks."""
    return list(boxes)


def package_per_object_prompts(plan: PromptPlan) -> list[dict]:
    """Package the plan into a list of per-object prompt dicts.

    Each dict carries:
        - arm_name, dataset, subject_id, slice_id, count_level,
          realization_id, image_dim, box (xyxy), positive_points
          [(x, y), ...], negative_points [(x, y), ...], confidence,
          fallback_used, fallback_reason.
    Pure: no I/O, no model call.
    """
    items = []
    for box in plan.boxes:
        items.append({
            "arm_name": plan.arm_name,
            "dataset": plan.dataset,
            "subject_id": plan.subject_id,
            "slice_id": plan.slice_id,
            "count_level": plan.count_level,
            "realization_id": plan.realization_id,
            "image_dim": 128,
            "box_xyxy": (box.x0, box.y0, box.x1, box.y1),
            "positive_points": [(p.x, p.y) for p in plan.positive_points],
            "negative_points": [(p.x, p.y) for p in plan.negative_points],
            "confidence": plan.confidence,
            "fallback_used": plan.fallback_used,
            "fallback_reason": plan.fallback_reason,
        })
    return items


def max_object_continuous_logit(boxes: Iterable[Box], logits: Iterable[float]) -> Box:
    """Pick the box with the maximum continuous_logit value (before the
    per-prior min-max transform). Ties broken by the first occurrence.

    Raises ValueError if the two iterables have different lengths or
    are empty.
    """
    boxes = list(boxes)
    logits = list(logits)
    if not boxes:
        raise ValueError("boxes must be non-empty")
    if len(boxes) != len(logits):
        raise ValueError(f"boxes and logits length mismatch: {len(boxes)} vs {len(logits)}")
    # Take the box with the maximum logit (NaN-safe).
    best_idx = 0
    best_logit = logits[0]
    for i in range(1, len(logits)):
        if logits[i] > best_logit:
            best_logit = logits[i]
            best_idx = i
    return boxes[best_idx]


def second_round_packaging(
    prev_boxes: Iterable[Box],
    prev_logits: Iterable[float],
    new_boxes: Iterable[Box],
    new_points_pos: Iterable[Point],
    new_points_neg: Iterable[Point],
    image_dim: int = 128,
    max_objects: int = 4,
) -> tuple[list[Box], list[Point], list[Point]]:
    """Second-round packaging: combine previous low-resolution logits with
    correction points. Returns (boxes, positive_points, negative_points)
    after duplicate-rejection and bound checks. Pure."""
    # 1. Aggregate boxes (previous + new), max_object cap.
    combined = list(prev_boxes) + list(new_boxes)
    # 2. Reject duplicates (IoU == 1.0).
    combined = reject_duplicate_boxes(combined)
    # 3. Bound-check.
    combined = reject_out_of_bounds(combined, image_dim)
    # 4. Cap to max_objects (deterministic: keep first max_objects).
    combined = combined[:max_objects]
    # 5. Reject points with non-finite / out-of-bounds coords.
    pos = [p for p in new_points_pos
           if 0 <= p.x < image_dim and 0 <= p.y < image_dim]
    neg = [p for p in new_points_neg
           if 0 <= p.x < image_dim and 0 <= p.y < image_dim]
    return combined, pos, neg


def leakage_field_rejection(response: dict) -> dict:
    """Reject any response that contains reference / mask / MR / CT / metric
    fields. Returns the cleaned dict with those fields removed; if any
    such field was present the caller must record fallback.

    Pure: does not modify the input dict in-place.
    """
    cleaned = {}
    rejected = []
    for k, v in response.items():
        if k in FORBIDDEN_RESPONSE_FIELDS:
            rejected.append(k)
            continue
        cleaned[k] = v
    if rejected:
        cleaned["_leakage_fields_rejected"] = rejected
    return cleaned


def record_fallback(reason: str) -> dict:
    """Deterministic fallback record. Pure (no I/O)."""
    return {
        "fallback_used": True,
        "fallback_reason": reason,
        "intention_to_treat": True,
    }


# ============================================================================
# DET_COMPONENT deterministic planner (no VLM, no frozen-metric inspection)
# ============================================================================

def det_component_boxes_from_mlem10(
    mlem10: np.ndarray,
    image_dim: int = 128,
    threshold_quantile: float = 0.85,
    padding_px: int = 2,
    component_order: str = "area_descending",
    max_objects: int = 4,
) -> list[Box]:
    """Deterministic DET_COMPONENT planner.

    1. Threshold |mlem10| at its `threshold_quantile` quantile.
    2. Connected-component labeling of the foreground mask.
    3. Per component: tight XYXY bbox, optionally inflated by `padding_px`.
    4. Bound-clip to image dimensions (no silent clipping beyond bounds).
    5. Sort by component area (deterministic: largest first).
    6. Truncate to `max_objects`.
    7. Deduplicate (IoU == 1.0).

    No frozen-metric inspection, no VLM, no decoder. Pure function
    over the in-memory `mlem10` array.

    NOTE: this function does NOT require connected-components-3d; a
    simple 4-connected flood-fill is sufficient because the threshold
    operates on a single 128x128 map.
    """
    if mlem10.ndim != 2 or mlem10.shape[0] != image_dim or mlem10.shape[1] != image_dim:
        raise ValueError(f"mlem10 must be {image_dim}x{image_dim}; got {mlem10.shape}")
    if not (0.0 < threshold_quantile < 1.0):
        raise ValueError(f"threshold_quantile must be in (0, 1): {threshold_quantile}")
    if padding_px < 0:
        raise ValueError(f"padding_px must be >= 0: {padding_px}")

    abs_map = np.abs(np.asarray(mlem10, dtype=np.float32))
    thr = float(np.quantile(abs_map, threshold_quantile))
    mask = (abs_map >= thr).astype(np.uint8)

    # Connected-components via flood-fill.
    labels = _flood_fill_components(mask)
    boxes = []
    for k in range(1, int(labels.max()) + 1):
        ys, xs = np.where(labels == k)
        if len(xs) == 0:
            continue
        x0 = int(xs.min())
        x1 = int(xs.max()) + 1  # XYXY exclusive upper
        y0 = int(ys.min())
        y1 = int(ys.max()) + 1
        # Pad.
        x0 = max(0, x0 - padding_px)
        y0 = max(0, y0 - padding_px)
        x1 = min(image_dim, x1 + padding_px)
        y1 = min(image_dim, y1 + padding_px)
        if x0 >= x1 or y0 >= y1:
            continue
        boxes.append(Box(x0, y0, x1, y1))

    # Deterministic component_order (area_descending or xy_min).
    if component_order == "area_descending":
        boxes.sort(key=lambda b: (-(b.x1 - b.x0) * (b.y1 - b.y0), b.x0, b.y0))
    else:
        boxes.sort(key=lambda b: (b.x0, b.y0))
    boxes = reject_duplicate_boxes(boxes[:max_objects])
    return boxes


def _flood_fill_components(mask: np.ndarray) -> np.ndarray:
    """4-connected flood-fill labeling. Deterministic, iterative.

    Returns a label array of shape mask.shape with 0 = background and
    positive integers = component ids.
    """
    h, w = mask.shape
    labels = np.zeros((h, w), dtype=np.int32)
    cur = 1
    # Stack-based BFS for deterministic iteration.
    for sy in range(h):
        for sx in range(w):
            if mask[sy, sx] and labels[sy, sx] == 0:
                # BFS
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


# ============================================================================
# Schema validator for prompt-plan JSON payloads
# ============================================================================

class PromptPlanSchemaError(ValueError):
    pass


def parse_prompt_plan_json(payload: dict) -> PromptPlan:
    """Parse and validate a JSON payload into a PromptPlan.

    Raises PromptPlanSchemaError on any contract violation. Pure
    (no I/O).
    """
    required = {"arm_name", "dataset", "subject_id", "slice_id",
                "count_level", "realization_id", "boxes",
                "positive_points", "negative_points", "confidence"}
    missing = required - set(payload.keys())
    if missing:
        raise PromptPlanSchemaError(f"missing required keys: {sorted(missing)}")

    arm = payload["arm_name"]
    if arm not in ARM_RULES:
        raise PromptPlanSchemaError(
            f"arm_name must be one of {list(ARM_RULES.keys())}; got {arm!r}")
    rules = ARM_RULES[arm]

    boxes_raw = payload["boxes"]
    if not isinstance(boxes_raw, list):
        raise PromptPlanSchemaError(f"boxes must be a list; got {type(boxes_raw)}")
    if len(boxes_raw) > rules["max_objects"]:
        raise PromptPlanSchemaError(
            f"too many boxes for arm {arm}: {len(boxes_raw)} > max {rules['max_objects']}")

    boxes = []
    for i, b in enumerate(boxes_raw):
        if not (isinstance(b, list) or isinstance(b, tuple)) or len(b) != 4:
            raise PromptPlanSchemaError(f"box[{i}] must be [x0,y0,x1,y1]; got {b}")
        try:
            boxes.append(Box.from_xyxy(*b))
        except ValueError as e:
            raise PromptPlanSchemaError(f"box[{i}] invalid: {e}")

    pos_raw = payload["positive_points"]
    neg_raw = payload["negative_points"]
    if not isinstance(pos_raw, list) or not isinstance(neg_raw, list):
        raise PromptPlanSchemaError("points must be lists")
    if len(pos_raw) < rules["min_positive_points"]:
        raise PromptPlanSchemaError(
            f"too few positive_points for arm {arm}: "
            f"{len(pos_raw)} < min {rules['min_positive_points']}")
    if len(pos_raw) > rules["max_positive_points"]:
        raise PromptPlanSchemaError(
            f"too many positive_points for arm {arm}: "
            f"{len(pos_raw)} > max {rules['max_positive_points']}")
    if len(neg_raw) < rules["min_negative_points"]:
        raise PromptPlanSchemaError(
            f"too few negative_points for arm {arm}: "
            f"{len(neg_raw)} < min {rules['min_negative_points']}")
    if len(neg_raw) > rules["max_negative_points"]:
        raise PromptPlanSchemaError(
            f"too many negative_points for arm {arm}: "
            f"{len(neg_raw)} > max {rules['max_negative_points']}")
    if rules["require_box_plus_points"] and not (boxes and pos_raw):
        raise PromptPlanSchemaError(
            f"arm {arm} requires at least 1 box AND >=1 positive point")
    if arm == "VLM_BOX_POINT" and len(boxes) != 1:
        raise PromptPlanSchemaError(
            f"VLM_BOX_POINT requires exactly 1 box; got {len(boxes)}")

    positive = []
    negative = []
    for i, p in enumerate(pos_raw):
        if not (isinstance(p, list) or isinstance(p, tuple)) or len(p) != 2:
            raise PromptPlanSchemaError(f"positive_point[{i}] must be [x, y]; got {p}")
        try:
            positive.append(Point(int(p[0]), int(p[1]), "positive"))
        except ValueError as e:
            raise PromptPlanSchemaError(f"positive_point[{i}] invalid: {e}")
    for i, p in enumerate(neg_raw):
        if not (isinstance(p, list) or isinstance(p, tuple)) or len(p) != 2:
            raise PromptPlanSchemaError(f"negative_point[{i}] must be [x, y]; got {p}")
        try:
            negative.append(Point(int(p[0]), int(p[1]), "negative"))
        except ValueError as e:
            raise PromptPlanSchemaError(f"negative_point[{i}] invalid: {e}")

    confidence_raw = payload["confidence"]
    try:
        confidence = float(confidence_raw)
    except (TypeError, ValueError) as e:
        raise PromptPlanSchemaError(
            f"confidence must be a finite float in [0, 1]; got {confidence_raw!r}") from e
    if not (0.0 <= confidence <= 1.0):
        raise PromptPlanSchemaError(f"confidence must be in [0, 1]: {confidence}")

    # Apply duplicate-box rejection (deterministic, drop extras; do not raise).
    boxes = reject_duplicate_boxes(boxes)
    # Bound-check boxes against the canonical 128x128 image grid.
    # Raise PromptPlanSchemaError (not raw ValueError) for any OOB box.
    bounded = []
    for b in boxes:
        try:
            b.validate_in_bounds(128)
            bounded.append(b)
        except ValueError as e:
            raise PromptPlanSchemaError(
                f"box {b} out of image bounds [0, 128]: {e}") from e
    boxes = bounded
    if rules["require_box_plus_points"] and not boxes:
        raise PromptPlanSchemaError("VLM_BOX_POINT requires a valid in-bounds box")

    try:
        return PromptPlan(
            arm_name=arm,
            dataset=str(payload["dataset"]),
            subject_id=str(payload["subject_id"]),
            slice_id=str(payload["slice_id"]),
            count_level=int(payload["count_level"]),
            realization_id=int(payload["realization_id"]),
            boxes=tuple(boxes),
            positive_points=tuple(positive),
            negative_points=tuple(negative),
            confidence=confidence,
        )
    except ValueError as e:
        raise PromptPlanSchemaError(f"PromptPlan construction failed: {e}") from e


# ============================================================================
# Deterministic prompt-plan builders (pure)
# ============================================================================

def build_det_component_plan(
    dataset: str, subject_id: str, slice_id: str,
    count_level: int, realization_id: int,
    mlem10: np.ndarray,
    threshold_quantile: float = 0.85,
    padding_px: int = 2,
    confidence: float = 1.0,
) -> PromptPlan:
    """Deterministic DET_COMPONENT plan builder (Stage 7A)."""
    boxes = det_component_boxes_from_mlem10(
        mlem10,
        image_dim=128,
        threshold_quantile=threshold_quantile,
        padding_px=padding_px,
        component_order="area_descending",
        max_objects=4,
    )
    return PromptPlan(
        arm_name="DET_COMPONENT",
        dataset=dataset,
        subject_id=subject_id,
        slice_id=slice_id,
        count_level=count_level,
        realization_id=realization_id,
        boxes=tuple(boxes),
        positive_points=tuple(),
        negative_points=tuple(),
        confidence=confidence,
    )


# ============================================================================
# Helper: deterministic arm-call counter
# ============================================================================

class CallCounter:
    """Deterministic call counter (mutable). Not thread-safe."""

    def __init__(self):
        self._n = 0

    def incr(self) -> int:
        self._n += 1
        return self._n

    @property
    def value(self) -> int:
        return self._n


def enforce_max_calls(arm: str, counter: CallCounter) -> None:
    """Raise PromptPlanSchemaError if the arm has exceeded max_calls_per_case."""
    rules = ARM_RULES[arm]
    if counter.value > rules["max_calls_per_case"]:
        raise PromptPlanSchemaError(
            f"arm {arm} exceeded max_calls_per_case={rules['max_calls_per_case']}")
