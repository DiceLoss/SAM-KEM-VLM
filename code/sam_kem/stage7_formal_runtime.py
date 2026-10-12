"""Strict runtime helpers for the preregistered Stage 7 formal experiment."""
from __future__ import annotations

import hashlib
import io
import json
import math
from dataclasses import asdict
from typing import Any

import numpy as np

from .stage7a_prompt_plan import PromptObject


class FormalPromptError(ValueError):
    """The VLM response does not satisfy the frozen formal contract."""


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise FormalPromptError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _integer(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise FormalPromptError(f"{name} must be an integer")
    return value


def _point(value: Any, name: str) -> tuple[int, int]:
    if not isinstance(value, list) or len(value) != 2:
        raise FormalPromptError(f"{name} must be [x,y]")
    x, y = (_integer(value[0], name), _integer(value[1], name))
    if not (0 <= x < 128 and 0 <= y < 128):
        raise FormalPromptError(f"{name} lies outside the image")
    return x, y


def parse_formal_prompt(raw: str, *, dataset: str) -> tuple[PromptObject, ...]:
    """Parse exact JSON and enforce the dataset-specific frozen prompt contract."""
    try:
        value = json.loads(raw, object_pairs_hook=_unique_object)
    except (json.JSONDecodeError, FormalPromptError) as error:
        raise FormalPromptError(f"invalid JSON: {error}") from error
    if not isinstance(value, dict) or set(value) != {
        "schema_version", "objects", "abstain", "fallback_reason",
    }:
        raise FormalPromptError("top-level keys do not match the frozen schema")
    if value["schema_version"] != 4 or isinstance(value["schema_version"], bool):
        raise FormalPromptError("schema_version must equal integer 4")
    if not isinstance(value["abstain"], bool):
        raise FormalPromptError("abstain must be a boolean")
    if not isinstance(value["fallback_reason"], str):
        raise FormalPromptError("fallback_reason must be a string")
    if value["abstain"]:
        raise FormalPromptError(f"model abstained: {value['fallback_reason']}")
    if value["fallback_reason"]:
        raise FormalPromptError("fallback_reason must be empty when abstain is false")
    objects = value["objects"]
    expected_range = (3, 4) if dataset == "brainweb" else (1, 1)
    if dataset not in {"brainweb", "hecktor"} or not isinstance(objects, list):
        raise FormalPromptError("invalid dataset or objects value")
    if not expected_range[0] <= len(objects) <= expected_range[1]:
        raise FormalPromptError("object count violates the dataset contract")
    parsed: list[PromptObject] = []
    boxes: set[tuple[int, int, int, int]] = set()
    all_points: set[tuple[int, int]] = set()
    required_keys = {
        "object_id", "box_xyxy", "positive_points_xy",
        "negative_points_xy", "confidence",
    }
    for index, obj in enumerate(objects, start=1):
        if not isinstance(obj, dict) or set(obj) != required_keys:
            raise FormalPromptError("object keys do not match the frozen schema")
        if _integer(obj["object_id"], "object_id") != index:
            raise FormalPromptError("object IDs must be consecutive from 1")
        box_value = obj["box_xyxy"]
        if not isinstance(box_value, list) or len(box_value) != 4:
            raise FormalPromptError("box_xyxy must contain four integers")
        box = tuple(_integer(item, "box coordinate") for item in box_value)
        x0, y0, x1, y1 = box
        if not (0 <= x0 < x1 <= 128 and 0 <= y0 < y1 <= 128):
            raise FormalPromptError("box violates half-open image bounds")
        if box in boxes:
            raise FormalPromptError("duplicate boxes are forbidden")
        boxes.add(box)
        if not isinstance(obj["positive_points_xy"], list) or not isinstance(
            obj["negative_points_xy"], list
        ):
            raise FormalPromptError("point collections must be lists")
        positive = tuple(
            _point(point, "positive point") for point in obj["positive_points_xy"]
        )
        negative = tuple(
            _point(point, "negative point") for point in obj["negative_points_xy"]
        )
        expected_negative = (0, 0) if dataset == "brainweb" else (1, 3)
        if len(positive) != 1 or not expected_negative[0] <= len(negative) <= expected_negative[1]:
            raise FormalPromptError("point counts violate the dataset contract")
        if not (x0 <= positive[0][0] < x1 and y0 <= positive[0][1] < y1):
            raise FormalPromptError("positive point must lie inside its box")
        if dataset == "hecktor" and any(
            x0 <= x < x1 and y0 <= y < y1 for x, y in negative
        ):
            raise FormalPromptError("HECKTOR negative points must lie outside the box")
        for point in positive + negative:
            if point in all_points:
                raise FormalPromptError("duplicate or cross-sign point")
            all_points.add(point)
        confidence = obj["confidence"]
        if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
            raise FormalPromptError("confidence must be numeric")
        confidence = float(confidence)
        if not math.isfinite(confidence) or not 0 <= confidence <= 1:
            raise FormalPromptError("confidence must be finite in [0,1]")
        parsed.append(PromptObject(index, box, positive, negative, confidence))
    return tuple(parsed)


def aggregate_logits(logits: tuple[np.ndarray, ...]) -> np.ndarray:
    """Aggregate independent object logits exactly by pixelwise maximum."""
    if not logits:
        raise ValueError("at least one logit plane is required")
    arrays = tuple(np.asarray(value, dtype=np.float32) for value in logits)
    if len({value.shape for value in arrays}) != 1 or not all(
        value.ndim == 2 and np.isfinite(value).all() for value in arrays
    ):
        raise ValueError("all logit planes must be finite 2-D arrays with one shape")
    return np.maximum.reduce(arrays).astype(np.float32)


def render_pet_display(pet: np.ndarray, *, view_mode: str) -> tuple[bytes, str]:
    """Render a deterministic grayscale, jet, or side-by-side PET PNG."""
    from PIL import Image, ImageDraw

    array = np.asarray(pet, dtype=np.float64)
    if array.shape != (128, 128) or not np.isfinite(array).all():
        raise ValueError("PET display input must be a finite 128x128 array")
    low, high = float(array.min()), float(array.max())
    scaled = np.zeros_like(array) if high <= low else (array - low) / (high - low)
    gray = np.rint(scaled * 255).astype(np.uint8)
    red = np.clip(1.5 - np.abs(4 * scaled - 3), 0, 1)
    green = np.clip(1.5 - np.abs(4 * scaled - 2), 0, 1)
    blue = np.clip(1.5 - np.abs(4 * scaled - 1), 0, 1)
    jet = np.rint(np.stack((red, green, blue), axis=-1) * 255).astype(np.uint8)
    gray_image = Image.fromarray(gray, mode="L").convert("RGB").resize((512, 512))
    jet_image = Image.fromarray(jet, mode="RGB").resize((512, 512))
    if view_mode == "grayscale_plus_jet":
        canvas = Image.new("RGB", (1056, 560), "white")
        canvas.paste(gray_image, (0, 48))
        canvas.paste(jet_image, (544, 48))
        titles = ((8, "grayscale MLEM-10"), (552, "jet MLEM-10"))
    elif view_mode == "grayscale_only":
        canvas = Image.new("RGB", (512, 560), "white")
        canvas.paste(gray_image, (0, 48))
        titles = ((8, "grayscale MLEM-10"),)
    elif view_mode == "jet_only":
        canvas = Image.new("RGB", (512, 560), "white")
        canvas.paste(jet_image, (0, 48))
        titles = ((8, "jet MLEM-10"),)
    else:
        raise ValueError(f"unknown view mode: {view_mode}")
    draw = ImageDraw.Draw(canvas)
    for x, title in titles:
        draw.text((x, 16), title, fill="black")
    buffer = io.BytesIO()
    canvas.save(buffer, format="PNG", optimize=False)
    payload = buffer.getvalue()
    return payload, hashlib.sha256(payload).hexdigest()


def render_pet_views(pet: np.ndarray) -> tuple[bytes, str]:
    """Backward-compatible name for the frozen dual-view formal rendering."""
    return render_pet_display(pet, view_mode="grayscale_plus_jet")


def prompt_objects_json(objects: tuple[PromptObject, ...]) -> list[dict[str, Any]]:
    """Return JSON-safe prompt evidence."""
    return [asdict(obj) for obj in objects]
