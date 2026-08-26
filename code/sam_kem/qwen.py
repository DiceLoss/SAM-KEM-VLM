"""Auditable Qwen3-VL-Plus box requests through DashScope's compatible API."""

from __future__ import annotations

import base64
import hashlib
import json
import os
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from .boxes import Box
from .artifacts import atomic_write_json


DEFAULT_ENDPOINT = "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"


@dataclass(frozen=True)
class QwenBoxResponse:
  box: Box | None
  request_id: str
  model: str
  image_sha256: str
  prompt_sha256: str
  latency_seconds: float
  raw_response: dict


def _sha256(data: bytes) -> str:
  return hashlib.sha256(data).hexdigest()


def build_instruction(prompt: str) -> str:
  return (
    prompt + " 只输出严格JSON："
    '{"detected":true,"xmin":整数,"ymin":整数,"xmax":整数,"ymax":整数}；'
    '无明亮团块时输出{"detected":false}。坐标使用输入图像像素；'
    'xmin/ymin包含边界，xmax/ymax为不包含的右/下边界，范围为0到128。'
  )


def instruction_sha256(prompt: str) -> str:
  return _sha256(build_instruction(prompt).encode("utf-8"))


def _extract_json(content: str) -> dict:
  cleaned = content.strip()
  fenced = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", cleaned, flags=re.DOTALL)
  if fenced:
    cleaned = fenced.group(1)
  value = json.loads(cleaned)
  if not isinstance(value, dict):
    raise ValueError("Qwen response must be a JSON object")
  return value


def request_box(
  image_path: Path,
  *,
  prompt: str,
  model: str = "qwen3-vl-plus",
  endpoint: str = DEFAULT_ENDPOINT,
  api_key: str | None = None,
  timeout_seconds: float = 120,
) -> QwenBoxResponse:
  image_bytes = image_path.read_bytes()
  from PIL import Image
  with Image.open(image_path) as image:
    if image.size != (128, 128):
      raise ValueError(f"Qwen coordinate protocol requires a 128x128 image, got {image.size}")
  suffix = image_path.suffix.lower()
  mime = "image/png" if suffix == ".png" else "image/jpeg"
  instruction = build_instruction(prompt)
  payload = {
    "model": model,
    "temperature": 0,
    "messages": [{
      "role": "user",
      "content": [
        {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{base64.b64encode(image_bytes).decode('ascii')}"}},
        {"type": "text", "text": instruction},
      ],
    }],
  }
  token = api_key or os.environ.get("DASHSCOPE_API_KEY")
  if not token:
    raise RuntimeError("DASHSCOPE_API_KEY is not set")
  request = urllib.request.Request(
    endpoint,
    data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
    headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
    method="POST",
  )
  start = time.perf_counter()
  try:
    with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
      body = json.loads(response.read().decode("utf-8"))
      header_request_id = response.headers.get("x-request-id", "")
  except urllib.error.HTTPError as error:
    detail = error.read().decode("utf-8", errors="replace")
    raise RuntimeError(f"DashScope HTTP {error.code}: {detail[:500]}") from error
  latency = time.perf_counter() - start
  content = body["choices"][0]["message"]["content"]
  if isinstance(content, list):
    content = "".join(
      str(item.get("text", "")) for item in content if isinstance(item, dict)
    )
  parsed = _extract_json(content)
  box = None
  if parsed.get("detected") is True:
    box = Box(*(int(parsed[name]) for name in ("xmin", "ymin", "xmax", "ymax")))
  return QwenBoxResponse(
    box=box,
    request_id=str(body.get("id") or header_request_id),
    model=str(body.get("model") or model),
    image_sha256=_sha256(image_bytes),
    prompt_sha256=instruction_sha256(prompt),
    latency_seconds=latency,
    raw_response=body,
  )


def request_and_archive_box(
  image_path: Path, output_path: Path, *, cfg: dict, dataset: str,
  subject_id: str, slice_id: str, count_level: int,
) -> tuple[dict, bool]:
  """Make exactly one configured request and immutably archive success/failure."""
  if output_path.exists():
    raise FileExistsError(f"refusing to repeat an archived Qwen attempt: {output_path}")
  metadata_path = image_path.with_suffix(".json")
  if not metadata_path.is_file():
    raise FileNotFoundError(f"prompt-render provenance missing: {metadata_path}")
  image_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
  image_sha = _sha256(image_path.read_bytes())
  if (
    image_metadata.get("schema_version") != 2
    or image_metadata.get("artifact_kind") != "vlm_prompt_render"
    or image_metadata.get("png_sha256") != image_sha
  ):
    raise ValueError("prompt PNG differs from its rendering provenance")
  prompt = str(cfg["prompts"]["text"])
  model = str(cfg["prompts"]["model"])
  identity = {
    "run_id": str(cfg["run_id"]), "dataset": dataset,
    "subject_id": subject_id, "slice_id": slice_id,
    "count_level": int(count_level), "image_path": str(image_path),
    "image_sha256": image_sha,
    "source_mlem10_sha256": image_metadata["source_mlem10_sha256"],
    "prompt_sha256": instruction_sha256(prompt), "model": model,
  }
  raw_response = None
  try:
    response = request_box(image_path, prompt=prompt, model=model)
    if response.box is not None:
      response.box.validate(width=128, height=128)
    record = {
      **identity,
      "source": "qwen3-vl-plus" if response.box is not None else "qwen_no_detection",
      "request_id": response.request_id, "latency_seconds": response.latency_seconds,
      "box": None if response.box is None else response.box.__dict__,
    }
    raw_response = response.raw_response
    success = True
  except Exception as error:
    record = {
      **identity, "source": "qwen_request_failure", "request_id": "",
      "latency_seconds": "", "box": None, "error": repr(error),
    }
    success = False
  payload = {"record": record}
  if raw_response is not None:
    payload["raw_response"] = raw_response
  atomic_write_json(output_path, payload)
  return record, success
