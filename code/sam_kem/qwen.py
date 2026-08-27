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
from .artifacts import atomic_write_json, file_sha256


DEFAULT_ENDPOINT = "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"


class QwenAttemptError(RuntimeError):
  def __init__(
    self, message: str, *, raw_response=None, http_status: int | None = None,
    request_id: str = "", latency_seconds: float | None = None,
  ):
    super().__init__(message)
    self.raw_response = raw_response
    self.http_status = http_status
    self.request_id = request_id
    self.latency_seconds = latency_seconds


class QwenBatchFatalError(RuntimeError):
  """A global request-configuration failure that must never become fallback."""


@dataclass(frozen=True)
class QwenBoxResponse:
  box: Box | None
  request_id: str
  model: str
  image_sha256: str
  prompt_sha256: str
  latency_seconds: float
  raw_response: dict


def load_request_policy(path: Path) -> dict:
  policy = json.loads(path.read_text(encoding="utf-8"))
  attempts = int(policy.get("max_attempts_per_key", 0))
  backoff = [float(value) for value in policy.get("backoff_seconds_before_attempt", [])]
  if (
    policy.get("schema_version") != 1
    or not str(policy.get("policy_id", ""))
    or policy.get("endpoint") != DEFAULT_ENDPOINT
    or float(policy.get("temperature", -1)) != 0.0
    or float(policy.get("timeout_seconds", 0)) <= 0
    or attempts < 1
    or len(backoff) != attempts
    or backoff[0] != 0.0
    or any(value < 0 for value in backoff)
    or int(policy.get("max_consecutive_live_failed_keys", 0)) < 1
    or policy.get("archive_every_attempt") is not True
    or policy.get("selection_policy") != "identical_for_every_cache_key_no_selective_retry"
    or sorted(int(value) for value in policy.get("retryable_http_statuses", []))
    != [408, 409, 425, 429]
    or [int(value) for value in policy.get("retryable_http_status_classes", [])] != [5]
    or policy.get("nonretryable_http_action")
    != "archive_one_attempt_and_abort_batch_without_fallback"
  ):
    raise ValueError(f"invalid Qwen request policy: {path}")
  return policy


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
  if not isinstance(value.get("detected"), bool):
    raise ValueError("Qwen response must contain boolean detected")
  return value


def _portable_policy_path(path: Path) -> str:
  try:
    return path.resolve().relative_to(Path.cwd().resolve()).as_posix()
  except ValueError:
    return str(path.resolve())


def request_box(
  image_path: Path,
  *,
  prompt: str,
  model: str = "qwen3-vl-plus",
  endpoint: str = DEFAULT_ENDPOINT,
  api_key: str | None = None,
  timeout_seconds: float = 120,
  temperature: float = 0,
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
    "temperature": temperature,
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
      raw_body = response.read().decode("utf-8", errors="replace")
      header_request_id = response.headers.get("x-request-id", "")
  except urllib.error.HTTPError as error:
    latency = time.perf_counter() - start
    detail = error.read().decode("utf-8", errors="replace")
    request_id = str(error.headers.get("x-request-id", "")) if error.headers else ""
    raise QwenAttemptError(
      f"DashScope HTTP {error.code}: {detail[:500]}",
      raw_response={"http_status": error.code, "body": detail[:5000]},
      http_status=int(error.code), request_id=request_id, latency_seconds=latency,
    ) from error
  try:
    body = json.loads(raw_body)
  except json.JSONDecodeError as error:
    raise QwenAttemptError(
      "DashScope returned non-JSON content",
      raw_response={"body": raw_body[:5000]},
    ) from error
  latency = time.perf_counter() - start
  try:
    content = body["choices"][0]["message"]["content"]
    if isinstance(content, list):
      content = "".join(
        str(item.get("text", "")) for item in content if isinstance(item, dict)
      )
    parsed = _extract_json(content)
    box = None
    if parsed.get("detected") is True:
      box = Box(*(int(parsed[name]) for name in ("xmin", "ymin", "xmax", "ymax")))
  except Exception as error:
    raise QwenAttemptError(
      f"invalid Qwen response: {error!r}", raw_response=body,
    ) from error
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
  policy: dict, policy_path: Path,
) -> tuple[dict, bool]:
  """Run one fixed-policy request task and archive every attempt immutably."""
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
  policy_record = {
    "policy_id": str(policy["policy_id"]),
    "path": _portable_policy_path(policy_path),
    "sha256": file_sha256(policy_path),
  }
  attempts = []
  raw_response = None
  success = False
  response = None
  batch_fatal_error = None
  backoff = [float(value) for value in policy["backoff_seconds_before_attempt"]]
  for attempt_index in range(1, int(policy["max_attempts_per_key"]) + 1):
    if backoff[attempt_index - 1] > 0:
      time.sleep(backoff[attempt_index - 1])
    response = None
    try:
      response = request_box(
        image_path, prompt=prompt, model=model,
        endpoint=str(policy["endpoint"]),
        timeout_seconds=float(policy["timeout_seconds"]),
        temperature=float(policy["temperature"]),
      )
      if (
        not response.request_id
        or response.model != model
        or response.image_sha256 != image_sha
        or response.prompt_sha256 != identity["prompt_sha256"]
      ):
        raise ValueError("Qwen response identity differs from the configured request")
      if response.box is not None:
        response.box.validate(width=128, height=128)
      attempts.append({
        "attempt_index": attempt_index, "status": "success",
        "request_id": response.request_id,
        "latency_seconds": response.latency_seconds,
      })
      record = {
        **identity,
        "source": "qwen3-vl-plus" if response.box is not None else "qwen_no_detection",
        "request_id": response.request_id, "latency_seconds": response.latency_seconds,
        "attempt_count": attempt_index,
        "box": None if response.box is None else response.box.__dict__,
      }
      raw_response = response.raw_response
      success = True
      break
    except Exception as error:
      failed_attempt = {
        "attempt_index": attempt_index, "status": "failure", "error": repr(error),
      }
      status = getattr(error, "http_status", None)
      if status is not None:
        failed_attempt["http_status"] = int(status)
      error_request_id = str(getattr(error, "request_id", ""))
      if error_request_id:
        failed_attempt["request_id"] = error_request_id
      error_latency = getattr(error, "latency_seconds", None)
      if error_latency is not None:
        failed_attempt["latency_seconds"] = float(error_latency)
      error_response = getattr(error, "raw_response", None)
      if error_response is not None:
        failed_attempt["raw_response"] = error_response
      elif response is not None:
        failed_attempt["raw_response"] = response.raw_response
        failed_attempt["request_id"] = response.request_id
        failed_attempt["latency_seconds"] = response.latency_seconds
      attempts.append(failed_attempt)
      if status is not None:
        retryable = (
          int(status) in {int(value) for value in policy["retryable_http_statuses"]}
          or int(status) // 100 in {
            int(value) for value in policy["retryable_http_status_classes"]
          }
        )
        if not retryable:
          batch_fatal_error = error
          break
  if not success:
    record = {
      **identity,
      "source": (
        "qwen_batch_configuration_failure"
        if batch_fatal_error is not None else "qwen_request_failure"
      ),
      "request_id": "",
      "latency_seconds": "", "box": None, "attempt_count": len(attempts),
      "error": attempts[-1]["error"],
    }
  payload = {"record": record, "attempts": attempts, "request_policy": policy_record}
  if raw_response is not None:
    payload["raw_response"] = raw_response
  atomic_write_json(output_path, payload)
  if batch_fatal_error is not None:
    raise QwenBatchFatalError(
      "non-retryable Qwen HTTP error archived; aborting batch without fallback"
    ) from batch_fatal_error
  return record, success
