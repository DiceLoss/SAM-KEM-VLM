import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from sam_kem.artifacts import file_sha256
from sam_kem.boxes import Box
from sam_kem.display import render_prompt_png
from sam_kem.qwen import (
  REQUEST_IMPLEMENTATION_PATH, QwenAttemptError, QwenBatchFatalError,
  QwenBoxResponse, instruction_sha256, load_request_policy,
  request_and_archive_box,
)


ROOT = Path(__file__).resolve().parents[1]
POLICY_PATH = ROOT / "inputs/clean_v1/qwen_request_policy.json"


def _prompt_case(tmp_path: Path):
  source = tmp_path / "mlem10.npy"
  np.save(source, np.arange(128 * 128, dtype=np.float32).reshape(128, 128))
  image = tmp_path / "c200000.png"
  render_prompt_png(source, image)
  cfg = {
    "run_id": "test_run",
    "prompts": {"text": "框选出明亮团块。", "model": "qwen3-vl-plus"},
  }
  return image, cfg


def test_qwen_policy_retries_every_failed_attempt_and_archives_history(tmp_path, monkeypatch):
  image, cfg = _prompt_case(tmp_path)
  policy = load_request_policy(POLICY_PATH)
  calls = []

  def fake_request(image_path, **kwargs):
    calls.append((image_path, kwargs))
    if len(calls) == 1:
      raise QwenAttemptError("transient", raw_response={"error": "rate_limit"})
    return QwenBoxResponse(
      Box(1, 2, 20, 30), "request-2", "qwen3-vl-plus",
      hashlib.sha256(image.read_bytes()).hexdigest(),
      instruction_sha256(cfg["prompts"]["text"]), 0.5, {"id": "request-2"},
    )

  delays = []
  monkeypatch.setattr("sam_kem.qwen.request_box", fake_request)
  monkeypatch.setattr("sam_kem.qwen.time.sleep", delays.append)
  output = tmp_path / "response.json"
  record, success = request_and_archive_box(
    image, output, cfg=cfg, dataset="brainweb", subject_id="s1", slice_id="z1",
    count_level=200000, policy=policy, policy_path=POLICY_PATH,
  )
  document = json.loads(output.read_text(encoding="utf-8"))
  assert success and record["attempt_count"] == 2
  assert [attempt["status"] for attempt in document["attempts"]] == ["failure", "success"]
  assert document["attempts"][1]["http_status"] == 200
  assert document["attempts"][0]["raw_response"] == {"error": "rate_limit"}
  assert document["request_policy"]["sha256"] == file_sha256(POLICY_PATH)
  assert document["request_implementation"]["sha256"] == file_sha256(
    REQUEST_IMPLEMENTATION_PATH
  )
  assert delays == [2.0]


def test_qwen_policy_archives_terminal_failure_after_fixed_budget(tmp_path, monkeypatch):
  image, cfg = _prompt_case(tmp_path)
  policy = load_request_policy(POLICY_PATH)
  monkeypatch.setattr(
    "sam_kem.qwen.request_box",
    lambda *args, **kwargs: (_ for _ in ()).throw(QwenAttemptError("offline")),
  )
  delays = []
  monkeypatch.setattr("sam_kem.qwen.time.sleep", delays.append)
  output = tmp_path / "response.json"
  record, success = request_and_archive_box(
    image, output, cfg=cfg, dataset="hecktor", subject_id="1", slice_id="z072",
    count_level=100000, policy=policy, policy_path=POLICY_PATH,
  )
  document = json.loads(output.read_text(encoding="utf-8"))
  assert not success and record["source"] == "qwen_request_failure"
  assert len(document["attempts"]) == policy["max_attempts_per_key"]
  assert delays == [2.0, 8.0]


def test_nonretryable_http_error_aborts_after_one_archived_attempt(tmp_path, monkeypatch):
  image, cfg = _prompt_case(tmp_path)
  policy = load_request_policy(POLICY_PATH)
  monkeypatch.setattr(
    "sam_kem.qwen.request_box",
    lambda *args, **kwargs: (_ for _ in ()).throw(QwenAttemptError(
      "unauthorized", raw_response={"http_status": 401}, http_status=401,
      request_id="auth-request", latency_seconds=0.25,
    )),
  )
  delays = []
  monkeypatch.setattr("sam_kem.qwen.time.sleep", delays.append)
  output = tmp_path / "response.json"
  with pytest.raises(QwenBatchFatalError):
    request_and_archive_box(
      image, output, cfg=cfg, dataset="brainweb", subject_id="s1", slice_id="z1",
      count_level=100000, policy=policy, policy_path=POLICY_PATH,
    )
  document = json.loads(output.read_text(encoding="utf-8"))
  assert document["record"]["source"] == "qwen_batch_configuration_failure"
  assert len(document["attempts"]) == 1
  assert document["attempts"][0]["http_status"] == 401
  assert document["attempts"][0]["request_id"] == "auth-request"
  assert document["attempts"][0]["latency_seconds"] == 0.25
  assert delays == []
