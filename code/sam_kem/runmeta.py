"""Immutable run metadata shared by every experiment."""

from __future__ import annotations

import hashlib
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from .artifacts import atomic_write_json, atomic_write_text


def _sha(path: Path) -> str:
  return hashlib.sha256(path.read_bytes()).hexdigest()


def write_run_metadata(output_dir: Path, *, root: Path, config_path: Path, protocol_path: Path) -> Path:
  output_dir.mkdir(parents=True, exist_ok=True)
  metadata_path = output_dir / "run_metadata.json"
  commit = subprocess.run(
    ["git", "rev-parse", "HEAD"], cwd=root, check=True,
    capture_output=True, text=True,
  ).stdout.strip()
  dirty = bool(subprocess.run(
    ["git", "status", "--porcelain"], cwd=root, check=True,
    capture_output=True, text=True,
  ).stdout.strip())
  if dirty:
    raise RuntimeError("refusing to start a scientific run from a dirty Git worktree")
  freeze = subprocess.run(
    [sys.executable, "-m", "pip", "freeze", "--all"], check=True,
    capture_output=True, text=True,
  ).stdout
  freeze_path = output_dir / "environment_freeze.txt"
  atomic_write_text(freeze_path, freeze)
  accelerator = {"torch_available": False}
  try:
    import torch
    accelerator = {
      "torch_available": True, "torch_version": torch.__version__,
      "cuda_version": torch.version.cuda,
      "cudnn_version": torch.backends.cudnn.version(),
      "cuda_available": torch.cuda.is_available(),
      "gpu_names": [
        torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())
      ],
    }
  except ImportError:
    pass
  metadata = {
    "created_utc": datetime.now(timezone.utc).isoformat(),
    "git_commit": commit, "git_worktree_dirty": dirty,
    "config_path": str(config_path.relative_to(root)), "config_sha256": _sha(config_path),
    "protocol_path": str(protocol_path.relative_to(root)), "protocol_sha256": _sha(protocol_path),
    "python": sys.version, "python_executable": sys.executable,
    "platform": platform.platform(), "pid": os.getpid(),
    "environment_freeze_sha256": _sha(freeze_path),
    "accelerator": accelerator,
  }
  atomic_write_json(metadata_path, metadata)
  return metadata_path
