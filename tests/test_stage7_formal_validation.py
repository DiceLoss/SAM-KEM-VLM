"""Portable unit checks for formal Stage 7 routing and resume identity."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

EXPERIMENTS = Path(__file__).resolve().parents[1] / "code" / "experiments"
sys.path.insert(0, str(EXPERIMENTS))

from run_stage7_formal_validation import (
    _completed,
    _is_agent,
    _source_path,
)


def row(dataset="brainweb", count="100000"):
    return {
        "dataset": dataset, "count_level": count,
        "mlem10_input_path": (
            f"arrays/dev_trainmax_v1/{dataset}/subject/z/c{count}/r0/mlem10.npy"
        ),
    }


def test_formal_routing_is_exact():
    assert _is_agent(row("brainweb", "100000"))
    assert not _is_agent(row("brainweb", "200000"))
    assert _is_agent(row("hecktor", "1000000"))


def test_dataset_root_remapping_does_not_use_manifest_repo_root(tmp_path):
    brainweb = tmp_path / "bw"
    hecktor = tmp_path / "he"
    assert _source_path(
        row(), "mlem10_input_path", brainweb, hecktor,
    ) == brainweb / "subject/z/c100000/r0/mlem10.npy"
    assert _source_path(
        row("hecktor"), "mlem10_input_path", brainweb, hecktor,
    ) == hecktor / "subject/z/c100000/r0/mlem10.npy"


def test_resume_accepts_only_hash_bound_success(tmp_path):
    result = tmp_path / "result.json"
    result.write_text("{}\n", encoding="utf-8")
    digest = hashlib.sha256(result.read_bytes()).hexdigest()
    (tmp_path / "_SUCCESS.json").write_text(
        json.dumps({"result_sha256": digest}), encoding="utf-8",
    )
    assert _completed(tmp_path)
    result.write_text('{"changed":true}\n', encoding="utf-8")
    assert not _completed(tmp_path)
