"""Regression tests for the formal Stage 7 Agent zero-write gate."""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
CHECKER = REPO / "code" / "experiments" / "stage7_formal_release_check.py"
PROTOCOL = REPO / "code" / "configs" / "protocols" / "stage7_formal_agent_v1.yaml"
INPUTS = (
    "results/dev_trainmax_v1/validation_case_manifest.csv",
    "results/dev_trainmax_v1/frozen_hyperparameters.csv",
    "code/configs/prompts/stage7_formal_brainweb_v1.txt",
    "code/configs/prompts/stage7_formal_hecktor_v1.txt",
    "code/configs/protocols/stage7_formal_agent_v1.yaml",
    "code/experiments/stage7_formal_release_check.py",
)


def _fixture(tmp_path: Path) -> tuple[Path, str]:
    root = tmp_path / "repo"
    root.mkdir()
    for relative in INPUTS:
        source = REPO / relative
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    subprocess.run(["git", "init"], cwd=root, check=True, capture_output=True)
    subprocess.run(["git", "symbolic-ref", "HEAD", "refs/heads/main"],
                   cwd=root, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.name", "test"], cwd=root,
                   check=True)
    subprocess.run(["git", "config", "user.email", "test@example.invalid"],
                   cwd=root, check=True)
    subprocess.run(["git", "add", "-A"], cwd=root, check=True)
    subprocess.run(["git", "commit", "-m", "fixture"], cwd=root,
                   check=True, capture_output=True)
    head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    return root, head


def _run(root: Path, head: str, output: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(root / CHECKER.relative_to(REPO)),
         "--root", str(root),
         "--protocol", str(root / PROTOCOL.relative_to(REPO)),
         "--output", str(output), "--expected-head", head],
        cwd=root, text=True, capture_output=True, check=False,
    )


def test_formal_release_check_passes_without_creating_outputs(tmp_path):
    root, head = _fixture(tmp_path)
    output = tmp_path / "audit.json"
    proc = _run(root, head, output)
    assert proc.returncode == 0, proc.stderr + proc.stdout
    audit = json.loads(output.read_text(encoding="utf-8"))
    assert audit["status"] == "PASS_ready_for_fixed_case_smoke"
    assert audit["vlm_calls"] == 0
    assert audit["medsam_calls"] == 0
    assert audit["kem_endpoints"] == 0
    assert audit["checks"]["manifest"] == {
        "cell_count": 24,
        "exact_breakdown_match": True,
        "unique_cases": 8892,
    }
    assert audit["checks"]["budgets"]["vlm_calls_max"] == 2628
    for relative in (
        "results/dev_trainmax_v1/stage7_formal_agent_v1",
        "runs/dev_trainmax_v1/stage7_formal_agent_v1",
    ):
        assert not (root / relative).exists()


def test_formal_release_check_rejects_prompt_mutation(tmp_path):
    root, _head = _fixture(tmp_path)
    prompt = root / "code/configs/prompts/stage7_formal_brainweb_v1.txt"
    prompt.write_text(prompt.read_text(encoding="utf-8") + "mutation\n",
                      encoding="utf-8")
    subprocess.run(["git", "add", "-A"], cwd=root, check=True)
    subprocess.run(["git", "commit", "-m", "mutate"], cwd=root,
                   check=True, capture_output=True)
    head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    output = tmp_path / "blocked.json"
    proc = _run(root, head, output)
    assert proc.returncode != 0
    audit = json.loads(output.read_text(encoding="utf-8"))
    assert audit["status"] == "BLOCKED_release_invariants_failed"
    assert any("prompt_templates.brainweb" in item
               for item in audit["failures"])


def test_formal_release_check_rejects_output_inside_formal_namespace(tmp_path):
    root, head = _fixture(tmp_path)
    output = (root / "results/dev_trainmax_v1/stage7_formal_agent_v1"
              / "release.json")
    proc = _run(root, head, output)
    assert proc.returncode != 0
    audit = json.loads(output.read_text(encoding="utf-8"))
    assert not audit["checks"]["output_outside_formal_namespaces"]
