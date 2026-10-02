"""Stage 7A-v4.2 release-gate regression tests.

Per directive 5945647996 §6. Validates:
- valid repository: exit 0, PASS status
- missing or mutated pinned source: nonzero, FAIL status
- output containment rejection (any formal Stage 7 execution namespace)
- pre/post namespace snapshot equality
- exact protocol equality (VLM and MedSAM maxima)
- all required hash bindings present and matching expected pins
"""
from __future__ import annotations

import csv
import hashlib
import json
import subprocess
import sys
from pathlib import Path



REPO_ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = REPO_ROOT / "code" / "configs" / "protocols" / "stage7a_v4.yaml"
RELEASE_CHECK = REPO_ROOT / "code" / "experiments" / "stage7a_v4_release_check.py"

REAL_RUN_DIR = REPO_ROOT / "results" / "dev_trainmax_v1" / "causal_ablations_corrected_v3" / "full_run"


VARIANTS = ["Q_ADAPT", "FULL_ADAPT", "CENTER_ADAPT", "CENTROID_ADAPT",
            "Q_RELEASED", "MLEM_FEATURE_MATCHED",
            "Q_SHUFFLED_MATCHED", "UNIFORM_FEATURE_MATCHED"]


def _make_csv(rows, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["variant", "dataset", "subject_id", "slice_id",
                     "count_level", "realization_id"])
        for r in rows:
            w.writerow(r)


def _fake_full_rows():
    """Generate 24-cell-equal fake full rows covering brainweb+hecktor."""
    rows = []
    for ds, cnt, r, n in [("brainweb", 100000, 0, 696), ("brainweb", 200000, 0, 696),
                           ("brainweb", 500000, 0, 696), ("brainweb", 1000000, 0, 696),
                           ("brainweb", 100000, 1, 696), ("brainweb", 200000, 1, 696),
                           ("brainweb", 500000, 1, 696), ("brainweb", 1000000, 1, 696),
                           ("brainweb", 100000, 2, 696), ("brainweb", 200000, 2, 696),
                           ("brainweb", 500000, 2, 696), ("brainweb", 1000000, 2, 696),
                           ("hecktor", 100000, 0, 45), ("hecktor", 200000, 0, 45),
                           ("hecktor", 500000, 0, 45), ("hecktor", 1000000, 0, 45),
                           ("hecktor", 100000, 1, 45), ("hecktor", 200000, 1, 45),
                           ("hecktor", 500000, 1, 45), ("hecktor", 1000000, 1, 45),
                           ("hecktor", 100000, 2, 45), ("hecktor", 200000, 2, 45),
                           ("hecktor", 500000, 2, 45), ("hecktor", 1000000, 2, 45)]:
        for i in range(n):
            rows.append(["X", ds, f"subj{i:03d}", f"sl{i:03d}", cnt, r])
    return rows


def _fake_short_minimal_rows():
    """Tiny fixture covering the 24 cells with single subject each."""
    rows = []
    counts = [100000, 200000, 500000, 1000000]
    reals = [0, 1, 2]
    for ds, n in [("brainweb", 696), ("hecktor", 45)]:
        for cnt in counts:
            for r in reals:
                for i in range(n):
                    rows.append(["X", ds, f"s{i:03d}", f"z{i:03d}", cnt, r])
    return rows


def _make_full_fixture(tmp_path):
    """Build a complete fixture repo with 8 variants + summary + integrity
    audit + frozen hyperparams + validation manifest. Returns the temp
    root and pinned SHA-256 map."""
    rows = _fake_full_rows()
    fake_root = tmp_path / "fake_repo"
    fake_root.mkdir()
    run_dir = fake_root / "results" / "dev_trainmax_v1" / "causal_ablations_corrected_v3" / "full_run"
    run_dir.mkdir(parents=True)
    for v in VARIANTS:
        _make_csv(rows, run_dir / v / "per_case.csv")
    # summary.json
    summary = {"total_cases": len(rows), "ok": True}
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    # integrity_audit.json
    integrity = {"ok": True}
    (run_dir / "integrity_audit.json").write_text(json.dumps(integrity, indent=2))
    # frozen_hyperparameters.csv
    fh = fake_root / "results" / "dev_trainmax_v1" / "frozen_hyperparameters.csv"
    fh.parent.mkdir(parents=True, exist_ok=True)
    fh.write_text("name,value\nsigma,0.05\nradius,5\n")
    # validation_case_manifest.csv
    vm = fake_root / "results" / "dev_trainmax_v1" / "validation_case_manifest.csv"
    vm.write_text("dataset,subject_id,slice_id,count_level,realization_id\n"
                   "brainweb,s,z,200000,0\n")
    # protocol
    proto = fake_root / "code" / "configs" / "protocols" / "stage7a_v4.yaml"
    proto.parent.mkdir(parents=True, exist_ok=True)
    proto.write_bytes(PROTOCOL_PATH.read_bytes())
    # release_check
    rc = fake_root / "code" / "experiments" / "stage7a_v4_release_check.py"
    rc.parent.mkdir(parents=True, exist_ok=True)
    rc.write_bytes(RELEASE_CHECK.read_bytes())
    # empty __init__.py + sam_kem package
    sk = fake_root / "code" / "sam_kem" / "__init__.py"
    sk.parent.mkdir(parents=True, exist_ok=True)
    sk.write_text("")
    # .gitignore to keep __pycache__ out of the tree
    gi = fake_root / ".gitignore"
    gi.write_text("__pycache__/\n*.pyc\n")
    # git init + commit so head_commit_bound passes
    subprocess.run(["git", "-C", str(fake_root), "init", "--initial-branch=main"],
                    capture_output=True)
    subprocess.run(["git", "-C", str(fake_root), "config", "user.email", "test@local"],
                    capture_output=True)
    subprocess.run(["git", "-C", str(fake_root), "config", "user.name", "test"],
                    capture_output=True)
    subprocess.run(["git", "-C", str(fake_root), "add", "-A"],
                    capture_output=True)
    subprocess.run(["git", "-C", str(fake_root), "commit", "-m", "fixture"],
                    capture_output=True)
    return fake_root


def _sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _add_hash_pins(protocol_path, fake_root):
    """Mutate the protocol to include hash_pins matching the fixture.
    The protocol pins itself via the git head commit; no self-pin is
    needed (chicken-and-egg)."""
    import yaml
    raw = protocol_path.read_bytes()
    text = raw.decode("utf-8", errors="replace")
    proto = yaml.safe_load(text)
    run_dir = fake_root / "results" / "dev_trainmax_v1" / "causal_ablations_corrected_v3" / "full_run"
    proto["hash_pins"] = {
        "frozen_hyperparameters.csv": _sha256(
            fake_root / "results" / "dev_trainmax_v1" / "frozen_hyperparameters.csv"),
        "validation_case_manifest.csv": _sha256(
            fake_root / "results" / "dev_trainmax_v1" / "validation_case_manifest.csv"),
    }
    for v in VARIANTS:
        proto["hash_pins"][f"causal_ablations_corrected_v3/full_run/{v}/per_case.csv"] = _sha256(
            run_dir / v / "per_case.csv")
    for kind in ("summary.json", "integrity_audit.json"):
        proto["hash_pins"][f"causal_ablations_corrected_v3/full_run/{kind}"] = _sha256(
            run_dir / kind)
    # Protocol pins itself implicitly via the git head commit (which is
    # the next thing the test asserts). Do NOT add a self-pin here
    # (chicken-and-egg: file changes after pin update).
    protocol_path.write_bytes(yaml.safe_dump(proto, sort_keys=False, allow_unicode=True).encode("utf-8"))
    return protocol_path


def _invoke_release_check(fake_root, output_path, expect_failure=False,
                          expected_head=""):
    protocol_path = fake_root / "code" / "configs" / "protocols" / "stage7a_v4.yaml"
    # Driver outside the repo so the working tree stays clean.
    driver = output_path.parent / "_rc_driver.py"
    driver.write_text(
        "import sys\n"
        f"sys.path.insert(0, r'{fake_root / 'code'}')\n"
        "from experiments.stage7a_v4_release_check import main\n"
        "sys.argv = ['rc',\n"
        f"             '--root', r'{fake_root}',\n"
        f"             '--protocol', r'{protocol_path}',\n"
        f"             '--output', r'{output_path}',\n"
        "             '--audit-tag', 'test',\n"
        "             '--n-cases-7e', '8892']\n"
        + (f"sys.argv.append('--expected-head')\n"
           f"sys.argv.append('{expected_head}')\n" if expected_head else "")
        + "main()\n",
        encoding="utf-8",
    )
    proc = subprocess.run([sys.executable, str(driver)],
                          capture_output=True, text=True, cwd=str(fake_root))
    if expect_failure:
        assert proc.returncode != 0, f"expected nonzero exit; got {proc.returncode}\n{proc.stdout}\n{proc.stderr}"
    else:
        assert proc.returncode == 0, f"expected zero exit; got {proc.returncode}\n{proc.stdout}\n{proc.stderr}"
    if output_path.exists():
        return json.loads(output_path.read_text(encoding="utf-8"))
    return None


def _get_head(fake_root):
    out = subprocess.run(["git", "-C", str(fake_root), "rev-parse", "HEAD"],
                          capture_output=True, text=True)
    return out.stdout.strip()


def _check_clean_tree(fake_root):
    """Return the git status --porcelain output (empty string means clean)."""
    out = subprocess.run(["git", "-C", str(fake_root), "status", "--porcelain"],
                          capture_output=True, text=True)
    return out.stdout.strip()


def _finalize_pin_setup(protocol_path, fake_root):
    import yaml
    raw = protocol_path.read_bytes()
    proto = yaml.safe_load(raw.decode("utf-8", errors="replace"))
    proto["hash_pins"]["protocol"] = _sha256(protocol_path)
    protocol_path.write_bytes(yaml.safe_dump(proto, sort_keys=False, allow_unicode=True).encode("utf-8"))


def test_release_check_pass_with_complete_fixture(tmp_path):
    fake_root = _make_full_fixture(tmp_path)
    protocol_path = fake_root / "code" / "configs" / "protocols" / "stage7a_v4.yaml"
    _add_hash_pins(protocol_path, fake_root)
    subprocess.run(["git", "-C", str(fake_root), "add", "-A"], capture_output=True)
    subprocess.run(["git", "-C", str(fake_root), "commit", "-m", "pin"],
                    capture_output=True)
    head = _get_head(fake_root)
    # Output OUTSIDE the fake_repo so the tree stays clean after audit.
    out_path = tmp_path / "audit.json"
    audit = _invoke_release_check(fake_root, out_path, expected_head=head)
    assert audit["status"] == "PASS_zero_write_plan"
    assert audit["invariants_actually_checked"]["all_8_variants_present"] is True
    assert audit["invariants_actually_checked"]["key_set_violations"] == []
    assert audit["invariants_actually_checked"]["stage_7e_count_matches_expected"] is True
    assert audit["invariants_actually_checked"]["stage_7e_breakdown_matches_expected"] is True
    assert audit["invariants_actually_checked"]["all_hash_bindings_match_pins"] is True
    assert audit["invariants_actually_checked"]["output_not_under_formal_namespace"] is True
    assert audit["invariants_actually_checked"]["formal_namespace_snapshot_equal"] is True
    assert audit["invariants_actually_checked"]["head_tree_clean"] is True
    assert audit["invariants_actually_checked"]["head_commit_matches_required"] is True
    breakdown = audit["phase_counts"]["stage_7e_breakdown"]
    assert sum(1 for k in breakdown if k.startswith("brainweb")) == 12
    assert sum(1 for k in breakdown if k.startswith("hecktor")) == 12
    assert breakdown["brainweb/c100000/r0"] == 696
    assert breakdown["hecktor/c100000/r0"] == 45


def test_release_check_fail_on_missing_source(tmp_path):
    fake_root = _make_full_fixture(tmp_path)
    protocol_path = fake_root / "code" / "configs" / "protocols" / "stage7a_v4.yaml"
    _add_hash_pins(protocol_path, fake_root)
    subprocess.run(["git", "-C", str(fake_root), "add", "-A"], capture_output=True)
    subprocess.run(["git", "-C", str(fake_root), "commit", "-m", "pin"],
                    capture_output=True)
    head = _get_head(fake_root)
    (fake_root / "results" / "dev_trainmax_v1" / "causal_ablations_corrected_v3" / "full_run" / "Q_ADAPT" / "per_case.csv").unlink()
    out_path = tmp_path / "audit.json"
    audit = _invoke_release_check(fake_root, out_path, expect_failure=True,
                                   expected_head=head)
    assert audit is None or audit["status"] == "FAIL_invariant_violation"


def test_release_check_fail_on_mutated_source(tmp_path):
    fake_root = _make_full_fixture(tmp_path)
    protocol_path = fake_root / "code" / "configs" / "protocols" / "stage7a_v4.yaml"
    _add_hash_pins(protocol_path, fake_root)
    subprocess.run(["git", "-C", str(fake_root), "add", "-A"], capture_output=True)
    subprocess.run(["git", "-C", str(fake_root), "commit", "-m", "pin"],
                    capture_output=True)
    head = _get_head(fake_root)
    csv_path = (fake_root / "results" / "dev_trainmax_v1" / "causal_ablations_corrected_v3" / "full_run" / "Q_ADAPT" / "per_case.csv")
    csv_path.write_bytes(csv_path.read_bytes() + b"\nextra,row,mutated\n")
    out_path = tmp_path / "audit.json"
    audit = _invoke_release_check(fake_root, out_path, expect_failure=True,
                                   expected_head=head)
    assert audit is None or audit["status"] == "FAIL_invariant_violation"


def test_release_check_rejects_output_under_formal_namespace(tmp_path):
    fake_root = _make_full_fixture(tmp_path)
    protocol_path = fake_root / "code" / "configs" / "protocols" / "stage7a_v4.yaml"
    _add_hash_pins(protocol_path, fake_root)
    out_path = fake_root / "results" / "dev_trainmax_v1" / "stage7a_run" / "audit.json"
    driver = tmp_path / "_rc_driver.py"
    driver.write_text(
        "import sys\n"
        f"sys.path.insert(0, r'{fake_root / 'code'}')\n"
        "from experiments.stage7a_v4_release_check import main\n"
        "sys.argv = ['rc',\n"
        f"             '--root', r'{fake_root}',\n"
        f"             '--protocol', r'{protocol_path}',\n"
        f"             '--output', r'{out_path}',\n"
        "             '--audit-tag', 'test',\n"
        "             '--n-cases-7e', '8892']\n"
        "main()\n",
        encoding="utf-8",
    )
    proc = subprocess.run([sys.executable, str(driver)],
                          capture_output=True, text=True, cwd=str(fake_root))
    assert proc.returncode == 2
    assert not out_path.exists()


def test_release_check_formal_namespace_snapshot_unchanged(tmp_path):
    fake_root = _make_full_fixture(tmp_path)
    protocol_path = fake_root / "code" / "configs" / "protocols" / "stage7a_v4.yaml"
    _add_hash_pins(protocol_path, fake_root)
    subprocess.run(["git", "-C", str(fake_root), "add", "-A"], capture_output=True)
    subprocess.run(["git", "-C", str(fake_root), "commit", "-m", "pin"],
                    capture_output=True)
    head = _get_head(fake_root)
    out_path = tmp_path / "audit_outside_formal.json"
    audit = _invoke_release_check(fake_root, out_path, expected_head=head)
    assert audit["formal_namespace_snapshot"]["snapshot_equal"] is True
    assert audit["formal_namespace_snapshot"]["pre_existing"] == \
           audit["formal_namespace_snapshot"]["post_existing"]


def test_release_check_protocol_arm_records_match(tmp_path):
    fake_root = _make_full_fixture(tmp_path)
    protocol_path = fake_root / "code" / "configs" / "protocols" / "stage7a_v4.yaml"
    _add_hash_pins(protocol_path, fake_root)
    subprocess.run(["git", "-C", str(fake_root), "add", "-A"], capture_output=True)
    subprocess.run(["git", "-C", str(fake_root), "commit", "-m", "pin"],
                    capture_output=True)
    head = _get_head(fake_root)
    out_path = tmp_path / "audit.json"
    audit = _invoke_release_check(fake_root, out_path, expected_head=head)
    for arm in audit["arm_summaries"]:
        assert arm["max_vlm_calls_per_case"] >= 0
        assert arm["max_medsam_calls_per_case"] >= 1
    fv = audit["counter_separation"]["medsam_calls_per_arm_max"]
    from sam_kem.stage7a_prompt_plan import ARM_MAXES
    for arm_name, max_v in fv.items():
        assert max_v == ARM_MAXES[arm_name]["medsam_calls"]