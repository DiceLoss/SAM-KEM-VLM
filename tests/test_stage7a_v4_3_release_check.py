"""Stage 7A-v4.3 release-gate regression tests.

Per directive 5950637029 §D. Validates:
- canonical-LF pins accept LF input
- canonical-LF pins accept CRLF input (after normalization)
- semantic byte mutation fails nonzero
- valid repository: exit 0, PASS status
- missing or mutated pinned source: nonzero, FAIL status
- output containment rejection (any formal Stage 7 execution namespace)
- pre/post namespace snapshot equality (post taken AFTER audit write)
"""
from __future__ import annotations

import csv
import hashlib
import json
import subprocess
import sys
from pathlib import Path



REPO_ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = REPO_ROOT / "code" / "configs" / "protocols" / "stage7a_v4_3.yaml"
RELEASE_CHECK = REPO_ROOT / "code" / "experiments" / "stage7a_v4_3_release_check.py"


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
    rows = []
    for ds, cnt, r, n in [("brainweb", 100000, 0, 696),
                            ("brainweb", 200000, 0, 696),
                            ("brainweb", 500000, 0, 696),
                            ("brainweb", 1000000, 0, 696),
                            ("brainweb", 100000, 1, 696),
                            ("brainweb", 200000, 1, 696),
                            ("brainweb", 500000, 1, 696),
                            ("brainweb", 1000000, 1, 696),
                            ("brainweb", 100000, 2, 696),
                            ("brainweb", 200000, 2, 696),
                            ("brainweb", 500000, 2, 696),
                            ("brainweb", 1000000, 2, 696),
                            ("hecktor", 100000, 0, 45),
                            ("hecktor", 200000, 0, 45),
                            ("hecktor", 500000, 0, 45),
                            ("hecktor", 1000000, 0, 45),
                            ("hecktor", 100000, 1, 45),
                            ("hecktor", 200000, 1, 45),
                            ("hecktor", 500000, 1, 45),
                            ("hecktor", 1000000, 1, 45),
                            ("hecktor", 100000, 2, 45),
                            ("hecktor", 200000, 2, 45),
                            ("hecktor", 500000, 2, 45),
                            ("hecktor", 1000000, 2, 45)]:
        for i in range(n):
            rows.append(["X", ds, f"subj{i:03d}", f"sl{i:03d}", cnt, r])
    return rows


def _make_full_fixture(tmp_path, line_ending="lf"):
    """Build a complete fixture repo. Pass `line_ending="crlf"` to write
    files with CRLF endings (must still pass canonical LF pins)."""
    le = b"\r\n" if line_ending == "crlf" else b"\n"
    rows = _fake_full_rows()
    fake_root = tmp_path / "fake_repo"
    fake_root.mkdir()
    run_dir = fake_root / "results" / "dev_trainmax_v1" / "causal_ablations_corrected_v3" / "full_run"
    run_dir.mkdir(parents=True)
    for v in VARIANTS:
        p = run_dir / v / "per_case.csv"
        p.parent.mkdir(parents=True, exist_ok=True)
        # Build CSV bytes with the chosen line ending
        # (csv.writer's lineterminator is for joining; we build raw bytes
        # directly to support both LF and CRLF choices.)
        lines = ["variant,dataset,subject_id,slice_id,count_level,realization_id"]
        for r in rows:
            lines.append(",".join(str(x) for x in r))
        raw_text = (le.decode("ascii").join(lines) + le.decode("ascii")).encode("utf-8")
        p.write_bytes(raw_text)
    summary_text = ("total_cases," + str(len(rows)) + "\nok,true\n").encode("utf-8")
    summary_text = summary_text.replace(b"\n", le)
    (run_dir / "summary.json").write_bytes(summary_text)
    (run_dir / "integrity_audit.json").write_bytes(
        b'{"ok":true}' + le)
    fh = fake_root / "results" / "dev_trainmax_v1" / "frozen_hyperparameters.csv"
    fh.parent.mkdir(parents=True, exist_ok=True)
    fh.write_bytes((b"name,value" + le + b"sigma,0.05" + le + b"radius,5" + le))
    vm = fake_root / "results" / "dev_trainmax_v1" / "validation_case_manifest.csv"
    vm.write_bytes((b"dataset,subject_id,slice_id,count_level,realization_id"
                     + le + b"brainweb,s,z,200000,0" + le))
    proto = fake_root / "code" / "configs" / "protocols" / "stage7a_v4_3.yaml"
    proto.parent.mkdir(parents=True, exist_ok=True)
    proto.write_bytes(PROTOCOL_PATH.read_bytes())
    rc = fake_root / "code" / "experiments" / "stage7a_v4_3_release_check.py"
    rc.parent.mkdir(parents=True, exist_ok=True)
    rc.write_bytes(RELEASE_CHECK.read_bytes())
    sk = fake_root / "code" / "sam_kem" / "__init__.py"
    sk.parent.mkdir(parents=True, exist_ok=True)
    sk.write_text("")
    gi = fake_root / ".gitignore"
    gi.write_text("__pycache__/\n*.pyc\n")
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


def _canonical_lf(p):
    raw = p.read_bytes()
    if b"\r" in raw:
        if b"\r\n" in raw:
            raw = raw.replace(b"\r\n", b"\n")
    return hashlib.sha256(raw).hexdigest()


def _invoke(fake_root, output_path, expect_failure=False, expected_head=""):
    protocol_path = fake_root / "code" / "configs" / "protocols" / "stage7a_v4_3.yaml"
    driver = output_path.parent / "_rc_driver.py"
    driver.write_text(
        "import sys\n"
        f"sys.path.insert(0, r'{fake_root / 'code'}')\n"
        "from experiments.stage7a_v4_3_release_check import main\n"
        "sys.argv = ['rc',\n"
        f"             '--root', r'{fake_root}',\n"
        f"             '--protocol', r'{protocol_path}',\n"
        f"             '--output', r'{output_path}',\n"
        "             '--audit-tag', 'test',\n"
        "             '--n-cases-7e', '8892',\n"
        + (f"             '--expected-head', '{expected_head}']\n" if expected_head
           else "        ]\n")
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


def _setup_pins(fake_root):
    """Mutate the protocol to pin current source SHAs (canonical LF)."""
    import yaml
    raw = (fake_root / "code" / "configs" / "protocols" / "stage7a_v4_3.yaml").read_bytes()
    proto = yaml.safe_load(raw.decode("utf-8", errors="replace"))
    run_dir = fake_root / "results" / "dev_trainmax_v1" / "causal_ablations_corrected_v3" / "full_run"
    pins = {
        "frozen_hyperparameters.csv": _canonical_lf(
            fake_root / "results" / "dev_trainmax_v1" / "frozen_hyperparameters.csv"),
        "validation_case_manifest.csv": _canonical_lf(
            fake_root / "results" / "dev_trainmax_v1" / "validation_case_manifest.csv"),
    }
    for v in VARIANTS:
        pins[f"causal_ablations_corrected_v3/full_run/{v}/per_case.csv"] = _canonical_lf(run_dir / v / "per_case.csv")
    for kind in ("summary.json", "integrity_audit.json"):
        pins[f"causal_ablations_corrected_v3/full_run/{kind}"] = _canonical_lf(run_dir / kind)
    proto["hash_pins"] = pins
    proto["repository_lf_hash_pins"] = pins
    (fake_root / "code" / "configs" / "protocols" / "stage7a_v4_3.yaml").write_bytes(
        yaml.safe_dump(proto, sort_keys=False, allow_unicode=True).encode("utf-8"))


def _get_head(fake_root):
    out = subprocess.run(["git", "-C", str(fake_root), "rev-parse", "HEAD"],
                          capture_output=True, text=True)
    return out.stdout.strip()


def test_release_check_pass_with_lf_fixture(tmp_path):
    fake_root = _make_full_fixture(tmp_path, line_ending="lf")
    _setup_pins(fake_root)
    subprocess.run(["git", "-C", str(fake_root), "add", "-A"], capture_output=True)
    subprocess.run(["git", "-C", str(fake_root), "commit", "-m", "pin"],
                    capture_output=True)
    head = _get_head(fake_root)
    out_path = tmp_path / "audit.json"
    audit = _invoke(fake_root, out_path, expected_head=head)
    assert audit["status"] == "PASS_zero_write_plan"
    assert audit["invariants_actually_checked"]["all_hash_bindings_match_pins"] is True
    assert audit["invariants_actually_checked"]["formal_namespace_snapshot_equal"] is True


def test_release_check_pass_with_crlf_fixture(tmp_path):
    """Same logical content written with CRLF line endings must still
    pass the canonical LF pins."""
    fake_root = _make_full_fixture(tmp_path, line_ending="crlf")
    _setup_pins(fake_root)
    subprocess.run(["git", "-C", str(fake_root), "add", "-A"], capture_output=True)
    subprocess.run(["git", "-C", str(fake_root), "commit", "-m", "pin"],
                    capture_output=True)
    head = _get_head(fake_root)
    out_path = tmp_path / "audit.json"
    audit = _invoke(fake_root, out_path, expected_head=head)
    assert audit["status"] == "PASS_zero_write_plan"
    assert audit["invariants_actually_checked"]["all_hash_bindings_match_pins"] is True


def test_release_check_fail_on_semantic_byte_mutation(tmp_path):
    fake_root = _make_full_fixture(tmp_path)
    _setup_pins(fake_root)
    subprocess.run(["git", "-C", str(fake_root), "add", "-A"], capture_output=True)
    subprocess.run(["git", "-C", str(fake_root), "commit", "-m", "pin"],
                    capture_output=True)
    head = _get_head(fake_root)
    # Mutate one source CSV (semantic change, not line-ending)
    csv_path = (fake_root / "results" / "dev_trainmax_v1" / "causal_ablations_corrected_v3" / "full_run" / "Q_ADAPT" / "per_case.csv")
    raw = csv_path.read_bytes()
    csv_path.write_bytes(raw.replace(b"X,brainweb", b"Y,brainweb"))
    out_path = tmp_path / "audit.json"
    audit = _invoke(fake_root, out_path, expect_failure=True, expected_head=head)
    assert audit is None or audit["status"] == "FAIL_invariant_violation"


def test_release_check_rejects_output_under_formal_namespace(tmp_path):
    fake_root = _make_full_fixture(tmp_path)
    _setup_pins(fake_root)
    out_path = fake_root / "results" / "dev_trainmax_v1" / "stage7a_run" / "audit.json"
    driver = tmp_path / "_rc_driver.py"
    driver.write_text(
        "import sys\n"
        f"sys.path.insert(0, r'{fake_root / 'code'}')\n"
        "from experiments.stage7a_v4_3_release_check import main\n"
        "sys.argv = ['rc',\n"
        f"             '--root', r'{fake_root}',\n"
        f"             '--protocol', r'{fake_root / 'code' / 'configs' / 'protocols' / 'stage7a_v4_3.yaml'}',\n"
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


def test_release_check_post_snapshot_after_write(tmp_path):
    """The post-snapshot must be taken AFTER the audit is written.
    Verifies that the audit's existence does not count as a forbidden
    namespace."""
    fake_root = _make_full_fixture(tmp_path)
    _setup_pins(fake_root)
    subprocess.run(["git", "-C", str(fake_root), "add", "-A"], capture_output=True)
    subprocess.run(["git", "-C", str(fake_root), "commit", "-m", "pin"],
                    capture_output=True)
    head = _get_head(fake_root)
    out_path = tmp_path / "audit.json"
    audit = _invoke(fake_root, out_path, expected_head=head)
    assert audit["formal_namespace_snapshot_equal"] is True
    # pre and post should both show no formal namespaces pre-existing
    pre = audit["formal_namespace_snapshot_pre"]["pre_existing"]
    post = audit["formal_namespace_snapshot_post"]["post_existing"]
    assert pre == post
    assert pre == []  # no formal namespaces created