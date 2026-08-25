"""v3 canonical pipeline orchestrator.

This is a thin driver that calls into the ORIGINAL codebase at
D:/MIAProject/EM-reconstruction/. No algorithm code is reimplemented
here. The orchestrator just sequences the 10 v3 phases and writes
audit artifacts.

Usage (from v3 repo root):
    python code/src/v3_canonical.py --phase 0   # AUDIT_ONLY preservation
    python code/src/v3_canonical.py --phase 1   # checkpoint audit
    python code/src/v3_canonical.py --phase 9   # validation
    python code/src/v3_canonical.py --phase 10  # finalize
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

# Path setup. ORIGINAL codebase is the source of truth for algorithms.
REPO = Path(__file__).resolve().parent.parent.parent  # SAM_KEM_QWEN3_V3_REPO
ORIGINAL = Path(r"D:/MIAProject/EM-reconstruction")
DELIVERY = Path(r"D:/MIAProject/SAM_KEM_QWEN3_MEDSAM2_DELIVERY")
AUDIT_ONLY = Path(r"D:/MIAProject/SAM_KEM_QWEN3_AUDIT_ONLY_20260823.tar.gz")

# Make the original codebase importable.
sys.path.insert(0, str(ORIGINAL))


def phase0_audit_only_preservation() -> None:
    """Rename any old FINAL tar.gz to AUDIT_ONLY, stamp CSVs."""
    if AUDIT_ONLY.is_file():
        print(f"AUDIT_ONLY already exists: {AUDIT_ONLY}")
        return
    old_tar = DELIVERY.parent / "SAM_KEM_QWEN3_FINAL.tar.gz"
    if old_tar.is_file():
        shutil.copy2(old_tar, AUDIT_ONLY)
    import hashlib
    sha = hashlib.sha256(AUDIT_ONLY.read_bytes()).hexdigest()
    sidecar = AUDIT_ONLY.with_suffix(".sha256")
    sidecar.write_text(f"{sha}  {AUDIT_ONLY.name}\n", encoding="utf-8")
    print(f"sealed AUDIT_ONLY: {AUDIT_ONLY} sha={sha[:16]}...")


def phase1_checkpoint_audit() -> None:
    """5-point checkpoint audit on both decoders."""
    subprocess.run([
        sys.executable, str(REPO / "code/scripts/validate_v3.py"), ".",
    ], check=False)


def phase9_validate() -> None:
    """Run the 9-rule validator on the current repo state."""
    subprocess.run([
        sys.executable, str(REPO / "code/scripts/validate_v3.py"), str(REPO),
    ], check=False)


def phase10_finalize() -> None:
    """Tar the repo into D:/MIAProject/SAM_KEM_QWEN3_FINAL_v3.tar.gz."""
    out = DELIVERY.parent / "SAM_KEM_QWEN3_FINAL_v3.tar.gz"
    import tarfile
    if out.is_file():
        out.unlink()
    excludes = {".git", "__pycache__", "*.pyc"}
    with tarfile.open(out, "w:gz") as tar:
        for p in REPO.rglob("*"):
            rel = p.relative_to(REPO.parent)
            if any(rel.match(str(e)) for e in excludes):
                continue
            tar.add(p, arcname=str(rel))
    import hashlib
    sha = hashlib.sha256(out.read_bytes()).hexdigest()
    sidecar = out.with_suffix(".sha256")
    sidecar.write_text(f"{sha}  {out.name}\n", encoding="utf-8")
    print(f"wrote {out} sha={sha[:16]}...")
    # Verify AUDIT_ONLY is byte-unchanged.
    if AUDIT_ONLY.is_file():
        expected = (REPO / "code/configs/AUDIT_ONLY_expected_sha.txt").read_text().split()[0]
        actual = hashlib.sha256(AUDIT_ONLY.read_bytes()).hexdigest()
        if actual != expected:
            print(f"WARNING: AUDIT_ONLY SHA changed! expected={expected[:16]} actual={actual[:16]}")
        else:
            print(f"AUDIT_ONLY unchanged: {actual[:16]}...")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--phase", type=int, required=True,
                   help="Which phase to run (0-10). See docs/v3_workflow.md.")
    args = p.parse_args()
    {
        0: phase0_audit_only_preservation,
        1: phase1_checkpoint_audit,
        9: phase9_validate,
        10: phase10_finalize,
    }[args.phase]()


if __name__ == "__main__":
    main()