"""Compatibility wrapper for the fail-closed canonical validator."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code"))

from sam_kem.audit import audit_repository


if __name__ == "__main__":
  root = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else ROOT
  issues = audit_repository(root, root / "code/configs/experiment.yaml")
  for issue in issues:
    print(f"{issue.severity.upper()} {issue.code}: {issue.message}")
  raise SystemExit(1 if issues else 0)
