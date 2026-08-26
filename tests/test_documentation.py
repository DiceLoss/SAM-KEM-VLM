from pathlib import Path
import re
from urllib.parse import unquote


ROOT = Path(__file__).resolve().parents[1]
MARKDOWN_LINK = re.compile(r"(?<!!)\[[^]]+\]\(([^)]+)\)")


def test_required_handoff_documents_exist():
  required = {
    "README.md",
    "SCIENTIFIC_STATUS.md",
    "CONTRIBUTING.md",
    "data/MANIFEST.md",
    "docs/README.md",
    "docs/START_HERE_ZH.md",
    "docs/REPOSITORY_MAP.md",
    "docs/DATA_DICTIONARY.md",
    "docs/EXPERIMENT_MATRIX.md",
    "docs/EXECUTION.md",
    "docs/GPU_WORKER_START_ZH.md",
    "docs/OPERATIONS_RUNBOOK.md",
    "docs/RESEARCHER_HANDOFF_ZH.md",
    "docs/ARTIFACT_SPEC.md",
    "code/experiments/README.md",
    "code/scripts/README.md",
  }
  missing = sorted(path for path in required if not (ROOT / path).is_file())
  assert not missing


def test_local_markdown_links_resolve():
  broken = []
  for document in sorted(ROOT.rglob("*.md")):
    if any(part in {".git", ".pytest_cache"} for part in document.parts):
      continue
    content = document.read_text(encoding="utf-8")
    for raw_target in MARKDOWN_LINK.findall(content):
      target = raw_target.strip().strip("<>")
      if not target or target.startswith(("#", "http://", "https://", "mailto:")):
        continue
      target = unquote(target.split("#", 1)[0])
      candidate = (ROOT / target.lstrip("/")) if target.startswith("/") else (document.parent / target)
      if not candidate.resolve().exists():
        broken.append(f"{document.relative_to(ROOT)} -> {raw_target}")
  assert not broken, "broken local Markdown links:\n" + "\n".join(broken)


def test_retired_entry_points_are_explicitly_marked_forensic():
  retired = [
    ROOT / "docs/v3_workflow.md",
    ROOT / "docs/decisions.md",
    ROOT / "code/scripts/README.md",
    ROOT / "code/src/original/README.md",
  ]
  for document in retired:
    opening = document.read_text(encoding="utf-8")[:500].lower()
    assert "forensic" in opening, document
    assert "do not" in opening or "not execution" in opening, document


def test_active_python_does_not_import_retired_modules():
  prohibited = ("code.src", "src.original", "code.scripts", "scripts._")
  violations = []
  for source_root in (ROOT / "code/sam_kem", ROOT / "code/experiments"):
    for source in source_root.rglob("*.py"):
      content = source.read_text(encoding="utf-8")
      if any(token in content for token in prohibited):
        violations.append(str(source.relative_to(ROOT)))
  assert not violations
