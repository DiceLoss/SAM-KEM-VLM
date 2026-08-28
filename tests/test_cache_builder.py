from pathlib import Path

from sam_kem.cache_builder import _find_repository_root


def test_repository_root_is_independent_of_config_nesting(tmp_path: Path):
  (tmp_path / "pyproject.toml").write_text(
    "[project]\n", encoding="utf-8", newline="\n",
  )
  (tmp_path / "code" / "sam_kem").mkdir(parents=True)
  shallow = tmp_path / "code" / "configs" / "experiment.yaml"
  nested = tmp_path / "code" / "configs" / "generated" / "candidate.yaml"
  shallow.parent.mkdir(parents=True, exist_ok=True)
  nested.parent.mkdir(parents=True, exist_ok=True)
  shallow.write_text("run_id: shallow\n", encoding="utf-8", newline="\n")
  nested.write_text("run_id: nested\n", encoding="utf-8", newline="\n")
  assert _find_repository_root(shallow) == tmp_path
  assert _find_repository_root(nested) == tmp_path
