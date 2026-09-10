import csv
from pathlib import Path
from types import SimpleNamespace

import sam_kem.sweep_union_selection_round2 as round2_union
from sam_kem.sweep_evaluation import SWEEP_FIELDS, parse_sweep_grid
from sam_kem.sweep_selection import _expected_parameters
from sam_kem.sweep_union_selection_round2 import (
  stream_subject_balanced_round2_union_selection,
)


def _protocol():
  return SimpleNamespace(
    methods=("MLEM", "OSEM", "Bowsher-KEM", "SelfKEM", "MR-KEM", "CT-KEM", "SAM"),
    raw={"search_space": {
      "sigma": [0.5, 1.0], "radius": [1, 2], "iterations": [2, 4],
      "alpha": [0.0, 0.5], "bowsher_neighbours": [2, 4],
      "bowsher_radius": [1, 2], "mlem_iterations": [1, 2],
      "osem_cycles": [1, 2],
    }},
  )


def _row(*, dataset: str, subject: str, method: str, parameters: tuple, score: float):
  sigma, radius, neighbours, iterations, alpha = parameters
  return {
    "run_id": "run", "dataset": dataset, "subject_id": subject,
    "slice_id": "z1", "realization_id": 0, "count_level": 100_000,
    "method": method, "split": "validation", "sigma": sigma,
    "radius": radius, "neighbours": neighbours, "iterations": iterations,
    "osem_subsets": 1, "effective_em_updates": iterations, "alpha": alpha,
    "nrmse": score, "ssim": 1.0 - score / 100,
  }


def _write_original(path: Path) -> None:
  protocol = _protocol()
  grid = parse_sweep_grid(protocol)
  methods = {
    "brainweb": tuple(method for method in protocol.methods if method != "CT-KEM"),
    "hecktor": tuple(method for method in protocol.methods if method != "MR-KEM"),
  }
  with path.open("w", encoding="utf-8", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=SWEEP_FIELDS, lineterminator="\n")
    writer.writeheader()
    for dataset, dataset_methods in methods.items():
      for subject in ("s1", "s2"):
        for method in dataset_methods:
          for parameters in _expected_parameters(method, grid):
            writer.writerow(_row(
              dataset=dataset, subject=subject, method=method,
              parameters=parameters, score=sum(parameters),
            ))


def _write_shell(path: Path, parameters: tuple, score: float) -> None:
  with path.open("w", encoding="utf-8", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=SWEEP_FIELDS, lineterminator="\n")
    writer.writeheader()
    for subject in ("s1", "s2"):
      writer.writerow(_row(
        dataset="brainweb", subject=subject, method="SelfKEM",
        parameters=parameters, score=score,
      ))


def _patch_candidate_maps(monkeypatch):
  round1_parameters = (2.0, 1, 0, 2, 0.0)
  round2_parameters = (4.0, 1, 0, 2, 0.0)
  identity = ("brainweb", 100_000, "SelfKEM")
  monkeypatch.setattr(round2_union, "_plan_candidate_maps", lambda _plan: (
    {identity: {round1_parameters}}, {}, {identity: (0.5, 1, 0, 2, 0.0)},
  ))
  monkeypatch.setattr(round2_union, "_round2_plan_candidate_maps", lambda _plan: (
    {identity: {round2_parameters}},
    {(*identity, round2_parameters): ("sigma",)},
    {identity: round1_parameters},
  ))


def test_round2_union_detects_winning_new_boundary(tmp_path, monkeypatch):
  original = tmp_path / "original.csv"
  round1 = tmp_path / "round1.csv"
  round2 = tmp_path / "round2.csv"
  _write_original(original)
  _write_shell(round1, (2.0, 1, 0, 2, 0.0), 0.2)
  _write_shell(round2, (4.0, 1, 0, 2, 0.0), 0.1)
  _patch_candidate_maps(monkeypatch)

  result = stream_subject_balanced_round2_union_selection(
    original, round1, round2, _protocol(), {}, {}, "run",
  )

  assert result["union_rows_streamed"] > 4
  assert result["round2_incremental_rows_streamed"] == 2
  assert result["expanded_boundary_selection_count"] == 1
  assert result["expanded_boundary_axis_counts"] == {"sigma": 1}


def test_round2_union_resolves_boundary_when_prior_winner_remains(tmp_path, monkeypatch):
  original = tmp_path / "original.csv"
  round1 = tmp_path / "round1.csv"
  round2 = tmp_path / "round2.csv"
  _write_original(original)
  _write_shell(round1, (2.0, 1, 0, 2, 0.0), 0.1)
  _write_shell(round2, (4.0, 1, 0, 2, 0.0), 0.2)
  _patch_candidate_maps(monkeypatch)

  result = stream_subject_balanced_round2_union_selection(
    original, round1, round2, _protocol(), {}, {}, "run",
  )

  assert result["round2_incremental_candidate_selected_count"] == 0
  assert result["resolved_reviewed_boundary_group_count"] == 1
  assert result["expanded_boundary_selections"] == []
