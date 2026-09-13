import csv
from pathlib import Path
from types import SimpleNamespace

import sam_kem.sweep_union_selection_round5 as round5_union
from sam_kem.sweep_evaluation import SWEEP_FIELDS, parse_sweep_grid
from sam_kem.sweep_selection import _expected_parameters
from sam_kem.sweep_union_selection_round5 import (
  stream_subject_balanced_round5_union_selection,
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
  identity = ("brainweb", 100_000, "SelfKEM")
  values = [
    (2.0, 1, 0, 2, 0.0),
    (4.0, 1, 0, 2, 0.0),
    (8.0, 1, 0, 2, 0.0),
    (16.0, 1, 0, 2, 0.0),
    (32.0, 1, 0, 2, 0.0),
  ]
  functions = (
    "_plan_candidate_maps", "_round2_plan_candidate_maps",
    "_round3_plan_candidate_maps", "_round4_plan_candidate_maps",
  )
  for index, name in enumerate(functions):
    monkeypatch.setattr(round5_union, name, lambda _plan, index=index: (
      {identity: {values[index]}}, {}, {identity: values[max(index - 1, 0)]},
    ))
  monkeypatch.setattr(round5_union, "_round5_plan_candidate_maps", lambda _plan: (
    {identity: {values[4]}},
    {(*identity, values[4]): ("sigma",)},
    {identity: values[3]},
  ))


def _run_selection(tmp_path, monkeypatch, *, round4_score: float, round5_score: float):
  paths = [tmp_path / f"source-{index}.csv" for index in range(6)]
  _write_original(paths[0])
  for path, parameters, score in zip(
    paths[1:],
    (
      (2.0, 1, 0, 2, 0.0), (4.0, 1, 0, 2, 0.0),
      (8.0, 1, 0, 2, 0.0), (16.0, 1, 0, 2, 0.0),
      (32.0, 1, 0, 2, 0.0),
    ),
    (0.5, 0.4, 0.3, round4_score, round5_score),
  ):
    _write_shell(path, parameters, score)
  _patch_candidate_maps(monkeypatch)
  return stream_subject_balanced_round5_union_selection(
    *paths, _protocol(), {}, {}, {}, {}, {}, "run",
  )


def test_round5_union_detects_winning_new_boundary(tmp_path, monkeypatch):
  result = _run_selection(
    tmp_path, monkeypatch, round4_score=0.2, round5_score=0.1,
  )
  assert result["union_rows_streamed"] > 10
  assert result["round5_incremental_rows_streamed"] == 2
  assert result["expanded_boundary_selection_count"] == 1
  assert result["expanded_boundary_axis_counts"] == {"sigma": 1}


def test_round5_union_resolves_boundary_when_prior_winner_remains(tmp_path, monkeypatch):
  result = _run_selection(
    tmp_path, monkeypatch, round4_score=0.1, round5_score=0.2,
  )
  assert result["round5_incremental_candidate_selected_count"] == 0
  assert result["resolved_reviewed_boundary_group_count"] == 1
  assert result["expanded_boundary_selections"] == []
