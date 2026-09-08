import csv
from pathlib import Path
from types import SimpleNamespace

from sam_kem.sweep_evaluation import SWEEP_FIELDS, parse_sweep_grid
from sam_kem.sweep_selection import _expected_parameters
from sam_kem.sweep_union_selection import stream_subject_balanced_union_selection
import sam_kem.sweep_union_selection as union


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
          for sigma, radius, neighbours, iterations, alpha in _expected_parameters(
            method, grid,
          ):
            score = sigma + radius + neighbours + iterations + alpha
            writer.writerow({
              "run_id": "run", "dataset": dataset, "subject_id": subject,
              "slice_id": "z1", "realization_id": 0, "count_level": 100_000,
              "method": method, "split": "validation", "sigma": sigma,
              "radius": radius, "neighbours": neighbours,
              "iterations": iterations, "osem_subsets": 1,
              "effective_em_updates": iterations, "alpha": alpha,
              "nrmse": score, "ssim": 1.0 - score / 100,
            })


def _write_incremental(path: Path) -> None:
  with path.open("w", encoding="utf-8", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=SWEEP_FIELDS, lineterminator="\n")
    writer.writeheader()
    for subject in ("s1", "s2"):
      writer.writerow({
        "run_id": "run", "dataset": "brainweb", "subject_id": subject,
        "slice_id": "z1", "realization_id": 0, "count_level": 100_000,
        "method": "SelfKEM", "split": "validation", "sigma": 2.0,
        "radius": 1, "neighbours": 0, "iterations": 2,
        "osem_subsets": 1, "effective_em_updates": 2, "alpha": 0.0,
        "nrmse": 0.1, "ssim": 0.99,
      })


def test_union_selection_detects_winning_expanded_boundary(tmp_path, monkeypatch):
  original = tmp_path / "original.csv"
  incremental = tmp_path / "incremental.csv"
  _write_original(original)
  _write_incremental(incremental)
  plan = {"groups": []}
  monkeypatch.setattr(union, "_plan_candidate_maps", lambda _plan: (
    {("brainweb", 100_000, "SelfKEM"): {(2.0, 1, 0, 2, 0.0)}},
    {("brainweb", 100_000, "SelfKEM", (2.0, 1, 0, 2, 0.0)): ("sigma",)},
    {("brainweb", 100_000, "SelfKEM"): (0.5, 1, 0, 2, 0.0)},
  ))
  result = stream_subject_balanced_union_selection(
    original, incremental, _protocol(), plan, "run",
  )
  assert result["incremental_rows_streamed"] == 2
  assert result["expanded_boundary_selection_count"] == 1
  assert result["expanded_boundary_axis_counts"] == {"sigma": 1}
