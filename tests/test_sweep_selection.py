import csv
from pathlib import Path
from types import SimpleNamespace

from sam_kem.sweep_evaluation import SWEEP_FIELDS
from sam_kem.sweep_selection import stream_subject_balanced_selection


def _protocol():
  return SimpleNamespace(
    methods=("MLEM", "OSEM", "Bowsher-KEM", "SelfKEM", "MR-KEM", "CT-KEM", "SAM"),
    raw={
      "search_space": {
        "sigma": [0.5, 1.0], "radius": [1, 2], "iterations": [2, 4],
        "alpha": [0.0, 0.5], "bowsher_neighbours": [2, 4],
        "bowsher_radius": [1, 2], "mlem_iterations": [1, 2],
        "osem_cycles": [1, 2],
      },
    },
  )


def _parameters(method: str):
  protocol = _protocol()
  search = protocol.raw["search_space"]
  if method == "MLEM":
    return [(0.0, 0, 0, value, 0.0) for value in search["mlem_iterations"]]
  if method == "OSEM":
    return [(0.0, 0, 0, value, 0.0) for value in search["osem_cycles"]]
  if method == "Bowsher-KEM":
    return [
      (0.0, radius, neighbours, iterations, 0.0)
      for radius in search["bowsher_radius"]
      for neighbours in search["bowsher_neighbours"]
      for iterations in search["iterations"]
    ]
  alphas = search["alpha"] if method == "SAM" else [0.0]
  return [
    (sigma, radius, 0, iterations, alpha)
    for sigma in search["sigma"]
    for radius in search["radius"]
    for iterations in search["iterations"]
    for alpha in alphas
  ]


def _write_fixture(path: Path) -> None:
  methods = {
    "brainweb": ("MLEM", "OSEM", "Bowsher-KEM", "SelfKEM", "MR-KEM", "SAM"),
    "hecktor": ("MLEM", "OSEM", "Bowsher-KEM", "SelfKEM", "CT-KEM", "SAM"),
  }
  with path.open("w", encoding="utf-8", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=SWEEP_FIELDS, lineterminator="\n")
    writer.writeheader()
    for dataset, dataset_methods in methods.items():
      for subject_index, subject in enumerate(("s1", "s2")):
        slice_count = 1 if subject == "s1" else 3
        for slice_index in range(slice_count):
          for method in dataset_methods:
            for sigma, radius, neighbours, iterations, alpha in _parameters(method):
              parameter_score = sigma + radius + neighbours + iterations + alpha
              writer.writerow({
                "run_id": "run", "dataset": dataset, "subject_id": subject,
                "slice_id": f"z{slice_index}", "realization_id": 0,
                "count_level": 100, "method": method, "split": "validation",
                "sigma": sigma, "radius": radius, "neighbours": neighbours,
                "iterations": iterations, "osem_subsets": 1,
                "effective_em_updates": iterations, "alpha": alpha,
                "nrmse": parameter_score + subject_index * 10,
                "ssim": 1.0 - parameter_score / 100,
              })


def test_stream_selection_is_subject_balanced_and_complete(tmp_path):
  sweep = tmp_path / "sweep.csv"
  _write_fixture(sweep)
  result = stream_subject_balanced_selection(sweep, _protocol(), "run")
  assert result["dataset_subject_counts"] == {"brainweb": 2, "hecktor": 2}
  assert result["dataset_selection_counts"] == {"brainweb": 6, "hecktor": 6}
  assert result["selection_count"] == 12
  mlem = next(
    row for row in result["selected_hyperparameters"]
    if row["dataset"] == "brainweb" and row["method"] == "MLEM"
  )
  assert mlem["iterations"] == 1
  assert mlem["validation_subject_mean_nrmse"] == 6.0
  assert result["boundary_selection_count"] > 0


def test_stream_selection_uses_ssim_tiebreaker(tmp_path):
  sweep = tmp_path / "sweep.csv"
  _write_fixture(sweep)
  rows = list(csv.DictReader(sweep.open("r", encoding="utf-8", newline="")))
  for row in rows:
    if row["dataset"] == "brainweb" and row["method"] == "MLEM":
      row["nrmse"] = "1.0"
      row["ssim"] = "0.9" if row["iterations"] == "2" else "0.8"
  with sweep.open("w", encoding="utf-8", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=SWEEP_FIELDS, lineterminator="\n")
    writer.writeheader(); writer.writerows(rows)
  result = stream_subject_balanced_selection(sweep, _protocol(), "run")
  selected = next(
    row for row in result["selected_hyperparameters"]
    if row["dataset"] == "brainweb" and row["method"] == "MLEM"
  )
  assert selected["iterations"] == 2
