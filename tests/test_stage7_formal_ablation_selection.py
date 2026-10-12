import random
import sys
from pathlib import Path

EXPERIMENTS = Path(__file__).resolve().parents[1] / "code" / "experiments"
sys.path.insert(0, str(EXPERIMENTS))

# The selector is an executable experiment module rather than package API.
from select_stage7_formal_ablation_cases import CELLS, select


def test_selection_is_order_invariant_and_subject_balanced():
    rows = []
    for dataset, count in CELLS:
        for subject_index in range(7):
            for realization in range(3):
                rows.append({
                    "dataset": dataset, "count_level": str(count),
                    "subject_id": f"s{subject_index}", "slice_id": "z001",
                    "realization_id": str(realization), "split": "validation",
                    "mlem10_input_path": "m.npy", "target_path": "t.npy",
                    "sinogram_path": "s.npy", "projection_scale": "1.0",
                })
    shuffled = list(rows)
    random.Random(4).shuffle(shuffled)
    first = select(rows)
    assert first == select(shuffled)
    assert len(first) == 20
    assert len({(row["dataset"], row["count_level"], row["subject_id"])
                for row in first}) == 20
