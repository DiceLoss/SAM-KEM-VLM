from pathlib import Path

from sam_kem.experiments import Case
from sam_kem.sweep_sharding import group_validation_cases


def _case(dataset: str, subject: str, slice_id: str, count: int, realization: int) -> Case:
  placeholder = Path("unused.npy")
  return Case(
    run_id="run", split_id="split", split_sha256="a" * 64,
    dataset=dataset, subject_id=subject, slice_id=slice_id,
    realization_id=realization, count_level=count, split="validation", seed=1,
    scale_constant_C=1.0, normalization_id="fixed", normalization_offset=0.0,
    normalization_clip=False, operator_id="operator", projection_scale=1.0,
    target_path=placeholder, sinogram_path=placeholder,
    mlem10_input_path=placeholder, anatomy_path=placeholder,
    sam_prior_path=placeholder, mask_path=None,
  )


def test_slice_group_plan_is_stable_and_complete():
  cases = [
    _case(dataset, subject, slice_id, count, realization)
    for dataset, subject, slice_id in (
      ("hecktor", "10", "z072"),
      ("brainweb", "subject04", "z038"),
    )
    for count in (1_000_000, 100_000, 500_000, 200_000)
    for realization in (2, 0, 1)
  ]
  shards = group_validation_cases(list(reversed(cases)))
  assert [item["shard_id"] for item in shards] == ["shard_0000", "shard_0001"]
  assert [item["dataset"] for item in shards] == ["brainweb", "hecktor"]
  assert all(item["case_count"] == 12 for item in shards)
  assert all(item["expected_result_rows"] == 6960 for item in shards)
  assert shards[0]["case_keys"] == [
    {"count_level": count, "realization_id": realization}
    for count in (100_000, 200_000, 500_000, 1_000_000)
    for realization in (0, 1, 2)
  ]
