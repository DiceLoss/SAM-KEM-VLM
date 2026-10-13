"""Create audited Stage 7 prompt, display, and repeatability analyses."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from sam_kem.stage7_formal_analysis import (
    bootstrap_mean_interval,
    holm_adjust,
    sign_flip_pvalue,
)

SEED = 20261012


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty analysis table: {path.name}")
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def case_key(row: dict) -> tuple[str, str, str, int, int]:
    return (row["dataset"], row["subject_id"], row["slice_id"],
            int(row["count_level"]), int(row["realization_id"]))


def prompt_case_contrasts(main_rows: list[dict], ablation_rows: list[dict]) -> list[dict]:
    """Return preregistered component contrasts conditional on valid source prompts."""
    grouped: dict[tuple, dict[str, dict]] = defaultdict(dict)
    for row in main_rows:
        key = case_key(row)
        if row["role"] == "policy":
            if str(row["fallback_used"]).lower() == "true":
                grouped[key]["source_fallback"] = row
            else:
                grouped[key]["full_agent"] = row
        elif row["role"] == "paired_comparator":
            grouped[key]["det_component"] = row
    for row in ablation_rows:
        grouped[case_key(row)][row["arm"]] = row
    output = []
    for key, arms in sorted(grouped.items()):
        if "source_fallback" in arms or "full_agent" not in arms:
            continue
        if key[0] == "brainweb":
            contrasts = (
                ("box_only_vs_det", "multi_box_only", "det_component"),
                ("positive_added", "full_agent", "multi_box_only"),
            )
        else:
            contrasts = (
                ("box_only_vs_det", "lesion_box_only", "det_component"),
                ("positive_added", "lesion_box_positive", "lesion_box_only"),
                ("negative_added", "full_agent", "lesion_box_positive"),
            )
        for name, treatment, reference in contrasts:
            if treatment not in arms or reference not in arms:
                raise ValueError(f"missing prompt arm for {key}: {name}")
            output.append({
                "dataset": key[0], "subject_id": key[1], "slice_id": key[2],
                "count_level": key[3], "realization_id": key[4],
                "contrast": name,
                "delta_nrmse": float(arms[treatment]["nrmse"])
                - float(arms[reference]["nrmse"]),
                "delta_ssim": float(arms[treatment]["ssim"])
                - float(arms[reference]["ssim"]),
            })
    return output


def subject_contrasts(case_rows: list[dict]) -> list[dict]:
    grouped: dict[tuple, list[dict]] = defaultdict(list)
    for row in case_rows:
        grouped[(row["dataset"], int(row["count_level"]), row["subject_id"],
                 row["contrast"])].append(row)
    return [{
        "dataset": key[0], "count_level": key[1], "subject_id": key[2],
        "contrast": key[3], "case_count": len(values),
        "mean_delta_nrmse": float(np.mean([x["delta_nrmse"] for x in values])),
        "mean_delta_ssim": float(np.mean([x["delta_ssim"] for x in values])),
    } for key, values in sorted(grouped.items())]


def infer_contrasts(subject_rows: list[dict]) -> list[dict]:
    grouped: dict[tuple, list[dict]] = defaultdict(list)
    for row in subject_rows:
        grouped[(row["dataset"], int(row["count_level"]), row["contrast"])].append(row)
    records = []
    family_indices: dict[tuple[str, str], list[int]] = defaultdict(list)
    for key, values in sorted(grouped.items()):
        for metric, favourable in (("nrmse", "negative"), ("ssim", "positive")):
            array = np.asarray([x[f"mean_delta_{metric}"] for x in values], dtype=float)
            low, high = bootstrap_mean_interval(array, samples=10000, seed=SEED)
            pvalue, mode = sign_flip_pvalue(
                array, exact_limit=20, monte_carlo_samples=100000, seed=SEED)
            index = len(records)
            family_indices[(key[0], key[2])].append(index)
            records.append({
                "dataset": key[0], "count_level": key[1], "contrast": key[2],
                "metric": metric, "favourable_direction": favourable,
                "subject_count": len(array), "mean_delta": float(array.mean()),
                "median_delta": float(np.median(array)),
                "bootstrap_ci_low": low, "bootstrap_ci_high": high,
                "sign_flip_pvalue": pvalue, "sign_flip_mode": mode,
                "favourable_subject_fraction": float(np.mean(array < 0))
                if favourable == "negative" else float(np.mean(array > 0)),
                "holm_family": f"{key[0]}:{key[2]}",
            })
    for indices in family_indices.values():
        adjusted = holm_adjust([records[index]["sign_flip_pvalue"] for index in indices])
        for index, value in zip(indices, adjusted, strict=True):
            records[index]["holm_adjusted_pvalue"] = value
    return records


def display_case_contrasts(case_results: list[dict]) -> tuple[list[dict], dict]:
    output = []
    exact = defaultdict(lambda: {"cases": 0, "objects_all_three_exact": 0,
                                 "priors_all_three_exact": 0})
    for result in case_results:
        calls = {c["view_mode"]: c for c in result["calls"] if c["family"] == "display"}
        if set(calls) != {"grayscale_only", "jet_only", "grayscale_plus_jet"}:
            raise ValueError(f"incomplete display arms: {result['case_key']}")
        values = [calls[name] for name in (
            "grayscale_only", "jet_only", "grayscale_plus_jet")]
        summary = exact[result["dataset"]]
        summary["cases"] += 1
        summary["objects_all_three_exact"] += values[0]["objects"] == values[1][
            "objects"] == values[2]["objects"]
        summary["priors_all_three_exact"] += len({x["prior_sha256"] for x in values}) == 1
        for name, treatment in (("jet_vs_gray", "jet_only"),
                                ("dual_vs_gray", "grayscale_plus_jet")):
            output.append({
                "dataset": result["dataset"], "subject_id": result["subject_id"],
                "count_level": int(result["count_level"]), "contrast": name,
                "delta_nrmse": calls[treatment]["nrmse"] - calls["grayscale_only"]["nrmse"],
                "delta_ssim": calls[treatment]["ssim"] - calls["grayscale_only"]["ssim"],
            })
    return output, dict(exact)


def box_iou(first: list[int], second: list[int]) -> float:
    x0, y0 = max(first[0], second[0]), max(first[1], second[1])
    x1, y1 = min(first[2], second[2]), min(first[3], second[3])
    intersection = max(0, x1 - x0) * max(0, y1 - y0)
    area_first = (first[2] - first[0]) * (first[3] - first[1])
    area_second = (second[2] - second[0]) * (second[3] - second[1])
    return intersection / (area_first + area_second - intersection)


def repeatability(case_results: list[dict], source_root: Path) -> dict:
    eligible = exact01 = exact02 = all_three = source_fallbacks = 0
    ious: list[float] = []
    positive_distances: list[float] = []
    negative_distances: list[float] = []
    nrmse_ranges: list[float] = []
    ssim_ranges: list[float] = []
    for result in case_results:
        if result["source_main_fallback_reason"]:
            source_fallbacks += 1
            continue
        repeats = sorted((c for c in result["calls"] if c["family"] == "repeatability"),
                         key=lambda value: value["replicate"])
        if len(repeats) != 2:
            raise ValueError(f"missing repeat calls: {result['case_key']}")
        objects = [result["source_main_objects"], repeats[0]["objects"], repeats[1]["objects"]]
        eligible += 1
        exact01 += objects[0] == objects[1]
        exact02 += objects[0] == objects[2]
        all_three += objects[0] == objects[1] == objects[2]
        if len({len(value) for value in objects}) == 1:
            for repeat in objects[1:]:
                for first, second in zip(objects[0], repeat, strict=True):
                    ious.append(box_iou(first["box_xyxy"], second["box_xyxy"]))
                    positive_distances.append(float(np.linalg.norm(
                        np.asarray(first["positive_points_xy"][0])
                        - np.asarray(second["positive_points_xy"][0]))))
                    if len(first["negative_points_xy"]) == len(second["negative_points_xy"]):
                        negative_distances.extend(float(np.linalg.norm(
                            np.asarray(a) - np.asarray(b))) for a, b in zip(
                                first["negative_points_xy"], second["negative_points_xy"],
                                strict=True))
        source_path = (source_root / "cases" / result["dataset"] / result["subject_id"]
                       / result["slice_id"] / f"c{result['count_level']}"
                       / f"r{result['realization_id']}" / "result.json")
        source = json.loads(source_path.read_text(encoding="utf-8"))
        policy = next(item for item in source["endpoints"] if item["role"] == "policy")
        nrmse = [policy["nrmse"], *(item["nrmse"] for item in repeats)]
        ssim = [policy["ssim"], *(item["ssim"] for item in repeats)]
        nrmse_ranges.append(max(nrmse) - min(nrmse))
        ssim_ranges.append(max(ssim) - min(ssim))

    def stats(values: list[float]) -> dict:
        return {"count": len(values), "mean": float(np.mean(values)),
                "median": float(np.median(values)), "max": float(np.max(values))}

    return {
        "eligible_cases": eligible, "source_fallback_cases": source_fallbacks,
        "exact_source_repeat1": exact01, "exact_source_repeat2": exact02,
        "exact_all_three": all_three,
        "box_iou": stats(ious), "positive_point_distance_pixels": stats(positive_distances),
        "negative_point_distance_pixels": stats(negative_distances),
        "nrmse_range": stats(nrmse_ranges), "ssim_range": stats(ssim_ranges),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--main-output", required=True, type=Path)
    parser.add_argument("--prompt-output", required=True, type=Path)
    parser.add_argument("--display-output", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite secondary analysis: {args.output}")
    main_rows = read_csv(args.main_output / "per_case_endpoints.csv")
    prompt_rows = read_csv(args.prompt_output / "prompt_ablation_endpoints.csv")
    prompt_cases = prompt_case_contrasts(main_rows, prompt_rows)
    prompt_subjects = subject_contrasts(prompt_cases)
    prompt_inference = infer_contrasts(prompt_subjects)
    case_results = [json.loads(path.read_text(encoding="utf-8"))
                    for path in sorted((args.display_output / "cases").glob("**/result.json"))]
    display_rows, display_exact = display_case_contrasts(case_results)
    args.output.mkdir(parents=True)
    write_csv(args.output / "prompt_case_contrasts.csv", prompt_cases)
    write_csv(args.output / "prompt_subject_contrasts.csv", prompt_subjects)
    write_csv(args.output / "prompt_inference.csv", prompt_inference)
    write_csv(args.output / "display_case_contrasts.csv", display_rows)
    repeat = repeatability(case_results, args.main_output)
    summary = {
        "artifact_kind": "stage7_formal_secondary_analysis",
        "status": "PASS_complete", "conditioning": "valid_source_agent_response",
        "prompt_case_contrast_rows": len(prompt_cases),
        "prompt_subject_contrast_rows": len(prompt_subjects),
        "prompt_inference_rows": len(prompt_inference),
        "display_case_contrast_rows": len(display_rows),
        "display_exactness": display_exact, "repeatability": repeat,
        "source_sha256": {
            "main_integrity": sha256_file(args.main_output / "integrity_audit.json"),
            "prompt_integrity": sha256_file(args.prompt_output / "integrity_audit.json"),
            "display_integrity": sha256_file(args.display_output / "integrity_audit.json"),
        },
    }
    (args.output / "secondary_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    audit = {**summary, "output_sha256": {
        name: sha256_file(args.output / name) for name in (
            "prompt_case_contrasts.csv", "prompt_subject_contrasts.csv",
            "prompt_inference.csv", "display_case_contrasts.csv",
            "secondary_summary.json")}}
    (args.output / "integrity_audit.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(audit, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
