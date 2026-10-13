"""Generate the frozen 20-case Stage 7 qualitative evidence package."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from run_stage7_formal_validation import _case_dir, _source_path
from sam_kem.stage7a_prompt_plan import det_component_objects

PANEL = 256
LABEL = 28


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _scale(array: np.ndarray) -> np.ndarray:
    value = np.asarray(array, dtype=np.float64)
    if value.shape != (128, 128) or not np.isfinite(value).all():
        raise ValueError("qualitative arrays must be finite 128x128 planes")
    low, high = float(value.min()), float(value.max())
    return np.zeros_like(value) if high <= low else (value - low) / (high - low)


def _rgb(array: np.ndarray, *, jet: bool) -> Image.Image:
    value = _scale(array)
    if jet:
        channels = [np.clip(1.5 - np.abs(4 * value - center), 0, 1)
                    for center in (3, 2, 1)]
        pixels = np.rint(np.stack(channels, axis=-1) * 255).astype(np.uint8)
    else:
        gray = np.rint(value * 255).astype(np.uint8)
        pixels = np.repeat(gray[..., None], 3, axis=-1)
    return Image.fromarray(pixels, mode="RGB").resize((PANEL, PANEL), Image.Resampling.NEAREST)


def _panel(array: np.ndarray, label: str, *, jet: bool = False) -> Image.Image:
    canvas = Image.new("RGB", (PANEL, PANEL + LABEL), "white")
    canvas.paste(_rgb(array, jet=jet), (0, LABEL))
    ImageDraw.Draw(canvas).text((6, 7), label, fill="black")
    return canvas


def _prompt_panel(mlem: np.ndarray, objects: list[dict], label: str) -> Image.Image:
    canvas = _panel(mlem, label, jet=False)
    draw = ImageDraw.Draw(canvas)
    scale = PANEL / 128
    for obj in objects:
        x0, y0, x1, y1 = obj["box_xyxy"]
        draw.rectangle((x0 * scale, LABEL + y0 * scale,
                        x1 * scale, LABEL + y1 * scale), outline="yellow", width=3)
        for x, y in obj["positive_points_xy"]:
            radius = 4
            draw.ellipse((x * scale - radius, LABEL + y * scale - radius,
                          x * scale + radius, LABEL + y * scale + radius), fill="lime")
        for x, y in obj["negative_points_xy"]:
            radius = 5
            draw.line((x * scale - radius, LABEL + y * scale - radius,
                       x * scale + radius, LABEL + y * scale + radius), fill="red", width=3)
            draw.line((x * scale - radius, LABEL + y * scale + radius,
                       x * scale + radius, LABEL + y * scale - radius), fill="red", width=3)
    return canvas


def _compose(panels: list[Image.Image], *, columns: int = 4) -> Image.Image:
    rows = (len(panels) + columns - 1) // columns
    canvas = Image.new("RGB", (columns * PANEL, rows * (PANEL + LABEL)), "white")
    for index, panel in enumerate(panels):
        canvas.paste(panel, ((index % columns) * PANEL,
                             (index // columns) * (PANEL + LABEL)))
    return canvas


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection", required=True, type=Path)
    parser.add_argument("--main-output", required=True, type=Path)
    parser.add_argument("--brainweb-array-root", required=True, type=Path)
    parser.add_argument("--hecktor-array-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite qualitative namespace: {args.output}")
    rows = list(csv.DictReader(args.selection.open(newline="", encoding="utf-8")))
    if len(rows) != 20:
        raise ValueError("qualitative selection must contain exactly 20 frozen cases")
    args.output.mkdir(parents=True)
    entries = []
    for row in rows:
        case_dir = _case_dir(args.main_output, row)
        result_path = case_dir / "result.json"
        result = json.loads(result_path.read_text(encoding="utf-8"))
        mlem_path = _source_path(row, "mlem10_input_path", args.brainweb_array_root,
                                 args.hecktor_array_root)
        target_path = _source_path(row, "target_path", args.brainweb_array_root,
                                   args.hecktor_array_root)
        mlem = np.load(mlem_path)
        target = np.load(target_path)
        policy_prior = np.load(case_dir / "policy_prior.npy")
        policy_recon = np.load(case_dir / "policy_reconstruction.npy")
        det_prior = np.load(case_dir / "det_prior.npy")
        det_recon = np.load(case_dir / "det_reconstruction.npy")
        agent_objects = result["policy_objects"]
        det_objects = [{
            "box_xyxy": list(value.box_xyxy),
            "positive_points_xy": [list(point) for point in value.positive_points_xy],
            "negative_points_xy": [list(point) for point in value.negative_points_xy],
        } for value in det_component_objects(mlem)]
        policy_metrics = next(item for item in result["endpoints"] if item["role"] == "policy")
        det_metrics = next(item for item in result["endpoints"]
                           if item["role"] == "paired_comparator")
        panels = [
            _panel(mlem, "MLEM-10 grayscale"), _panel(mlem, "MLEM-10 jet", jet=True),
            _prompt_panel(mlem, agent_objects, "Agent prompts"),
            _prompt_panel(mlem, det_objects, "DET prompts"),
            _panel(policy_prior, "Agent continuous prior", jet=True),
            _panel(policy_recon, f"Agent recon NRMSE={policy_metrics['nrmse']:.4f}"),
            _panel(np.abs(policy_recon - target), f"Agent abs error SSIM={policy_metrics['ssim']:.4f}", jet=True),
            _panel(target, "Reference activity"),
            _panel(det_prior, "DET continuous prior", jet=True),
            _panel(det_recon, f"DET recon NRMSE={det_metrics['nrmse']:.4f}"),
            _panel(np.abs(det_recon - target), f"DET abs error SSIM={det_metrics['ssim']:.4f}", jet=True),
            _panel(policy_recon - det_recon, "Agent - DET reconstruction", jet=True),
        ]
        filename = (f"{row['dataset']}__{row['subject_id']}__{row['slice_id']}__"
                    f"c{row['count_level']}__r{row['realization_id']}.png")
        output_path = args.output / filename
        _compose(panels).save(output_path, format="PNG", optimize=False)
        entries.append({
            "dataset": row["dataset"], "subject_id": row["subject_id"],
            "slice_id": row["slice_id"], "count_level": int(row["count_level"]),
            "realization_id": int(row["realization_id"]), "file": filename,
            "sha256": sha256_file(output_path), "source_result_sha256": sha256_file(result_path),
            "source_fallback_used": bool(result["fallback_reason"]),
        })
    manifest = {
        "artifact_kind": "stage7_formal_frozen_qualitative_evidence",
        "status": "PASS_complete", "case_count": len(entries),
        "selection_sha256": sha256_file(args.selection),
        "outcome_based_selection": False, "pet_only": True,
        "anatomical_inputs_used": False, "manual_or_ground_truth_prompts_used": False,
        "entries": entries,
    }
    manifest_path = args.output / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n",
                             encoding="utf-8")
    print(json.dumps({**manifest, "entries": f"{len(entries)} entries",
                      "manifest_sha256": sha256_file(manifest_path)}, indent=2,
                     sort_keys=True))


if __name__ == "__main__":
    main()
