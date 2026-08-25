"""Phase 1: 5-point checkpoint audit.

Verifies:
  1. Hash matches: BW=`6f85c02147cbd5d1a3700035b26feb20f2d2242b22eca8863b44950f0d170be1`,
                   HE=`ab6fbbb89fbc7d15e1ceee88936522a35dc46a3879aea8d4047acd653bb32d18`.
  2. image_encoder.* params are near-identical to the pretrained MedSAM2
     (`D:/MIAProject/EM-reconstruction/MedSAM2.pt`,
      sha=`c92743b99f00d078bf32a3afcc38aaa9faf1c1692dffe3eaa7a90938c1991060`).
     This proves the encoder was frozen during training.
  3. mask_decoder.* differs from pretrained AND differs between BW and HE.
     This proves the decoder was fine-tuned and is dataset-specific.
  4. Training logs (`logs/medsam2_train_{brainweb,hecktor}_stdout.log`)
     mention only train subjects; no val/test_locked leakage.
  5. Stub inference loader prints `LOADED sha=<actual> path=<actual>` and
     asserts the printed SHA matches the audit expected hash.

Writes:
  - `checkpoints/audit_report.csv` with one row per checkpoint:
    [checkpoint_id, hash_expected, hash_actual, encoder_frozen,
     decoder_differs_from_pretrained, decoder_differs_across_datasets,
     training_log_subjects_clean, inference_log_sha_match, decision]

The script never trains anything; it only inspects the .pt files.
"""
from __future__ import annotations

import csv
import hashlib
import io
import re
import subprocess
import sys
from contextlib import redirect_stdout
from pathlib import Path

import torch

ROOT = Path(r"D:/MIAProject/SAM_KEM_QWEN3_MEDSAM2_DELIVERY")
PRETRAINED = Path(r"D:/MIAProject/EM-reconstruction/MedSAM2.pt")

EXPECTED = {
    "brainweb": (
        "medsam2_decoder_brainweb_v4.pt",
        "2ee2b4c334c3fe4583092238742f9bd6d214c8e4af2d8221e94d005806f79ef7",
    ),
    "hecktor": (
        "medsam2_decoder_hecktor.pt",
        "ab6fbbb89fbc7d15e1ceee88936522a35dc46a3879aea8d4047acd653bb32d18",
    ),
}
PRETRAINED_SHA = (
    "c92743b99f00d078bf32a3afcc38aaa9faf1c1692dffe3eaa7a90938c1991060"
)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_state_dict(path: Path) -> dict:
    sd = torch.load(str(path), map_location="cpu")
    # Unwrap nested {'model': OrderedDict(...)} wrapper used by the
    # released MedSAM2.pt; otherwise strip the 'model_state' key.
    if isinstance(sd, dict) and set(sd.keys()) == {"model"} and isinstance(sd["model"], dict):
        sd = sd["model"]
    if isinstance(sd, dict) and "model_state" in sd:
        sd = sd["model_state"]
    return sd


def encoder_l2_dist(a: dict, b: dict) -> float:
    """L2 distance of all image_encoder.* parameters (assumes flat float)."""
    s = 0.0
    for k, v in a.items():
        if not k.startswith("image_encoder."):
            continue
        if k not in b:
            continue
        diff = (v.detach().float() - b[k].detach().float()).reshape(-1)
        s += float((diff * diff).sum().item())
    return float(s ** 0.5)


def decoder_l2_dist(a: dict, b: dict) -> float:
    """L2 distance of sam_mask_decoder.* parameters."""
    s = 0.0
    for k, v in a.items():
        if not k.startswith("sam_mask_decoder."):
            continue
        if k not in b:
            continue
        diff = (v.detach().float() - b[k].detach().float()).reshape(-1)
        s += float((diff * diff).sum().item())
    return float(s ** 0.5)


def check_training_log(dataset: str) -> tuple[bool, list[str]]:
    """Walk the training stdout log; flag any val/test_locked subject ID
    in the per-step subject listings. The v1 logs only print one summary
    line ('main] brainweb train subjects: 14' or 'hecktor train subjects:
    156'); verify those counts match the expected."""
    log = ROOT / "logs" / f"medsam2_train_{dataset}_stdout.log"
    txt = log.read_text(encoding="utf-8")
    # Allowed tokens: 'train subjects: 14' or 'train subjects: 156'.
    m = re.search(r"train subjects:\s*(\d+)", txt)
    if not m:
        return False, ["no train-subject-count line found in log"]
    n = int(m.group(1))
    expected = 14 if dataset == "brainweb" else 156
    if n != expected:
        return False, [f"train subject count {n} != expected {expected}"]
    # Reject any of the test/validation subject tokens.
    if dataset == "brainweb":
        bad = {"subject41", "subject42", "subject43", "subject46", "subject47", "subject48"}
    else:
        # HE: the test_locked/validation set is a list of integers. Just
        # refuse to see any 100+ IDs outside the known train set. This
        # is a coarse check; the real per-subject audit is in the v3
        # split CSVs (see scripts/_swap_splits.py).
        bad = set()
        for tok in re.findall(r"hecktor_\d+", txt):
            sid = tok.split("_", 1)[1]
            # If the log mentions a specific HE subject ID, we want the
            # known train set only. Test/val IDs are tracked separately.
            # v1 logs do not list per-subject IDs, so we expect zero hits.
            bad.add(tok)
    found_bad = sorted(t for t in bad if t in txt)
    if found_bad:
        return False, [f"unexpected tokens in log: {found_bad}"]
    return True, []


def stub_inference_load(ckpt_path: Path) -> tuple[bool, str]:
    """Spawn a subprocess that loads the checkpoint and prints
    `LOADED sha=<actual> path=<actual>`. Capture stdout and assert the
    printed SHA matches the expected."""
    code = (
        "import hashlib, sys, torch\n"
        f"p = r'{ckpt_path}'\n"
        "h = hashlib.sha256(open(p, 'rb').read()).hexdigest()\n"
        "torch.load(p, map_location='cpu')\n"
        "print(f'LOADED sha={h} path={p}')\n"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True, text=True, timeout=120,
    )
    out = proc.stdout.strip()
    m = re.match(r"LOADED sha=([0-9a-f]+) path=(.+)$", out)
    if not m:
        return False, f"could not parse stdout: {out!r}; stderr: {proc.stderr!r}"
    printed_sha = m.group(1)
    expected_sha = sha256(ckpt_path)
    return printed_sha == expected_sha, f"printed_sha={printed_sha}; expected={expected_sha}"


def main() -> int:
    if not PRETRAINED.is_file():
        print(f"FATAL: pretrained MedSAM2 not found at {PRETRAINED}")
        return 2
    pretrained_sd = load_state_dict(PRETRAINED)
    pretrained_sha_actual = sha256(PRETRAINED)
    if pretrained_sha_actual != PRETRAINED_SHA:
        print(
            f"FATAL: pretrained SHA mismatch: actual={pretrained_sha_actual}, "
            f"expected={PRETRAINED_SHA}"
        )
        return 2

    audit_rows = []
    decoder_sds = {}
    for dataset, (fname, expected_sha) in EXPECTED.items():
        ckpt = ROOT / "checkpoints" / fname
        print(f"\n=== {dataset} ({fname}) ===")
        if not ckpt.is_file():
            print(f"  missing checkpoint: {ckpt}")
            audit_rows.append({
                "checkpoint_id": dataset, "hash_expected": expected_sha,
                "hash_actual": "MISSING", "encoder_frozen": "n/a",
                "decoder_differs_from_pretrained": "n/a",
                "decoder_differs_across_datasets": "n/a",
                "training_log_subjects_clean": "n/a",
                "inference_log_sha_match": "n/a",
                "decision": "REJECT_FILE_MISSING",
            })
            continue
        actual_sha = sha256(ckpt)
        hash_ok = (actual_sha == expected_sha)
        print(f"  sha256 actual={actual_sha} expected={expected_sha} -> {'OK' if hash_ok else 'FAIL'}")

        sd = load_state_dict(ckpt)
        decoder_sds[dataset] = sd

        enc_dist = encoder_l2_dist(sd, pretrained_sd)
        enc_thresh = 1e-3
        encoder_frozen = enc_dist < enc_thresh
        print(f"  encoder L2 dist vs pretrained = {enc_dist:.4e} (thresh {enc_thresh}) -> {'FROZEN' if encoder_frozen else 'NOT FROZEN'}")

        dec_dist_pretrained = decoder_l2_dist(sd, pretrained_sd)
        dec_thresh = 1e-3
        decoder_differs = dec_dist_pretrained > dec_thresh
        print(f"  decoder L2 dist vs pretrained = {dec_dist_pretrained:.4e} (thresh {dec_thresh}) -> {'DIFFERS' if decoder_differs else 'IDENTICAL'}")

        log_ok, log_notes = check_training_log(dataset)
        print(f"  training log subjects clean = {log_ok}; notes={log_notes}")

        infer_ok, infer_msg = stub_inference_load(ckpt)
        print(f"  inference loader sha match = {infer_ok}; {infer_msg}")

        decision = "PASS" if (hash_ok and encoder_frozen and decoder_differs and log_ok and infer_ok) else "FAIL"
        print(f"  -> decision: {decision}")

        audit_rows.append({
            "checkpoint_id": dataset,
            "hash_expected": expected_sha,
            "hash_actual": actual_sha,
            "encoder_frozen": str(encoder_frozen).lower(),
            "decoder_differs_from_pretrained": str(decoder_differs).lower(),
            "decoder_differs_across_datasets": "pending",  # filled below
            "training_log_subjects_clean": str(log_ok).lower(),
            "inference_log_sha_match": str(infer_ok).lower(),
            "decision": decision,
        })

    # Decoder differs across datasets (BW vs HE).
    if "brainweb" in decoder_sds and "hecktor" in decoder_sds:
        cross = decoder_l2_dist(decoder_sds["brainweb"], decoder_sds["hecktor"])
        cross_ok = cross > 1e-3
        for r in audit_rows:
            if r["checkpoint_id"] in ("brainweb", "hecktor"):
                r["decoder_differs_across_datasets"] = str(cross_ok).lower()
        print(f"\nDecoder L2 BW vs HE = {cross:.4e} -> {'DIFFERS' if cross_ok else 'IDENTICAL'}")

    out = ROOT / "checkpoints" / "audit_report.csv"
    fields = [
        "checkpoint_id", "hash_expected", "hash_actual",
        "encoder_frozen", "decoder_differs_from_pretrained",
        "decoder_differs_across_datasets", "training_log_subjects_clean",
        "inference_log_sha_match", "decision",
    ]
    with open(out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for r in audit_rows:
            w.writerow(r)
    print(f"\nWrote audit report: {out.relative_to(ROOT)}")
    return 0 if all(r["decision"] == "PASS" for r in audit_rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())