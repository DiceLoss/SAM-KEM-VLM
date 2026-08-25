"""Compute SHA-256 of all original algorithm files and write to
code/src/original/SOURCES.txt for verification.

Usage:
    python code/scripts/hash_originals.py
"""
from __future__ import annotations
import hashlib
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent
ORIGINAL_DIR = REPO / "code" / "original"

# (current path in repo, source path in the original codebase)
FILES = [
    ("EM.py", "D:/MIAProject/EM-reconstruction/EM.py"),
    ("recon.py", "D:/MIAProject/SAM_KEM_QWEN3_MEDSAM2_DELIVERY/scripts/recon.py"),
    ("discrete_radon.py", "D:/MIAProject/SAM_KEM_QWEN3_MEDSAM2_DELIVERY/scripts/discrete_radon.py"),
    ("poisson_simulate.py", "D:/MIAProject/SAM_KEM_QWEN3_MEDSAM2_DELIVERY/scripts/poisson_simulate.py"),
    ("metrics.py", "D:/MIAProject/SAM_KEM_QWEN3_MEDSAM2_DELIVERY/scripts/metrics.py"),
    ("build_medsam2.py", "D:/MIAProject/EM-reconstruction/build_medsam2.py"),
    ("medsam2_decoder_train.py", "D:/MIAProject/SAM_KEM_QWEN3_MEDSAM2_DELIVERY/scripts/medsam2_decoder_train.py"),
    ("medsam2_decoder_infer.py", "D:/MIAProject/SAM_KEM_QWEN3_MEDSAM2_DELIVERY/scripts/medsam2_decoder_infer.py"),
    ("medsam2_inference.py", "D:/MIAProject/SAM_KEM_QWEN3_MEDSAM2_DELIVERY/scripts/medsam2_inference.py"),
    ("preprocessing.py", "D:/MIAProject/SAM_KEM_QWEN3_MEDSAM2_DELIVERY/scripts/preprocessing.py"),
    ("torch_radon_impl.py", "D:/MIAProject/EM-reconstruction/torch_radon_impl.py"),
    ("MydataLoader.py", "D:/MIAProject/EM-reconstruction/legacy_inactive/MydataLoader.py"),
]


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    out_lines = ["SHA-256 checksums of original algorithm files (computed at copy time)",
                  "(format: <sha256>  <filename>   <original-path>)",
                  ""]
    for name, src in FILES:
        repo_p = REPO / "code/src/original" / name
        if not repo_p.is_file():
            out_lines.append(f"MISSING  {name}  {src}")
            continue
        h = sha256(repo_p)
        out_lines.append(f"{h}  {name}  {src}")
    out = REPO / "code/src/original/SOURCES.txt"
    out.write_text("\n".join(out_lines) + "\n", encoding="utf-8")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())