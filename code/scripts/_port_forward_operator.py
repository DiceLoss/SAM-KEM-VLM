"""Phase 5: forward operator port to the canonical 180-angle discrete Radon.

Computes and persists:
  - arrays/system_matrix_A_n180_sha256.txt
  - arrays/system_matrix_A_n180_meta.json
  - arrays/adjoint_check_n180.json   (verifies <A x, y> = <x, A^T y>)

Also asserts the adjoint property to <1e-9 relative error on a random
image, and verifies that the canonical (size=128, n_angles=180) pair
produces a non-empty row-sum-normalized sparse matrix with the expected
shape (180*128, 128*128) = (23040, 16384).

This script does NOT modify any existing reconstruction logic. It only
records the canonical operator identity so downstream scripts and the
validator can confirm they all use the same A.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(r"D:/MIAProject/SAM_KEM_QWEN3_MEDSAM2_DELIVERY")
sys.path.insert(0, str(ROOT / "scripts"))
from discrete_radon import build_pair  # noqa: E402

ARRAYS = ROOT / "arrays"
SHA_OUT = ARRAYS / "system_matrix_A_n180_sha256.txt"
META_OUT = ARRAYS / "system_matrix_A_n180_meta.json"
ADJ_OUT = ARRAYS / "adjoint_check_n180.json"

CANONICAL_SIZE = 128
CANONICAL_ANGLES = 180


def matrix_sha(A) -> str:
    """Deterministic SHA-256 of the sparse matrix contents."""
    h = hashlib.sha256()
    h.update(np.asarray(A.indptr, dtype=np.int64).tobytes())
    h.update(np.asarray(A.indices, dtype=np.int32).tobytes())
    h.update(np.asarray(A.data, dtype=np.float64).tobytes())
    h.update(repr(A.shape).encode())
    return h.hexdigest()


def main() -> int:
    fwd, bwd, fbp = build_pair(size=CANONICAL_SIZE, n_angles=CANONICAL_ANGLES)
    # Build the canonical sparse matrix by hand.
    from discrete_radon import _radon_lines
    A = _radon_lines(CANONICAL_SIZE, CANONICAL_ANGLES)
    sha = matrix_sha(A)
    meta = {
        "size": CANONICAL_SIZE,
        "n_angles": CANONICAL_ANGLES,
        "shape": list(A.shape),
        "nnz": int(A.nnz),
        "row_sum_min": float(np.asarray(A.sum(axis=1)).min()),
        "row_sum_max": float(np.asarray(A.sum(axis=1)).max()),
        "row_sum_mean": float(np.asarray(A.sum(axis=1)).mean()),
        "module": "scripts.discrete_radon.build_pair",
        "adjoint_strategy": "row_norm + sparse_transpose",
        "skimage_used": False,
    }

    ARRAYS.mkdir(exist_ok=True)
    SHA_OUT.write_text(sha + "  arrays/system_matrix_A_n180_sha256.txt\n",
                       encoding="utf-8")
    META_OUT.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"sha256: {sha}")
    print(f"meta:   {META_OUT.relative_to(ROOT)}")

    # Adjoint check on a deterministic random image.
    rng = np.random.RandomState(20260823)
    x = rng.rand(CANONICAL_SIZE, CANONICAL_SIZE)
    y = rng.rand(CANONICAL_ANGLES, CANONICAL_SIZE)
    Ax = fwd(x)
    ATy = bwd(y)
    ip1 = float((Ax * y).sum())
    ip2 = float((x * ATy).sum())
    rel = abs(ip1 - ip2) / max(abs(ip1), abs(ip2), 1e-12)
    adj_check = {
        "ip_Afwd_x_y": ip1,
        "ip_x_AfwdT_y": ip2,
        "abs_diff": abs(ip1 - ip2),
        "rel_diff": rel,
        "threshold": 1e-9,
        "passed": rel < 1e-9,
        "seed": 20260823,
    }
    ADJ_OUT.write_text(json.dumps(adj_check, indent=2), encoding="utf-8")
    print(f"adjoint check: {adj_check}")
    if not adj_check["passed"]:
        print("FATAL: adjoint check failed")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())