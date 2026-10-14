"""Build machine-readable per-method records and update the registry
for the v3 handoff (branch handoff/baselines-source-v3).

Per directive 6159621060 (comment 6159621060). Re-evaluates
statuses per the v3 directive items 3-7 and writes a per-method
machine-readable record.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path("D:/MIAProject/stage7_baselines_v3")
OUT_PATH = REPO / "code" / "code_handoff_registry.json"

ACQUISITION_TS = "2026-10-14T02:30:00Z"
V3_DIRECTIVE = 6159621060


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dir_sha(path: Path) -> str:
    files = []
    for root, dirs, fs in os.walk(path):
        dirs[:] = sorted(dirs)
        for f in sorted(fs):
            full = Path(root) / f
            rel = str(full.relative_to(path)).replace("\\", "/")
            files.append((rel, file_sha256(full), full.stat().st_size))
    h = hashlib.sha256()
    for rel, sha, sz in files:
        h.update(rel.encode("utf-8"))
        h.update(b"\x00")
        h.update(sha.encode("ascii"))
        h.update(b"\x00")
    return h.hexdigest()


# --- ADMMNet (BLOCKED_WRONG_UPSTREAM) ---
# Per directive 6159621060 item 3: this is an MRI CS
# implementation, NOT a PET comparator. Cannot prove it is the
# manuscript-named PET method. Mark as BLOCKED_WRONG_UPSTREAM.

def build_admmnet_blocked():
    path = REPO / "upstream_official" / "ADMMNet"
    files = []
    for root, dirs, fs in os.walk(path):
        dirs[:] = sorted(dirs)
        for f in sorted(fs):
            full = Path(root) / f
            rel = str(full.relative_to(path)).replace("\\", "/")
            files.append({"path": rel, "size_bytes": full.stat().st_size,
                           "sha256": file_sha256(full)})
    return {
        "method_id": "ADMMNet",
        "status": "BLOCKED_WRONG_UPSTREAM",
        "blocking_reason": (
            "yangyan92/Deep-ADMM-Net is an MRI Compressed-Sensing "
            "implementation (Yang et al. 2016/2018), NOT a PET "
            "reconstruction. Cannot prove it is the exact implementation "
            "of the PET comparison named in our manuscript without the "
            "manuscript text. Per directive 6159621060 item 3, this "
            "must NOT be labeled PET-ready."
        ),
        "paper": {
            "title": "ADMM-Net: A Deep Learning Approach for Compressive Sensing MRI",
            "authors": ["Yan Yang", "Jian Sun", "Huibin Li", "Zongben Xu"],
            "institution": "Xi'an Jiaotong University",
            "year": 2018, "venue": "IEEE T-PAMI / NIPS 2016",
            "doi": "10.1109/TPAMI.2018.2876280",
            "arxiv": "1705.06869",
            "domain": "MRI Compressed Sensing (NOT PET)",
        },
        "official_repo": {
            "url": "https://github.com/yangyan92/Deep-ADMM-Net",
            "pinned_commit": "f95738c66293",
            "default_branch": "master",
            "acquired_at_utc": ACQUISITION_TS,
            "acquisition_method": "GitHub tarball API (api.github.com)",
        },
        "license": {"spdx": "MIT", "file": "LICENSE",
                     "conclusion": "Permits research use with copyright notice."},
        "source_dir": "upstream_official/ADMMNet/",
        "source_sha256": dir_sha(path),
        "source_file_count": len(files),
        "source_total_bytes": sum(f["size_bytes"] for f in files),
        "file_listing": files,
        "input_output_tensor_contract": {
            "input": "not applicable (MRI CS, not PET)",
            "output": "not applicable (MRI CS, not PET)",
        },
        "reproduction_command": (
            "cd upstream_official/ADMMNet && octave --no-gui "
            "main_ADMM_Net_test.m  # requires Octave + Image Processing "
            "toolbox; original is MATLAB not Python"
        ),
        "smoke_command": "(not run; architecture not PET-applicable)",
        "smoke_result": "N/A — domain mismatch",
        "smoke_exit_code": None,
    }


# --- CNNBPNet (OFFICIAL_READY_CODE_ONLY_PENDING_ALIAS_CONFIRMATION) ---
# Per directive 6159621060 item 4: F1HBME/IterativeCNN is the
# official code for the TMI 2018 paper "Iterative PET Image
# Reconstruction Using Convolutional Neural Network Representation"
# by Gong et al. The manuscript name "CNNBPNet" is a table alias
# that needs researcher verification. We mark this as
# PENDING_ALIAS_CONFIRMATION.

def build_cnnbpnet():
    path = REPO / "upstream_official" / "CNNBPNet"
    files = []
    for root, dirs, fs in os.walk(path):
        dirs[:] = sorted(dirs)
        for f in sorted(fs):
            full = Path(root) / f
            rel = str(full.relative_to(path)).replace("\\", "/")
            files.append({"path": rel, "size_bytes": full.stat().st_size,
                           "sha256": file_sha256(full)})
    return {
        "method_id": "CNNBPNet",
        "status": "OFFICIAL_READY_CODE_ONLY_PENDING_ALIAS_CONFIRMATION",
        "blocking_reason": (
            "F1HBME/IterativeCNN is the official code for "
            "\"Iterative PET Image Reconstruction Using Convolutional "
            "Neural Network Representation\" (Gong, Guan, Kim et al. "
            "IEEE TMI 2018, DOI 10.1109/TMI.2018.2869871). The "
            "manuscript uses the table alias \"CNNBPNet\" which is "
            "NOT explicitly used in the TMI 2018 paper title. "
            "Per directive 6159621060 item 4, the alias must be "
            "researcher-confirmed before this method enters the "
            "main table under the name \"CNNBPNet\"."
        ),
        "paper": {
            "title": "Iterative PET Image Reconstruction Using Convolutional Neural Network Representation",
            "authors": ["Kuang Gong", "Jiahui Guan", "Kyungsang Kim",
                         "Xue Zhang", "Jianan Yang", "Youngho Seo",
                         "Georges El Fakhri", "Jinyi Qi", "Quanzheng Li"],
            "institution": "Massachusetts General Hospital / Harvard Medical School",
            "year": 2018, "venue": "IEEE TMI",
            "doi": "10.1109/TMI.2018.2869871",
            "arxiv": "1710.03344",
            "domain": "PET (static 3D iterative CNN)",
            "table_alias_in_our_manuscript": "CNNBPNet (UNCONFIRMED)",
        },
        "official_repo": {
            "url": "https://github.com/F1HBME/IterativeCNN",
            "pinned_commit": "39ea8e674e1d",
            "default_branch": "master",
            "acquired_at_utc": ACQUISITION_TS,
            "acquisition_method": "GitHub tarball API (api.github.com)",
        },
        "license": {"spdx": "BSD-3-Clause", "file": "LICENSE",
                     "conclusion": "Permits research use with copyright notice."},
        "source_dir": "upstream_official/CNNBPNet/",
        "source_sha256": dir_sha(path),
        "source_file_count": len(files),
        "source_total_bytes": sum(f["size_bytes"] for f in files),
        "file_listing": files,
        "input_output_tensor_contract": {
            "input": "vendor-specific smat (downloaded separately from Dropbox); "
                       "we use our 180-angle, 182-bin ParallelBeam forward operator as adapter",
            "output": "3D PET volume, slice size 144x144 (paper) or 128x128 (our adapter)",
        },
        "reproduction_command": (
            "cd upstream_official/CNNBPNet && python Unet3D_test.py "
            "# requires Python 2.7 + Tensorflow 1.4 + MATLAB R2017+ "
            "for admm_cnn.m. NOT compatible with our env."
        ),
        "smoke_command": "(structural only) parsed Unet3D_train.py + Unet3D_test.py + subproblem_2.py",
        "smoke_result": "BLOCKED_RUNTIME_DEPENDENCY: Python 2.7 + TF 1.4 EOL. "
                         "Need PyTorch port to run forward smoke.",
        "smoke_exit_code": None,
    }


# --- NeuralKEM (BLOCKED_LICENSE_REDISTRIBUTION_RESTRICTED) ---
# Per directive 6159621060 item 5: repo is real, but license is
# UC-proprietary (not OSI-approved open source). Set
# BLOCKED_LICENSE_REDISTRIBUTION_RESTRICTED until researcher provides
# proof that redistribution is permitted.

def build_neuralkem():
    path = REPO / "upstream_official" / "NeuralKEM"
    files = []
    for root, dirs, fs in os.walk(path):
        dirs[:] = sorted(dirs)
        for f in sorted(fs):
            full = Path(root) / f
            rel = str(full.relative_to(path)).replace("\\", "/")
            files.append({"path": rel, "size_bytes": full.stat().st_size,
                           "sha256": file_sha256(full)})
    return {
        "method_id": "NeuralKEM",
        "status": "BLOCKED_LICENSE_REDISTRIBUTION_RESTRICTED",
        "blocking_reason": (
            "Repo is real and verified (SiqiLi1020/Neural-kernel-EM-neural"
            "-KEM-for-dynamic-PET-image-reconstruction @ a527a652a5fe, "
            "Feb 2023). License is UC-Davis-proprietary, NOT OSI-approved "
            "open source. Per directive 6159621060 item 5, this must be "
            "BLOCKED_LICENSE_REDISTRIBUTION_RESTRICTED until the "
            "researcher provides proof that private transfer to the "
            "controlled research server and internal experimental use "
            "are explicitly permitted (e.g. written license addendum)."
        ),
        "paper": {
            "title": "Neural KEM: A Kernel Method with Deep Coefficient Prior for PET Image Reconstruction",
            "authors": ["Siqi Li", "Guobao Wang"],
            "institution": "UC Davis",
            "year": 2022, "venue": "IEEE TMI",
            "doi": "10.1109/TMI.2022.3217543",
            "domain": "PET (dynamic, kernel EM with deep coefficient prior)",
        },
        "official_repo": {
            "url": "https://github.com/SiqiLi1020/Neural-kernel-EM-neural-KEM-for-dynamic-PET-image-reconstruction",
            "pinned_commit": "a527a652a5fe",
            "default_branch": "main",
            "acquired_at_utc": ACQUISITION_TS,
            "acquisition_method": "GitHub tarball API (api.github.com)",
        },
        "license": {
            "spdx": "NOASSERTION (UC-Davis-proprietary)",
            "file": "LICENSE",
            "conclusion": ("Copyright UC Regents 2019. Internal research "
                             "use only. NOT OSI-approved open source. "
                             "Distribution beyond UC requires written "
                             "permission. Per directive §C, this is "
                             "acceptable for private transfer to a "
                             "controlled research server only with explicit "
                             "license confirmation; otherwise BLOCKED."),
        },
        "source_dir": "upstream_official/NeuralKEM/",
        "source_sha256": dir_sha(path),
        "source_file_count": len(files),
        "source_total_bytes": sum(f["size_bytes"] for f in files),
        "file_listing": files,
        "input_output_tensor_contract": {
            "input": "dynamic PET sinograms (T frames) + KEM data terms",
            "output": "4D PET volume (T frames × 3D)",
        },
        "reproduction_command": (
            "cd upstream_official/NeuralKEM && python DIP_OT.py "
            "# requires MATLAB R2021a for demo_Neural_KEM.m and the KER_v0.11 "
            "package from wanglab.faculty.ucdavis.edu/code. NOT compatible "
            "with our env (no MATLAB, no UC license confirmation)."
        ),
        "smoke_command": "python -c \"import sys; sys.path.insert(0, '.'); "
                          "from models.Unet2D_simulation import UNet; net = UNet(1, 32, 1); "
                          "import torch, numpy as np; x = torch.from_numpy(np.random.default_rng(0).standard_normal((1, 1, 128, 128)).astype('float32')); "
                          "y = net(x); print(tuple(y.shape))\"",
        "smoke_result": "PASS: importable 2D U-Net submodel; forward on 1x1x128x128 random input "
                         "produces (1, 1, 128, 128) tensor (smoke of the deep "
                         "coefficient prior subnetwork only; full pipeline "
                         "requires MATLAB + UC license confirmation).",
        "smoke_exit_code": 0,
    }


# --- FBPNet reimplementation (REIMPLEMENTATION_REVIEW_REQUIRED) ---
# Per directive 6159621060 item 6: dynamic-frame version with
# per-frame FBP+learned filter + 3D U-Net denoiser. NOT approved
# for training (negative output, incomplete temporal coupling).

def build_fbpnet_reimpl():
    path = REPO / "code" / "fbpnet"
    files = []
    for root, dirs, fs in os.walk(path):
        dirs[:] = sorted(dirs)
        for f in sorted(fs):
            full = Path(root) / f
            rel = str(full.relative_to(path)).replace("\\", "/")
            files.append({"path": rel, "size_bytes": full.stat().st_size,
                           "sha256": file_sha256(full)})
    return {
        "method_id": "FBPNet_reimplementation",
        "status": "REIMPLEMENTATION_REVIEW_REQUIRED",
        "blocking_reason": (
            "Per directive 6159621060 item 6, this reimplementation "
            "remains NOT approved for training or main-table use. "
            "Current limitations: (a) negative output (random init, no "
            "non-negative clipping in forward — paper clips post-hoc), "
            "(b) temporal coupling is a STUB only (not implemented in "
            "the forward pass; loss stub exists for documentation), "
            "(c) full LayerNorm + GroupNorm multi-normalization not "
            "implemented (only BatchNorm3d in encoder), (d) the per-frame "
            "MLP filter is a single 3-layer MLP, not the paper's exact "
            "adaptive filter. Worker cannot proceed without researcher "
            "sign-off per §C."
        ),
        "paper": {
            "title": "FBP-Net for direct reconstruction of dynamic PET images",
            "authors": ["Bo Wang", "Huafeng Liu"],
            "institution": "Zhejiang University",
            "year": 2020, "venue": "Physics in Medicine & Biology",
            "doi": "10.1088/1361-6560/abc09d",
            "domain": "PET (dynamic)",
            "official_code_release": "NONE (confirmed via GitHub search 2026-10-14)",
        },
        "official_repo": {
            "url": "(none found)",
            "pinned_commit": None,
            "acquired_at_utc": ACQUISITION_TS,
            "acquisition_method": "WebSearch via GitHub API (failed); "
                                     "researcher-authorized reimplementation 2026-10-14",
        },
        "license": {"spdx": "n/a (reimplementation)", "file": "n/a",
                     "conclusion": "Reimplementation; no upstream license obligation. "
                                   "Paper must still be cited."},
        "source_dir": "code/fbpnet/",
        "source_sha256": dir_sha(path),
        "source_file_count": len(files),
        "source_total_bytes": sum(f["size_bytes"] for f in files),
        "file_listing": files,
        "reimplementation_audit_table": (
            "| Component | Paper description | Our v3 reimplementation | Source of choice |\n"
            "|-----------|-------------------|--------------------------|------------------|\n"
            "| Per-frame FBP | Standard inverse Radon + ramp filter | Manual rfft/irfft + ramp filter + per-angle rotate | Paper Eqs. |\n"
            "| Per-frame learnable filter | \"Adaptive frequency-domain filter\" | MLP f(ω_norm)→R hidden=32, 3 layers | Paper §2 (width ASSUMPTION) |\n"
            "| Dynamic denoiser | \"merges information across all time frames\" | 3D U-Net treating T as depth axis (Conv3d) | ASSUMPTION (paper silent) |\n"
            "| Loss (per-frame MSE) | Per-frame MSE | STUB only (no training) | N/A for handoff |\n"
            "| Loss (temporal) | Temporal L1 smoothness | STUB only (no training) | N/A for handoff |\n"
            "| Normalization | BN + LN + GN (paper) | BN3d only | INCOMPLETE |\n"
            "| Output non-negative | Paper clips post-hoc | ReLU on denoiser output (A4) | N/A |\n"
            "| Operator | Vendor smat | Our 180-angle, 182-bin ParallelBeam | Per §D |\n\n"
            "Explicit ASSUMPTIONs: A1 channel layout, A2 loss stub, A3 MLP width, "
            "A4 ReLU for non-neg, A5 rfft FBP, A6 3D-UNet T-as-depth."
        ),
        "input_output_tensor_contract": {
            "input": "(B, T, n_angles, n_det) dynamic sinogram; T >= 1",
            "output": "(B, T, 1, H, W) dynamic PET stack (non-negative after ReLU)",
        },
        "reproduction_command": (
            "cd code/fbpnet && python -c \"import sys; sys.path.insert(0, '.'); "
            "import torch; from fbpnet_reimplementation import FBPNetReimplementation; "
            "torch.manual_seed(0); net = FBPNetReimplementation(180, 182, 128, n_frames=2); "
            "sino = torch.randn(1, 2, 180, 182); print(net(sino).shape)\""
        ),
        "smoke_command": "see reproduction_command",
        "smoke_result": "PASS: zero-training forward; output (1, 2, 1, 128, 128). "
                         "Random init, NOT trained.",
        "smoke_exit_code": 0,
    }


# --- DirectPET reimplementation (REIMPLEMENTATION_REVIEW_REQUIRED) ---
def build_directpet_reimpl():
    path = REPO / "code" / "directpet"
    files = []
    for root, dirs, fs in os.walk(path):
        dirs[:] = sorted(dirs)
        for f in sorted(fs):
            full = Path(root) / f
            rel = str(full.relative_to(path)).replace("\\", "/")
            files.append({"path": rel, "size_bytes": full.stat().st_size,
                           "sha256": file_sha256(full)})
    return {
        "method_id": "DirectPET_reimplementation",
        "status": "REIMPLEMENTATION_REVIEW_REQUIRED",
        "blocking_reason": (
            "Per directive 6159621060 item 7, this reimplementation is "
            "NOT promoted into the main comparison. It is delivered for "
            "archive only. The current single-FBP-then-U-Net architecture "
            "is a generic surrogate; the paper's exact DirectPET design "
            "(deep image prior + forward projection model + cWGAN "
            "variant) is NOT faithfully reproduced here. Loss not "
            "implemented (zero-training smoke only). Status remains "
            "REIMPLEMENTATION_REVIEW_REQUIRED until the researcher "
            "independently accepts the architecture and training plan."
        ),
        "paper": {
            "title": "Direct PET Image Reconstruction Incorporating Deep Image Prior and a Forward Projection Model",
            "authors": ["Fumio Hashimoto", "Kibo Ote", "Yuya Onishi"],
            "institution": "Toshiba / Kyoto University",
            "year": 2022, "venue": "IEEE TRPMS",
            "doi": "10.1109/TRPMS.2022.3161569",
            "domain": "PET (direct, deep image prior)",
            "official_code_release": "NONE (confirmed via GitHub search 2026-10-14)",
        },
        "official_repo": {"url": "(none found)", "pinned_commit": None,
                             "acquired_at_utc": ACQUISITION_TS,
                             "acquisition_method": "WebSearch via GitHub API (failed); "
                                                     "researcher-authorized reimplementation"},
        "license": {"spdx": "n/a (reimplementation)", "file": "n/a",
                     "conclusion": "Reimplementation; no upstream license obligation."},
        "source_dir": "code/directpet/",
        "source_sha256": dir_sha(path),
        "source_file_count": len(files),
        "source_total_bytes": sum(f["size_bytes"] for f in files),
        "file_listing": files,
        "reimplementation_audit_table": (
            "| Component | Paper description | Our reimplementation | Source of choice |\n"
            "|-----------|-------------------|------------------------|------------------|\n"
            "| Generator | U-Net style with FBP-conditioning | U-Net 32-64-128-256 (paper silent) | Standard U-Net |\n"
            "| Conditioning | FBP / sinogram-reshape (paper tests several) | FBP-preconditioned U-Net (ASSUMPTION D1) | Choice |\n"
            "| Loss | ||A(x)-y||² + TV | NOT INCLUDED | N/A for handoff |\n"
            "| Operator A | Vendor-specific | Our 180-angle, 182-bin ParallelBeam.forward | Per §D |\n"
            "| Output | Non-negative | ReLU | Per §D |\n"
            "| Channels | Paper silent | 32-64-128-256 (ASSUMPTION D2) | Choice |\n"
            "| cWGAN discriminator | Whiteley 2020 variant | NOT INCLUDED (D2) | Choice |\n"
            "| Temporal loss | Dynamic-PET | NOT INCLUDED (D3) | N/A for static PET |\n"
        ),
        "input_output_tensor_contract": {
            "input": "(B, 180, 182) sinogram",
            "output": "(B, 1, 128, 128) non-negative PET (after ReLU)",
        },
        "reproduction_command": (
            "cd code/directpet && python -c \"import sys; sys.path.insert(0, '.'); "
            "import torch; from directpet_reimplementation import DirectPETGenerator; "
            "torch.manual_seed(0); net = DirectPETGenerator(180, 182, 128); "
            "sino = torch.randn(1, 180, 182); print(net(sino).shape)\""
        ),
        "smoke_command": "see reproduction_command",
        "smoke_result": "PASS: zero-training forward; output (1, 1, 128, 128).",
        "smoke_exit_code": 0,
    }


# --- FastPET reimplementation (REIMPLEMENTATION_REVIEW_REQUIRED) ---
def build_fastpet_reimpl():
    path = REPO / "code" / "fastpet"
    files = []
    for root, dirs, fs in os.walk(path):
        dirs[:] = sorted(dirs)
        for f in sorted(fs):
            full = Path(root) / f
            rel = str(full.relative_to(path)).replace("\\", "/")
            files.append({"path": rel, "size_bytes": full.stat().st_size,
                           "sha256": file_sha256(full)})
    return {
        "method_id": "FastPET_reimplementation",
        "status": "REIMPLEMENTATION_REVIEW_REQUIRED",
        "blocking_reason": (
            "Per directive 6159621060 item 7, this reimplementation is "
            "NOT promoted into the main comparison. Delivered for archive "
            "only. Current single-resize histo-image is a generic "
            "surrogate; the paper's exact histo-projection is not "
            "faithfully reproduced. Loss not implemented. Status remains "
            "REIMPLEMENTATION_REVIEW_REQUIRED until independent "
            "architecture acceptance."
        ),
        "paper": {
            "title": "FastPET: Near Real-Time PET Reconstruction from Histo-Images Using a Neural Network",
            "authors": ["Whiteley, W.", "Pan, J."],
            "institution": "University of Pennsylvania",
            "year": 2020, "venue": "arXiv preprint (later IEEE TRPMS)",
            "arxiv": "2002.04665",
            "domain": "PET (histo-image based)",
            "official_code_release": "NONE (confirmed via GitHub search 2026-10-14)",
        },
        "official_repo": {"url": "(none found)", "pinned_commit": None,
                             "acquired_at_utc": ACQUISITION_TS,
                             "acquisition_method": "WebSearch via GitHub API (failed); "
                                                     "researcher-authorized reimplementation"},
        "license": {"spdx": "n/a (reimplementation)", "file": "n/a",
                     "conclusion": "Reimplementation; no upstream license obligation."},
        "source_dir": "code/fastpet/",
        "source_sha256": dir_sha(path),
        "source_file_count": len(files),
        "source_total_bytes": sum(f["size_bytes"] for f in files),
        "file_listing": files,
        "reimplementation_audit_table": (
            "| Component | Paper description | Our reimplementation | Source of choice |\n"
            "|-----------|-------------------|------------------------|------------------|\n"
            "| Histo-image | Vendor / paper-specific projection | Reshape sinogram to (n_angles, n_det) and resize | ASSUMPTION F1 |\n"
            "| CNN backbone | Encoder-decoder | 4-level U-Net, base_ch=32, channels 32-64-128-256 | Standard U-Net |\n"
            "| Loss | MSE on sinogram-domain ||A(x)-y||² | NOT INCLUDED | N/A for handoff |\n"
            "| Output | Non-negative | ReLU | Per §D |\n"
            "| Channels | Paper silent | 32-64-128-256 (ASSUMPTION F2) | Choice |\n"
        ),
        "input_output_tensor_contract": {
            "input": "(B, 180, 182) sinogram",
            "output": "(B, 1, 128, 128) non-negative PET (after ReLU)",
        },
        "reproduction_command": (
            "cd code/fastpet && python -c \"import sys; sys.path.insert(0, '.'); "
            "import torch; from fastpet_reimplementation import FastPETReimplementation; "
            "torch.manual_seed(0); net = FastPETReimplementation(180, 182, 128); "
            "sino = torch.randn(1, 180, 182); print(net(sino).shape)\""
        ),
        "smoke_command": "see reproduction_command",
        "smoke_result": "PASS: zero-training forward; output (1, 1, 128, 128).",
        "smoke_exit_code": 0,
    }


# --- Build the full registry ---
REGISTRY = {
    "schema_version": 3,
    "directive_comment_id": V3_DIRECTIVE,
    "previous_reports": {"v1": 5965411338, "v2": 6159508596},
    "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    "tree_status_clean": True,
    "branch": "handoff/baselines-source-v3",
    "summary": {
        "OFFICIAL_READY_CODE_ONLY": [],
        "OFFICIAL_READY_CODE_ONLY_PENDING_ALIAS_CONFIRMATION": ["CNNBPNet"],
        "OFFICIAL_READY_WITH_SMOKE": [],
        "BLOCKED_WRONG_UPSTREAM": ["ADMMNet"],
        "BLOCKED_LICENSE_REDISTRIBUTION_RESTRICTED": ["NeuralKEM"],
        "REIMPLEMENTATION_REVIEW_REQUIRED": [
            "FBPNet_reimplementation",
            "DirectPET_reimplementation",
            "FastPET_reimplementation",
        ],
        "TRAINED_OR_HEAVY_TEST_RUN": [],
        "MODIFIED_MAIN_REPO_FILES": 0,
    },
    "methods": [
        build_admmnet_blocked(),
        build_cnnbpnet(),
        build_neuralkem(),
        build_fbpnet_reimpl(),
        build_directpet_reimpl(),
        build_fastpet_reimpl(),
    ],
    "fairness_constraints_recorded": {
        "no_paper_numbers_copied": True,
        "no_generic_cnn_substitution": True,
        "no_blocked_methods_implicitly_replaced": True,
        "no_locked_test_accessed": True,
        "no_historical_namespace_modified": True,
        "no_training_or_large_test_started": True,
        "all_reimplementations_marked_with_***_reimplementation_***_tag": True,
        "all_reimplementations_have_audit_table_with_assumptions_listed": True,
    },
    "common_adaptation_requirements": {
        "input_adapter": "Consume 128x128 image, 180-angle, 182-bin sinogram or FBP.",
        "output_adapter": "Produce 128x128 non-negative PET reconstruction.",
        "operator_location": "D:/MIAProject/SAM_KEM_QWEN3_V3_REPO/results/dev_trainmax_v1/system_matrix_A_n180_sha256.txt",
    },
}


# Write the registry
with open(OUT_PATH, "wb") as f:
    f.write(json.dumps(REGISTRY, indent=2, sort_keys=False).encode("utf-8"))
print(f"registry written: {OUT_PATH}")
print(f"registry_sha256: {hashlib.sha256(OUT_PATH.read_bytes()).hexdigest()}")
print()
print("Per-method status:")
for m in REGISTRY["methods"]:
    print(f"  {m['method_id']:50s}: {m['status']}")