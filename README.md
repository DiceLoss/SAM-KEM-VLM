# Stage 7 baseline code handoff v3 (handoff/baselines-source-v3)

Per directive 6159621060 (issue #2). Source-only delivery; **no
training authorization delegated to the experiment worker by this
directive**. Training is the researcher's responsibility on V100.

## Contents

- `upstream_official/`
  - `ADMMNet/` — yangyan92/Deep-ADMM-Net @ f95738c (MIT). Status
    `BLOCKED_WRONG_UPSTREAM`: MRI CS, not PET.
  - `CNNBPNet/` — F1HBME/IterativeCNN @ 39ea8e6 (BSD-3). Status
    `OFFICIAL_READY_CODE_ONLY_PENDING_ALIAS_CONFIRMATION`: researcher
    must confirm that "CNNBPNet" in the manuscript == this
    repository's "Iterative PET Image Reconstruction Using CNN
    Representation" (Gong et al. TMI 2018).
  - `NeuralKEM/` — SiqiLi1020/Neural-kernel-EM @ a527a65. Status
    `BLOCKED_LICENSE_REDISTRIBUTION_RESTRICTED`: UC-Davis-proprietary,
    not OSI-approved open source; needs researcher license proof.
- `code/`
  - `fbpnet/fbpnet_reimplementation.py` — Wang/Liu PMB 2020 dynamic
    reimplementation. Status `REIMPLEMENTATION_REVIEW_REQUIRED`:
    not approved for training (negative output, incomplete temporal
    coupling, BN3d only).
  - `directpet/directpet_reimplementation.py` — Hashimoto TRPMS 2022
    reimplementation. Status `REIMPLEMENTATION_REVIEW_REQUIRED`:
    generic-surrogate architecture, not paper-exact.
  - `fastpet/fastpet_reimplementation.py` — Whiteley 2020
    reimplementation. Status `REIMPLEMENTATION_REVIEW_REQUIRED`:
    generic-surrogate architecture, not paper-exact.
  - `code_handoff_registry.json` — machine-readable per-method
    registry with paper, upstream URL, commit, license, file list +
    SHA-256, tensor contract, repro command, smoke result.
- `build_registry_v3.py` — rebuild script for the registry.

## Reproducing the smoke tests

```bash
# FBPNet (reimplementation, dynamic-frame, zero-training)
cd code/fbpnet && python -c "import sys; sys.path.insert(0, '.'); \
  import torch; from fbpnet_reimplementation import FBPNetReimplementation; \
  torch.manual_seed(0); net = FBPNetReimplementation(180, 182, 128, n_frames=2); \
  sino = torch.randn(1, 2, 180, 182); print(net(sino).shape)"

# DirectPET (reimplementation, zero-training)
cd code/directpet && python -c "import sys; sys.path.insert(0, '.'); \
  import torch; from directpet_reimplementation import DirectPETGenerator; \
  torch.manual_seed(0); net = DirectPETGenerator(180, 182, 128); \
  sino = torch.randn(1, 180, 182); print(net(sino).shape)"

# FastPET (reimplementation, zero-training)
cd code/fastpet && python -c "import sys; sys.path.insert(0, '.'); \
  import torch; from fastpet_reimplementation import FastPETReimplementation; \
  torch.manual_seed(0); net = FastPETReimplementation(180, 182, 128); \
  sino = torch.randn(1, 180, 182); print(net(sino).shape)"
```

All three reimplementations are zero-training: weights are random
initialization per `torch.manual_seed(0)`. No paper metrics are
copied as our results.

## Provenance

All upstream repos were acquired on 2026-10-14 via the GitHub
tarball API (`https://api.github.com/repos/{owner}/{repo}/tarball`).
No `git clone` was used (network constraints). All upstream commits
were independently verified via the GitHub commits API.

## License and distribution notes

- ADMMNet: MIT (upstream). Permits research use.
- CNNBPNet: BSD-3-Clause (upstream). Permits research use.
- NeuralKEM: UC-Davis proprietary (NOT OSI-approved open source).
  Internal research use only. **Distribution beyond UC requires
  written permission.** Status remains BLOCKED_LICENSE_REDISTRIBUTION_RESTRICTED
  until the researcher provides proof that private transfer to the
  controlled research server and internal experimental use are
  explicitly permitted.
- Three reimplementations: no upstream license obligation (paper
  equations + standard PyTorch patterns only).

## What's NOT in this handoff

- Locked test data: per directive §C, none.
- Paper numbers: per directive §C, none.
- Training code that touches the main repo:
  `D:/MIAProject/SAM_KEM_QWEN3_V3_REPO/` is unmodified.
- Frozen arrays / historical results: per directive §C, none.
- Weights: this handoff ships architectures only. No checkpoints
  transferred.

## Next actions (researcher-controlled)

1. Review the registry's `reimplementation_audit_table` for each
   `REIMPLEMENTATION_REVIEW_REQUIRED` method. Sign off on the
   explicit ASSUMPTIONs A1-A6 (FBPNet), D1-D3 (DirectPET), F1-F3
   (FastPET) before training.
2. Provide a written license addendum for NeuralKEM (UC-Davis) to
   lift the `BLOCKED_LICENSE_REDISTRIBUTION_RESTRICTED` status.
3. Confirm or correct the "CNNBPNet" table alias for the F1HBME
   repository to lift the `PENDING_ALIAS_CONFIRMATION` status.
4. Confirm or correct that ADMMNet is the manuscript's PET
   comparison, or remove from main table entirely.
5. Once cleared, deploy the cleared code on V100 and run training.
   The worker will NOT execute this step.
