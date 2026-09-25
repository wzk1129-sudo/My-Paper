# ProSAC APGC ControlNet probe

This branch starts from Anatomy-ControlNet V1 commit 092433bdf4bb13fc99968a20e53c958d120fac4e. It compares frozen MedDINOv3 anatomy (C3) with a ProSAC-style Sobel/Laplacian fusion (C4) on the fixed Mayo Fast Development Protocol. This is not a full ProSAC-CT reproduction or a formal paper result.

The code is in ../../prosac_apgc_control_v1. The experiment config, provenance, six-node Fast64 EMA histories, per-slice deltas, access audits, and report are in this directory. Run-time checkpoints, logs, and medical images are intentionally excluded from Git.

The official MedDINOv3 source is an external dependency, pinned to https://github.com/ricklisz/MedDINOv3 commit a2460c2547899283446e520b84523454f7e25608. On the remote experiment worktree it is cloned at official_source/MedDINOv3. The author ViT-B/16 CT-3M checkpoint is not distributed here; the expected SHA256 is 492b2b96c00e9d3b93ba3b07a8d387aa2196b5e04698378fcc23049b447966d2. The runtime loader verifies this hash before use.

C3 and C4 each trained from the same frozen Direct 12k EMA initialization on L067/L143. C1/C2 controls were reused only after Fast protocol, L333-fast64 split, and evaluator source hashes matched. The 2k gate promoted both arms to 6k. L506 and external-memory opens were zero. See PROSAC_APGC_REPORT.md for measured outcomes and PROSAC_PAPER_RECONSTRUCTION.md for the paper-to-code boundary.

