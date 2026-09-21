# Exact executable snapshot

The formal entry is `python -m mayo2016_smallset_v1.train`, run from this directory.
`train_rddm_rleb.py` is the earlier minimal synthetic/NPY trainer; it was **not**
used for the Mayo 60k run. `smoke_rddm_rleb.py` tests the standalone architecture.
No original full-FoundDiff modules or pretrained weights are required.

Model: 13,130,945 parameters, 10 RLEBs, four U-Net stages, four encoder skips plus
one stem skip, three spatial downsamplings and three upsamplings. The single
center LDCT is concatenated with the diffusion state. The residual-only objective
uses the original RDDM diffusion formulation and restored time-only conditioning.

## Recorded execution (not executed by this publication)

The original Linux/WSL environment is listed in the experiment's `environment.json`.
No dependency installation is performed by these instructions. The archived
configuration intentionally retains the original absolute dataset path to preserve
its hash and run identity. Provide the authorized private data at that path on the
original host. Do not enumerate or load the held-out test patient.

From this directory, in the existing `corediff` environment, the formal invocation is:

```bash
env CUDA_VISIBLE_DEVICES=0 CUBLAS_WORKSPACE_CONFIG=:4096:8 \
  OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1 \
  python -m mayo2016_smallset_v1.train \
  --config mayo2016_smallset_v1/CONFIG.json \
  --output /absolute/path/to/NEW_RUN_DIR --device cuda:0 \
  --gate mayo2016_smallset_v1/PREFLIGHT_PASS.json
```

Long training must run in a separately named tmux session. This experiment already
completed 60k; these commands are documentation, not a request to run again.
For an interrupted run below step 60000, add `--resume /absolute/checkpoint.pt`
and use a **new** output directory. Exact source/config hashes and saved RNG/data
stream state must match. Source data paths and previously selected checkpoint
locations must remain available; relocation is not automatically supported.
Resume from the completed 60k checkpoint is deliberately rejected.

The trainer stores raw and EMA states. Published metrics use `ema_state_dict`,
not `model_state_dict`. Weights are not included in this repository.

## Acceptance evidence and portability limits

`PREFLIGHT_PASS.json` records the original successful 20-step full-resolution
B=2 smoke, exact step-10-to-20 resume comparison, repeated deterministic validation
and 200-step benchmark. Its source and formal-config hashes are enforced by the
trainer. This is evidence for the recorded environment, not certification of a
different machine. A fresh data/environment adaptation needs its own preflight.

`acceptance.py` also checks metric AST identity against original RDDM-CT files at
the recorded paths. Those external project files and private acceptance
checkpoints are intentionally not bundled. Historical reports may link to such
remote-only artifacts. The exported model/trainer are byte-identical originals;
no portability patch or new architecture has been introduced.
