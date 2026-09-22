# Frozen Fast Development Protocol

This protocol is for **method screening only**, never final paper performance.
It preserves the FoundDiff-official-code reconstruction of the clean RDDM+RLEB
backbone; it is not an official FoundDiff Table-V reproduction.

## Immutable split and supervision rules

1. Optimization/query training patients: **L067 and L143**, exactly **1145 paired
   slices** (560 + 585). No other patient may enter optimization batches.
2. External-memory patients only: **L096, L109, L192, L286, L291, L310** (3655 pairs).
   They must never enter fast-dev optimization batches.
3. Validation: **L333-fast64**, using the existing frozen 64 **0-based** indices in
   `../../splits/L333_fast64.txt`. Never resample for a method, model or seed.
   Split SHA-256: `0d3170fd2462c7b19d1cc0f187b12e03bac2c9b15babca7c1d881568adde6895`.
4. **L506 must never be accessed during development**, including enumeration,
   loading, visualization, evaluation or checkpoint selection. Its 526 slices are
   a declaration from the existing protocol, not independently checked here.
5. For Retrieval, LDCT structure keys / NDCT detail values may come only from the
   six external-memory patients. **L333 and L506 never enter external memory.**
   External-memory NDCT is never supervision GT for the current query. The query
   GT is its own paired NDCT from L067/L143. All memory construction and retrieval
   provenance must retain `patient_id`; pair records expose this field and label
   donors `external_memory_only` / `eligible_as_query_supervision=False`.
   This archive implements the split/interface only, not a retrieval model.

## Unique comparison baseline

Before screening any method, run the **unmodified original clean RDDM+RLEB fast
baseline** once: L067+L143, at most **8000 optimizer steps**, validation at
**2000/4000/6000/8000**, fixed L333-fast64. The explicitly authorized reference run
is `rddm_fast_baseline_8k_20260922`; its completed, verified results are recorded in
`baseline_8k/`. No smoke or formal checkpoint initializes this run.

This single run is the sole fast-dev comparison reference. Compare matched steps:

`Delta_PSNR = method_fast_PSNR - baseline_fast_PSNR`.

**Do not claim a method gain by comparing against the formal 8-patient/60k
40.65805 dB score.** Different data/training/validation budgets are not interchangeable.

## Engineering promotion heuristic

- Delta_PSNR must be positive at both **6k and 8k**.
- At **8k**, Delta_PSNR must be **>= +0.05 dB**.
- SSIM / RMSE HU / MAE HU must not show material reverse degradation. Because no
  numerical tolerance for “material” was specified, the helper flags any reverse
  direction for human review instead of automatically approving it.
- These are engineering screening rules, **not statistical significance claims**.

After fast64 passes, evaluate the **same selected checkpoint** on all **610 L333
slices**, with the same EMA/sampling/metrics. Compare against the corresponding
baseline checkpoint under FULL as well. Only methods retaining positive full-L333
improvement may proceed to formal 8-patient training. No test-set selection.
Final paper performance must come from **fresh formal-protocol training**;
fast-dev values must never be presented as final paper results.

## Fixed training, preprocessing and inference

Canonical configs: `../../configs/fast_dev_protocol.json` and
`../../configs/rddm_fast_dev.json` (JSON, matching the existing project format).
Model, training update and evaluation code are the accepted baseline implementation.
13,130,945 parameters; 10 RLEBs; dim64; mults(1,2,4,8); one channel and one U-Net;
input `[x_t, center_LDCT]`. Residual-only pred_res/l2; T1000, 2 sampling steps,
condition=True, sum_scale=.01. No DA-CLIP/DACB/Mamba/CSSM/attention/dose/anatomy module.

Stored value - 1024 gives HU; clip **[-1024,3072]**, normalize [0,1]. Diffusion
internally converts to [-1,1]. Evaluation converts prediction and target to HU,
clips both to **[-1000,1000]**, then averages per-slice PSNR, SSIM, RMSE HU, MAE HU.
FAST and FULL share the exact formal metric implementation. No HU protocol change.

Adam lr2e-4, betas(.9,.99), wd1e-9; batch2/accumulation1/effective2; seed20260829;
AMP off; gradient clip1; no scheduler. EMA .995 updated every step, no warmup.
Paired horizontal flip p=.5 plus uniform 0/90/180/270 rotation, no crop.
Validation: EMA, batch1, fixed seed20260829+original_slice_index, two sampling
steps. Training RNG is restored after validation.

At each validation, record PSNR/SSIM/RMSE/MAE, best step and delta versus the previous
checkpoint. Label PLATEAU if PSNR(8k)-PSNR(6k)<.01 dB. This is reporting only:
no automatic early stop or extension. All matched checkpoints remain remote.

## Provenance and smoke preservation

The historical 50-step acceptance files `FINAL_AUDIT.json` and
`smoke50_summary.json` are archived **unchanged**. Their “8k not started” statements
refer to the previous smoke task; they are not the status of the newly authorized
8k reference run. The original remote smoke outputs/logs/checkpoints are untouched.

The freeze updates only protocol metadata/documentation and the future-method
promotion/memory-provenance helpers. Model, train.py, runtime.py, evaluate.py and
all three split files match the accepted smoke source. Config changes only the
protocol SHA-256 after freezing these rules; no numerical training setting changes.
`archive_origin.json` records the original file hashes; `freeze_changes.json`
records the exact differences. The new Git branch starts at formal baseline commit
`d961c30f311911b893e190b526603e9369ea45e1`.

Patient selection evidence is in `data_audit.json`. L067/L143 were closest to the
median slice count (572.5), with complete expected pairs and no detected integrity
flags. NPY lacks physical z metadata; ordinal index and tissue fractions are only
coverage proxies, not anatomical completeness or clinical quality certification.

## Commands and isolation

From `baselines/founddiff_rddm_rleb`, in the existing remote corediff environment:

```bash
env CUDA_VISIBLE_DEVICES=0 CUBLAS_WORKSPACE_CONFIG=:4096:8 \
  OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1 \
  /home/xujian/miniconda3/envs/corediff/bin/python -m fast_dev.train \
  --config configs/rddm_fast_dev.json --device cuda:0 \
  --output experiments/fast_dev/runs/rddm_fast_baseline_8k_20260922
```

The reference run uses a dedicated tmux session and fresh output. This command is
recorded for reproducibility, not authorization to rerun or extend it. Resume an
interruption with `--resume /absolute/fast_checkpoint.pt` into a new directory,
with identical config/source and existing checkpoint paths preserved.

```bash
python -m fast_dev.evaluate --config configs/rddm_fast_dev.json \
  --checkpoint /absolute/fast_checkpoint.pt --mode FAST \
  --output experiments/fast_dev/runs/NEW_FAST_REVIEW
# Change --mode to FULL only for an authorized full-L333 review, with a new output.
```

No dataset, medical image, checkpoint, model weight, large training log, cache or
secret belongs in Git. Only code, frozen configs/splits and aggregate results are
archived. The formal baseline branch and its existing files remain unchanged.

## Completed unique reference

The authorized reference run completed 8000 optimizer steps. See [verified results](baseline_8k/RESULTS.md) and [verification](baseline_8k/VERIFICATION.json). Best fast64 EMA PSNR: 40.484568 dB at 8000 steps. All four matched validation checkpoints passed verification. No full-L333 inference or training extension was performed. This is the unique frozen fast comparison reference, not a final paper result.
