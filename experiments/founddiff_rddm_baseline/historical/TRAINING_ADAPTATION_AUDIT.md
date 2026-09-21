# FoundDiff_RDDM_RLEB_Mayo2016_smallset_v1

Description: **FoundDiff RDDM+RLEB backbone adapted to our fixed Mayo-2016 protocol**.
This is not an official FoundDiff Table-V RDDM reproduction.

## Frozen model

`src/rddm_rleb.py` is unchanged, SHA-256 `e8ed93223fbdf4a0e2521f73c7a4f213561f6a191a1a7b2a5f95934b85ebf574`.
Official-source ancestor: hao1635/FoundDiff `c85a3a0e87ae2e4587650f0fb7efc15f3311d8e2`; current worktree branch `audit/rddm-rleb-table-v-20260921`.
No edits to old `train_rddm_rleb.py` or `smoke_rddm_rleb.py`. Formal adaptation lives in this directory.

The factory remains dim=64, dim_mults=(1,2,4,8), channels=1, num_unet=1, pred_res, l2, T=1000, sampling=2, condition=True, sum_scale=0.01. 13,130,945 parameters; 10 RLEBs. Input to U-Net is `[x_t, center_LDCT]`, two channels. No neighboring image, dose/anatomy embedding, DA-CLIP, DACB, attention, Mamba, CSSM, or additional structural module.

## Read-only audit before adaptation

RDDM-CT source consulted:
`/home/xujian/projects/RDDM_CVPR2024/RDDM_CT/experiments/7_RDDM_CT_Backbone_v1/{config.json,rddm_ct.py,evaluation.py}`.
Latest structure-worktree config was checked only for the existing data/evaluation agreement; no structural network was imported.

Dataset root: `/home/xujian/projects/CoreDiff/data_preprocess/gen_data/mayo_2016_npy`.
Allowed named-file probes (never a glob/listdir of the mixed-patient data directory) confirmed:

| Role | Patient | Paired slices |
|---|---|---:|
| Train | L067 | 560 |
| Train | L096 | 823 |
| Train | L109 | 318 |
| Train | L143 | 585 |
| Train | L192 | 600 |
| Train | L286 | 525 |
| Train | L291 | 856 |
| Train | L310 | 533 |
| Train total | 8 existing patients | 4800 |
| Validation | L333 | 610 |

All permitted patient slice-index sets are contiguous from zero. Every center `PATIENT_25_NNN_img.npy` has the same-patient/same-index `PATIENT_target_NNN_img.npy`. No patient reassignment or new split.
Patient-key/slice-key alignment was verified. This does not claim independent DICOM geometry/registration re-validation of these already generated NPY pairs.

The held-out test patient was not enumerated, loaded, visualized, evaluated, or used for selection. The new data API accepts only `train` and `validation`; all accesses use hardcoded allowed patient/count contracts and exact paths. No original medical data is written, moved, repaired or copied locally.

18 read-only pair samples (2 per allowed patient) were 512x512 uint16, finite. Loader outputs are float32 `[1,512,512]`, in [0,1]. Global sampled normalized extrema were 0 and 0.999755859375. Full metadata and per-pair stats: `DATA_AUDIT.json` and each run's `data_audit.json`.

## Normalization and metric reuse

Copied verbatim by AST-selected source definitions into `frozen_protocol.py`:
- `normalize_stored_ct`, `normalized_to_hu` from existing `rddm_ct.py`;
- `_metric_tensors`, `slice_metrics` from existing `evaluation.py`.

Source paths/hashes/function inventory: `protocol_provenance.json`. The acceptance test checks AST identity of every copied function. This avoids importing the other project's model, foundation modules or 2.5D loader.

Normalization: HU = float32(stored) - 1024; clip HU to [-1024,3072]; normalize (HU+1024)/4096. The unchanged diffusion performs its own [0,1] to [-1,1] conversion. Validation predictions/targets are converted back to HU and clipped to the existing [-1000,1000] metric window. Whole-image, per-slice PSNR/SSIM/RMSE-HU/MAE-HU are averaged arithmetically over all 610 L333 slices. PSNR data_range=2000 HU. SSIM uses scikit-image structural_similarity, win_size=7, gaussian_weights=False, sample covariance, K1=.01, K2=.03, grayscale and float64 arrays, matching frozen source.

## Training configuration

Canonical configuration: `CONFIG.json`. Adam lr=2e-4, betas=(.9,.99), weight_decay=1e-9; batch=2, accumulation=1, effective batch=2; seed=20260829; AMP=False; max_optimizer_steps=60000; gradient max_norm=1.0. No scheduler was requested; LR is constant and scheduler_state_dict=None is saved.

EMA: decay=.995 (FoundDiff training's decay), float32 copy at step 0, update each optimizer step, no warmup. Validation and checkpoint selection always use EMA. This explicit update schedule is an adaptation choice, not a claim about the authors' unreleased ablation schedule.

Paired augmentation: horizontal flip p=.5 and uniform rotation among 0/90/180/270 degrees. One random draw set is applied identically to each LDCT/NDCT pair. No crop, intensity jitter, augmentation noise injection, elastic transform or spatial scaling. Forward diffusion noise remains unchanged as required.

Serial data stream (`num_workers=0`) records permutation, cursor, epoch and its own NumPy RNG; this makes resume exact without untracked DataLoader prefetch state. Each epoch contains all 4800 samples in a fresh random permutation, batch=2. Deterministic algorithms are enabled; cuDNN benchmark=False; TF32=False; CUBLAS_WORKSPACE_CONFIG=:4096:8. The timing benchmark measures these same settings.

## Validation / checkpoint / recovery behavior

- Every 10,000 optimizer updates, evaluate **all 610 L333 slices**, stride=1, sampling_timesteps=2, EMA only.
- Isolated CPU/CUDA RNG scope; for each slice seed=20260829+slice_index, with fixed B=1. This reproduces initial sampling noise without altering model/sampler math. Validation loss uses a separate fixed seed offset +1000000. Training RNG restored afterward.
- `validation_metrics.csv` includes step, weights, slice count, PSNR, SSIM, RMSE HU, MAE HU, validation loss, train loss, pre-clip gradient norm, LR, peak VRAM, sec/step, validation duration and immutable checkpoint path.
- Save `checkpoints/step_10000.pt` through `step_60000.pt`, including all requested 30k/40k/50k/60k matched checkpoints. None are deleted. Save recovery `checkpoints/latest.pt` every 1000 updates within the same run.
- `best_psnr.pt` is an atomic hardlink alias to the best immutable full-L333 EMA checkpoint. Raw model and EMA states are both stored. L333 mean PSNR alone selects best; no test selection.
- Each checkpoint stores raw model, EMA, EMA update count, optimizer, optimizer step, Python/NumPy/PyTorch/CUDA RNG, sampler permutation/cursor/RNG, complete config, source SHA-256s, Git commit, validation history, best-checkpoint location, timing, gradient statistics and VRAM.
- Resume requires `--resume /absolute/checkpoint.pt` and a **new output directory**. Source hashes and configuration must match; optimizer step and all random/data-stream state are restored. Previous outputs remain intact. Matched prior checkpoints remain at their recorded immutable locations.
- Checkpoint and JSON writes use fsync plus atomic replace. New run directories use exist_ok=False; duplicate starts cannot overwrite a run.
- 60k is a hard stop, with no automatic extension. Final summary includes all matched metrics, best step/path, recent trends, time, memory and gradient/failure statistics. Plateau is explicitly a heuristic: both last 10k PSNR changes have magnitude below .05 dB. Potential extension is flagged for discussion only if the last interval gains >=.1 dB with nondecreasing SSIM.

## Executable entry points

All commands execute through remote SSH in the existing FoundDiff worktree, with existing Conda `corediff` and GPU 0. No install or environment change.

```bash
# After passing the recorded preflight gate:
python -m mayo2016_smallset_v1.train --config mayo2016_smallset_v1/CONFIG.json --output NEW_RUN_DIR --device cuda:0 --gate mayo2016_smallset_v1/PREFLIGHT_PASS.json
# Recovery into a new directory, same fixed config and gate:
python -m mayo2016_smallset_v1.train --config mayo2016_smallset_v1/CONFIG.json --output NEW_RESUME_DIR --device cuda:0 --gate mayo2016_smallset_v1/PREFLIGHT_PASS.json --resume CHECKPOINT.pt
```

Long executions use tmux. Every launch records session, complete command, current Git state, Conda environment, GPU, log/output path, timestamp and termination condition in `launches/*_launch.json`.

## Acceptance and runtime artifacts

Data audit PASS before smoke. Planned gates, only marked PASS once their evidence exists:
1. 20 updates with real Train pairs, full 512x512 B=2, finite loss and all parameter gradients, output shape and VRAM checks.
2. Resume from update 10 into a new run, complete update 20, compare raw weights, EMA weights, optimizer tensors, RNG and sampler state exactly against uninterrupted update 20.
3. Paired augmentation identity check, frozen metric/normalization AST equality, two repeated L333 validation samples with unchanged training RNG.
4. Separate fresh 200-update timing run, same full configuration; report all-200 and post-20-warmup seconds/update, peak VRAM and projected 60k compute time. No benchmark weights initialize formal training.
5. Formal start permitted only when `PREFLIGHT_PASS.json` matches current source/model/config hashes and gate sizes.

`train.jsonl`, `validation_metrics.csv`, `checkpoint_manifest.json`, `source_manifest.json`, `CONFIG.json`, `progress.json` and `final_training_summary.{json,md}` live in each run directory. A running summary explicitly says RUNNING and leaves unavailable final metrics empty. Error/failure logs are preserved. No clinical accuracy, convergence, plateau or final PSNR is claimed before observed results.

## Completed preflight results

All gates PASS on 2026-09-21. 20 full-resolution real-pair updates and exact interrupted/resumed step-20 comparison passed. Paired transforms, metric-source equivalence, RNG-isolated repeated validation passed. 200-step benchmark: 0.3502815500026918 s/update (all steps), 0.35639697439134277 s/update (after first 20); peak 4470513664 bytes; estimated 60k compute 5.939949573189046 hours. No nonfinite values, maximum pre-clip norm 46.26458740234375. Formal initialization is from scratch, not from any acceptance checkpoint.

## Formal launch verified

Start: 2026-09-21T17:15:16.797102+08:00

Session: `founddiff_rddm_rleb_mayo2016_60k_20260921`

Output: `/home/xujian/projects/CoreDiff/experiments/founddiff_table_v_20260921/rddm_rleb/mayo2016_smallset_v1/runs/FoundDiff_RDDM_RLEB_Mayo2016_smallset_v1_60k_20260921`

Log: `/home/xujian/projects/CoreDiff/experiments/founddiff_table_v_20260921/rddm_rleb/mayo2016_smallset_v1/launches/formal60k.log`

Complete invocation is frozen in `launches/formal60k_launch.json`. Initial model started at step 0; launch verification observed step 252 with finite loss/gradient. Hourly completion/failure-only follow-up: `founddiff-rleb-mayo2016-60k`. 60k metrics are pending; running summaries do not claim completed results.
