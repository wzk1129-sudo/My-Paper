# FoundDiff RDDM+RLEB backbone adapted to our fixed Mayo-2016 protocol

Experiment: `FoundDiff_RDDM_RLEB_Mayo2016_smallset_v1`.
Run: `FoundDiff_RDDM_RLEB_Mayo2016_smallset_v1_60k_20260921`.
Status: **COMPLETE at optimizer step 60000**, no automatic extension.
This is a reconstructed backbone adapted to our protocol, not an official
FoundDiff Table-V reproduction.

## Fixed data and evaluation protocol

- Dataset: Mayo-2016.
- Train: the existing 8 patients, exactly 4800 paired slices; no re-splitting.
  IDs/counts are frozen in `CONFIG.json`.
- Validation: L333, all 610 slices, stride 1.
- Test: L506, all 526 slices in the pre-existing protocol. This count is recorded
  from the fixed protocol; it was not verified by accessing test data in this run.
- **L506 is strictly untouched during development/model selection.** It was not
  enumerated, loaded, visualized, evaluated or used for checkpoint selection.
- Preprocessing: stored NPY value minus 1024 gives HU; clip to [-1024,3072] HU;
  normalize to [0,1] with `(HU + 1024) / 4096`, float32, full 512x512.
- Inputs: center LDCT and matching center NDCT only, each `[B,1,512,512]`.
  No adjacent slices enter the model. Diffusion converts to [-1,1] internally.
- Evaluation: convert predictions and targets back to HU, clip both to
  [-1000,1000] HU, evaluate whole slices, and average per-slice metrics.
- Metrics: PSNR, SSIM, RMSE(HU), MAE(HU). PSNR uses data range 2000 HU.
  SSIM: window 7, Gaussian weights off, sample covariance, K1=0.01, K2=0.03,
  float64 grayscale, matching the frozen RDDM-CT evaluation functions.
- **EMA weights are used for validation.** Two sampling steps, batch 1; seed
  `20260829 + slice_index` fixes initial sampling noise. Validation restores the
  training RNG afterward. Validation loss uses a separate +1000000 seed offset.
- **Model selection is based only on L333 validation PSNR.**

## Network and training

`dim=64`, `dim_mults=(1,2,4,8)`, `channels=1`, `num_unet=1`;
13,130,945 parameters and 10 RLEBs. Input `[x_t, center_LDCT]`.
RDDM residual diffusion, `objective=pred_res`, `loss_type=l2`, `timesteps=1000`,
`sampling_timesteps=2`, `condition=True`, `sum_scale=0.01`.
The LDCT image condition and timestep are retained; DA-CLIP, dose/anatomy
conditioning, DACB, adaLN-Zero, attention, Mamba/SS2D and CSSM are absent.

Adam: LR 2e-4, betas (0.9,0.99), weight decay 1e-9. Batch 2, gradient
accumulation 1, effective batch 2, max 60000 optimizer steps, seed 20260829.
AMP off, no scheduler, clip gradient norm to 1.0 and log pre-clip norm.
EMA decay 0.995, update every step without warmup. Paired horizontal flip p=0.5
and uniform rotations 0/90/180/270; no cropping or intensity augmentation.

Checkpoints include model, EMA, optimizer, step, Python/NumPy/PyTorch/CUDA RNG,
data-stream permutation/cursor/RNG, config and source hashes. Recovery checkpoints
were saved every 1000 updates and full L333 validation every 10000 updates.
The 30k/40k/50k/60k checkpoints remain on the original remote host.

## Observed validation results

These numbers come from the completed run's `validation_metrics.csv` and
`final_training_summary.json`, not estimates. All rows use full-L333 EMA validation.

| Step | PSNR (dB) | SSIM | RMSE HU | MAE HU |
|---:|---:|---:|---:|---:|
| 30k | 40.6322 | 0.945992 | 18.7013 | 12.2164 |
| 40k | 40.6332 | 0.945891 | 18.6977 | 12.2135 |
| 50k | 40.6350 | 0.945868 | 18.6980 | 12.2154 |
| **60k** | **40.6581** | **0.946132** | **18.6489** | **12.1817** |

Final best: **step 60000**, PSNR **40.658050390663206 dB**.
`best_psnr.pt` aliases `checkpoints/step_60000.pt`; load `ema_state_dict` to
reproduce the validation metrics. SHA-256:
`49aa8ce2ee5c98c4deeaf3067d7059f54f18d7715e30439cd9326b6177867469`.
The checkpoint manifest is included; the weights are deliberately excluded.

The earlier 50k best of 40.6350 dB was provisional. The completed 60k validation
supersedes it; checkpoint selection is no longer pending.

## Conclusions and runtime

Performance substantially plateaued after 30k, with small further gains rather
than complete stagnation. PSNR improved 0.02488 dB over 40k-to-60k, including
0.02303 dB over 50k-to-60k. These observations do not establish a benefit from
extending to 80k/100k; no extension was started. There are no test results.

Total recorded wall time: 19051.8766 seconds (5h17m32s); training compute:
18255.2434 seconds. Peak allocated VRAM: 4470513664 bytes (4.1635 GiB).
All 60000 training steps were continuous with finite loss/gradient values.
Pre-clip gradient maximum was 46.2646 during early training; count above 100 was
zero. No sustained abnormal gradient or nonfinite failure was observed. AMP was
disabled, so AMP overflow does not apply. Validation loss at 60k: 8.85190e-5.

Acceptance passed before the formal run: data audit; 20 steps at 512x512/B=2;
bit-identical interrupted/resumed step-20 model/EMA/optimizer/RNG/data stream;
paired transform and metric AST checks; repeated deterministic validation;
200-step timing benchmark. No acceptance weights initialized formal training.

## Snapshot integrity and included evidence

Seven executable source files match the original run's source hashes byte for
byte. `CONFIG.json` has SHA-256
`2eb687f8eb28901704b9d7d6e53f099530f7b8297ec91e3929282b3682d9af0e`.
The training snapshot is under `../../baselines/founddiff_rddm_rleb/`.
The exported source/config bytes are unchanged; only publication documentation
and exclusion rules were added. Original absolute paths in the archived metadata
identify the actual run and are not portable defaults.

`historical/` preserves the initial architecture audit and pre-training adaptation
audit verbatim. Their statements such as "60k metrics pending" describe the date
they were written and are superseded by this final report. Their referenced
medical samples, full logs and checkpoints remain remote and are not included.

No code change, package installation, new training, test-set access or modification
to other running jobs was performed for this publication. Only explicitly
allowlisted code and lightweight aggregate records were transferred locally.
