# FoundDiff-Lite-TimeDACB-2p5D

Independent 8k baseline based on [official FoundDiff](https://github.com/hao1635/FoundDiff), commit `c85a3a0e87ae2e4587650f0fb7efc15f3311d8e2`.

Retains the official RDDM residual diffusion, original single-convolution ResnetBlock/RLEB, four encoder/decoder stages, bottleneck, skips, time_mlp and residual head. Removes DA-CLIP, dose/prompt embeddings, CSSM/Mamba and attention. Nine zero-initialized TimeOnlyDACB modules apply `x + gate * (LayerNorm(x) * (1 + scale) + shift)` using only the official timestep embedding.

The U-Net input is `[I_t, LDCT_previous, LDCT_center, LDCT_next]`; the diffusion state and center residual output remain single-channel. Adjacent slices are patient-local with center replication at boundaries.

## Configuration

512x512; HU clip [-1000,1000]; model input [-1,1]. Train patients: L067, L096, L109, L143, L192, L286, L291, L310. Validation: historical fixed L333-Fast64 and its original order. No L506 access and no augmentation.

`dim=64`, `dim_mults=(1,2,4,8)`, `num_unet=1`, `pred_res`, residual L2 only, 1000 timesteps, 2 sampling steps, `sum_scale=0.01`, batch=2, accumulation=2, Adam lr=2e-4, betas=(0.9,0.99), weight_decay=0, EMA=0.995 per optimizer step, AMP=False, seed=10. Trained from random initialization. Parameters: **13,614,209**. Peak allocated GPU memory: **4.93 GiB** on RTX 3090. Runtime including validation: **64.23 minutes**.

## EMA results

Main window [-1000,1000] HU; PSNR data_range=2000, SSIM win_size=7. Metrics are computed independently per slice and averaged over 64 slices.

| Step | PSNR dB | SSIM | RMSE HU | MAE HU |
|---:|---:|---:|---:|---:|
| 2000 | 40.266046 | 0.943362 | 19.498945 | 13.454564 |
| 4000 | 40.692783 | 0.947166 | 18.569038 | 12.655514 |
| 6000 | 40.797116 | 0.947982 | 18.348867 | 12.506822 |
| 8000 | **40.846046** | **0.948312** | **18.247448** | **12.438258** |

8k soft-tissue window [-160,240] HU: PSNR 28.527524 dB, SSIM 0.842327, RMSE 15.112703 HU. Per-slice CSVs and the full table are in `experiments/FoundDiff_Lite_TimeDACB_8k/`.

MLSF+Pyramid@8k reference: 40.956836033 dB; this baseline is 0.110790 dB lower. Historical simplified FoundDiff_RDDM_RLEB@8k: approximately 40.4846 dB. These are contextual comparisons: architecture, loss, initialization, training patients, normalization, effective batch, seed and/or diffusion implementation differ. No matched causal comparison is claimed.

## Reproduction

Environment used: Python 3.11.16, PyTorch 2.8.0+cu128, NumPy 2.3.2, scikit-image 0.25.2, einops 0.8.1. Dependencies are listed in `requirements.txt`.

From this directory, set the local Mayo NPY root in `experiments/FoundDiff_Lite_TimeDACB_8k/config.json`. Expected filenames are `{patient}_25_{index:03d}_img.npy` and `{patient}_target_{index:03d}_img.npy`, shape 512x512, stored values converted to HU by subtracting 1024. The fixed validation list is included.

```bash
CUDA_VISIBLE_DEVICES=0 python -u preflight.py
CUDA_VISIBLE_DEVICES=0 python -u train.py
```

Preflight checks all nine DACBs, exact identity initialization, real-batch forward/backward, timestep gradients, two-step sampling and memory. Training saves step0 with RNG/optimizer/stream state and model/EMA checkpoints at 2k/4k/6k/8k. Best checkpoint: `checkpoints/step_8000.pt`, key `ema_state_dict`. Explicit resume: `python -u train.py --resume PATH`.

The artifact package contains source, configuration and numeric results. Private CT data, checkpoint binaries and training image outputs are retained on the experiment machine.

See `experiments/FoundDiff_Lite_TimeDACB_8k/REPORT.md` for architecture details and `FINAL_AUDIT.json` for the completed 8000-step/checkpoint/metric audit.

Packaging note: preflight reads the included upstream source reference so its AST audit works from this repository subdirectory. Trained model and diffusion code are unchanged from the experiment.
