# RDDM-CT Backbone v1

This experiment adapts the official CVPR 2024 RDDM restoration implementation to paired Mayo2016 CT denoising while preserving the official residual diffusion coefficients and reverse update equations.

## Frozen contract

- State: center-slice `x_t`, shape `[B,1,H,W]`.
- Condition: LDCT context `[z-1,z,z+1]`, shape `[B,3,H,W]`; volume boundaries replicate the nearest slice.
- Residual: `r = LDCT_z - NDCT_z` in the fixed normalized diffusion domain.
- Forward process: `x_t = NDCT_z + A_t r + B_t epsilon`.
- Objective: equal-weight `L1(residual) + L1(noise)`.
- Backbone: one shared official-topology U-Net (`64`, multipliers `1,2,4,8`) with dual shallow stems and two 1x1 prediction heads.
- Training diffusion steps: 1000; inference steps: 10; deterministic DDIM update (`eta=0`).
- Fixed patient split: train `L067,L096,L109,L143,L192,L286,L291,L310`; validation `L333`; final test `L506`.
- Fixed preprocessing: stored pixel to HU by subtracting 1024, clip to `[-1024,3072]`, normalize linearly to `[0,1]`; inverse is exact within the clipping interval.

The official implementation is imported read-only from experiment 2 and its SHA-256 is checked by acceptance. No pretrained checkpoint is loaded unless `--resume` is explicitly supplied.

## Commands

Run the one-time acceptance and 512x512 batch calibration:

```bash
PYTHONDONTWRITEBYTECODE=1 LD_LIBRARY_PATH=/usr/lib/wsl/lib:${LD_LIBRARY_PATH:-} /home/xujian/miniconda3/bin/conda run -n rddm_cvpr2024 python acceptance.py --config config.json
```

Start formal training from scratch:

```bash
PYTHONDONTWRITEBYTECODE=1 LD_LIBRARY_PATH=/usr/lib/wsl/lib:${LD_LIBRARY_PATH:-} /home/xujian/miniconda3/bin/conda run -n rddm_cvpr2024 python train.py --config config.json
```

Resume is explicit and fails if the checkpoint is missing:

```bash
... python train.py --config config.json --resume /absolute/path/to/last.pt
```

Evaluate a checkpoint with slice-, patient-, and overall metrics:

```bash
... python evaluate.py --config config.json --checkpoint /absolute/path/to/best_psnr.pt --split test
```
