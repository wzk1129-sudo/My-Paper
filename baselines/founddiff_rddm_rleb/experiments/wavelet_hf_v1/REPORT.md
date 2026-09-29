# Wavelet HF supervision on 2P5D Direct RDDM

Only the training loss adds fixed orthonormal Haar LH/HL/HH residual supervision. Model, sampler, evaluator, patients, seeds, optimizer and 2.5D inputs are unchanged.

Calibration lambda_HF: 0.83191919; status PASS; formal training started from a fresh matched initialization.

| Step | PSNR | SSIM | RMSE HU | MAE HU | Delta PSNR vs Direct |
|---:|---:|---:|---:|---:|---:|
| 2000 | 39.043841 | 0.931680 | 22.4288 | 15.1510 | +1.069041 |
| 4000 | 40.277239 | 0.943322 | 19.4664 | 12.9552 | +0.072876 |
| 6000 | 40.538047 | 0.945708 | 18.8958 | 12.4939 | +0.008430 |
| 8000 | 40.653121 | 0.946640 | 18.6479 | 12.2574 | -0.004856 |

Gate: NO_PROMOTION.
Completed optimizer steps: 8000.
Parameter count: 13137217; peak allocated VRAM: 4480969216 bytes.
Mean train seconds/step: 0.3261.
Validation uses EMA, fixed L333-fast64 and the original metric implementation.

Full L333 not run because the predefined promotion gate did not pass.

L506, external-memory and neighbor NDCT access: 0. No other architecture or loss was added.
