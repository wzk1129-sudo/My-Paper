# Sampler calibration

FoundDiff RDDM+RLEB backbone adapted to our fixed Mayo-2016 protocol.

Control reproduced: PASS. Full L333, 610 slices, stride=1, original seed and HU metrics.

| Stage | Steps | sum_scale | PSNR | SSIM | RMSE HU | MAE HU | Validation seconds |
|---|---:|---:|---:|---:|---:|---:|---:|
| control | 2 | 0.01 | 40.65805039 | 0.94613206 | 18.64887882 | 12.18173656 | 119.624 |
| timesteps | 1 | 0.01 | 40.65660907 | 0.94613636 | 18.65170720 | 12.18354293 | 70.007 |
| timesteps | 2 | 0.01 | 40.65805039 | 0.94613206 | 18.64887882 | 12.18173656 | 119.624 |
| timesteps | 4 | 0.01 | 40.65943533 | 0.94611557 | 18.64631527 | 12.18042919 | 163.137 |
| timesteps | 8 | 0.01 | 40.66300035 | 0.94617944 | 18.63841185 | 12.17606464 | 285.252 |
| sum_scale | 8 | 0.0 | 40.67628429 | 0.94627777 | 18.61057030 | 12.16186451 | 284.322 |
| sum_scale | 8 | 0.0025 | 40.67152004 | 0.94624123 | 18.62056334 | 12.16697820 | 283.608 |
| sum_scale | 8 | 0.005 | 40.66790164 | 0.94621417 | 18.62815257 | 12.17085474 | 281.386 |
| sum_scale | 8 | 0.01 | 40.66300035 | 0.94617944 | 18.63841185 | 12.17606464 | 285.252 |
| sum_scale | 8 | 0.02 | 40.65835604 | 0.94615214 | 18.64806029 | 12.18084401 | 282.834 |
| sum_scale | 8 | 0.04 | 40.65688363 | 0.94615869 | 18.65090576 | 12.18185961 | 279.798 |

Best: steps=8, sum_scale=0.0.

Deltas vs control: {"psnr": 0.01823390092906152, "ssim": 0.00014571002748942075, "rmse_hu": -0.03830852232401938, "mae_hu": -0.019872052552269537}

Model weights unchanged; no training performed; L506 untouched; evaluation protocol unchanged.
Frozen source, CONFIG, checkpoint and original metrics SHA256 unchanged.
Runtime sum_scale affects initial noise only; diffusion coefficient buffers remain unchanged.
Timing uses the original validate function and includes validation loss, data loading and metrics. Shared control / scale=0.01 measurements are reused, not rerun.
Selection is on L333; this is sampler calibration, not an independent generalization result. No baseline artifacts were overwritten.
