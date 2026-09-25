# Mayo-2016 Soft-Tissue Metric Comparison V1

Evaluation-only on L333; original model input normalization, checkpoints, and 2-step sampler are unchanged.
Per-slice arithmetic mean over all 610 slices; common interior uses indices 1 through 608 (608 slices).
Soft-tissue values are clipped to [-160,240] HU, normalized to [0,1] for PSNR/SSIM, with data_range=1.
RMSE and MAE are computed in clipped HU. Original-window values use frozen slice_metrics.

| Method | Training patients | PSNR [-1000,1000] | PSNR soft | SSIM soft | RMSE soft HU | MAE soft HU | Status |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| RAW_QUARTER | - | 34.7539 | 22.5561 | 0.7955 | 30.0617 | 14.9227 | INPUT_BASELINE |
| FORMAL_RDDM_60K | 8 fixed train patients | 40.6581 | 28.3294 | 0.8426 | 15.4683 | 7.6339 | PAPER_COMPARABLE_8PATIENT |
| DIRECT_CONCAT_DEV_12K | L067, L143 | 40.7778 | 28.4604 | 0.8518 | 15.2296 | 7.5280 | DEV_ONLY |

Literature reference values below were supplied in the task and were not independently source-verified.
Numerical positioning is descriptive; differences in acquisition and evaluation can affect comparability.

| Reference | PSNR | SSIM | RMSE HU | RDDM Delta PSNR | RDDM Delta SSIM | RDDM Delta RMSE HU |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| LDCT | 22.556 | 0.7895 | 30.062 | +5.7734 | +0.0531 | -14.5937 |
| RED-CNN | 27.935 | 0.8500 | 16.174 | +0.3944 | -0.0074 | -0.7057 |
| DuGAN | 28.088 | 0.8416 | 15.894 | +0.2414 | +0.0010 | -0.4257 |
| Uformer | 28.225 | 0.8504 | 15.647 | +0.1044 | -0.0078 | -0.1787 |
| CTformer | 28.037 | 0.8442 | 15.990 | +0.2924 | -0.0016 | -0.5217 |
| CT-Mamba | 27.932 | 0.8477 | 16.178 | +0.3974 | -0.0051 | -0.7097 |
| FMDNet | 28.340 | 0.8568 | 15.322 | -0.0106 | -0.0142 | +0.1463 |

The 2-patient Direct Concat result is DEV_ONLY and is not a fair paper comparison.
Full and common-interior mean/std/median for every metric are in SUMMARY.json.
Three fixed random slices passed independent PSNR/RMSE formula checks.
L506 opens = 0; external-memory opens = 0; training and optimizer steps = 0.
No predictions or medical images were saved.

The raw quarter-dose PSNR (22.5561 dB) and RMSE (30.0617 HU) nearly match the user-supplied LDCT reference (22.556 dB; 30.062 HU). Its SSIM differs (0.7955 versus 0.7895), so identical metric implementation is not established. Original-window and soft-window PSNR values have different clipping and data ranges and must not be compared directly.

The first inference-only pass completed both 610-slice loops but its final summary check failed because two archived-PSNR configuration keys were absent. The corrected second pass reran the same read-only inference and passed all checks. No model or checkpoint was changed and no optimizer step occurred.
