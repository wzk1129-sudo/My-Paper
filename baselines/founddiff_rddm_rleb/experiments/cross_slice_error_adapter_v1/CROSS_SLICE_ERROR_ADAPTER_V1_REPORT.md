# Cross-Slice Residual Error Adapter V1

Frozen 12k EMA Direct Concat; only the five-channel additive adapter was optimized.
L067/L143 train, fixed L333-fast64 EMA, original HU and 2-step sampler.

| Step | PSNR | Delta PSNR | SSIM | RMSE HU | MAE HU | delta_r RMS | Baseline MSE | Corrected MSE | corr(abs(delta_r), D_abs) |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 40.767627 | +0.000000 | 0.947724 | 18.4085 | 12.1283 | 0 | 8.84722e-05 | 8.84722e-05 | 0.00000 |
| 500 | 40.767471 | -0.000156 | 0.947725 | 18.4088 | 12.1272 | 1.65234e-05 | 8.84722e-05 | 8.84576e-05 | -0.17169 |
| 1000 | 40.767343 | -0.000285 | 0.947726 | 18.4090 | 12.1265 | 2.43784e-05 | 8.84722e-05 | 8.84502e-05 | -0.28805 |
| 2000 | 40.767220 | -0.000408 | 0.947726 | 18.4093 | 12.1252 | 4.30823e-05 | 8.84722e-05 | 8.84339e-05 | -0.11893 |
| 4000 | 40.767134 | -0.000493 | 0.947725 | 18.4094 | 12.1245 | 5.21537e-05 | 8.84722e-05 | 8.84278e-05 | -0.04450 |
| 6000 | 40.767370 | -0.000258 | 0.947722 | 18.4089 | 12.1243 | 6.06694e-05 | 8.84722e-05 | 8.84243e-05 | 0.25135 |

Promotion: **NO_PROMOTION**. Strong signal: **False**.
Failure interpretation: **ADAPTER_ACTIVE_BUT_NOT_USEFUL**.
Backbone state SHA256 unchanged: cf38b0b7a50d5ac5bc6579d64f7031e7e00e8f1824867debf6ec3b38ca1ddab9.
Original Direct Concat checkpoint SHA256 unchanged: 151b7e58d2ca0d936be0842434e3d17809b325f144d0d48e47c8565287bcdcdf.
No L506, external-memory, or neighbor-NDCT reads.
Full L333 performed only if promoted; paired analysis is within-stack stability.
