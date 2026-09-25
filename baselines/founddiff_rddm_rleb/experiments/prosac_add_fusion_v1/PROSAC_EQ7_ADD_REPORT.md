# ProSAC Eq.(7) add-fusion probe on RDDM ControlNet

Only the edge aggregation changed: concat(Sobel,Laplacian) -> Conv1x1_2toC became (Sobel + Laplacian) -> Conv1x1_1toC. The [-1000,1000] HU window, frozen MedDINO, APGC multiplication and all training/evaluation conditions remained fixed.

| Arm | 500 PSNR | 1k PSNR | 2k PSNR | Delta vs C3 | Delta vs old C4 | Positive slices vs C3 | RMSE HU | Soft PSNR |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| C1_CAPACITY | 40.76968745 | 40.77046399 | 40.77146815 | -0.01674104 | -0.01071542 | NA | 18.400825 | 28.458096 |
| C3_ANATOMY | 40.77394810 | 40.77986161 | 40.78820919 | +0.00000000 | +0.00602562 | NA | 18.366432 | 28.477844 |
| C4_APGC | 40.77079853 | 40.77533963 | 40.78218357 | -0.00602562 | +0.00000000 | NA | 18.378978 | 28.471218 |
| C4_ADD | 40.77228026 | 40.77787888 | 40.78541252 | -0.00279667 | +0.00322895 | 0.281 | 18.372295 | 28.474493 |

## Gate, slices, and structure activity

- 2k promotion to 4k/6k: False.
- C4_ADD trainable parameters: 6261536; old C4: 6261760; difference: -224.
- 2k vs C3_ANATOMY: mean -0.00279667, median -0.00211200, positive fraction 0.281, p10/p25/p50/p75/p90 = -0.008118/-0.004465/-0.002112/+0.000251/+0.001319.
- 2k vs C4_APGC: mean +0.00322895, median +0.00237416, positive fraction 0.875, p10/p25/p50/p75/p90 = -0.000107/+0.000968/+0.002374/+0.004608/+0.006401.
- 2k vs C1_CAPACITY: mean +0.01394437, median +0.00976009, positive fraction 0.984, p10/p25/p50/p75/p90 = +0.004207/+0.005860/+0.009760/+0.018506/+0.027087.
- Activity at last matched step is in CONTROL_ACTIVITY.csv and OLD_C4_CONTROL_ACTIVITY.csv. Sobel, Laplacian, and their sum RMS are recorded for C4_ADD; the archived old C4 did not record separate Sobel/Laplacian RMS.

## Integrity and scope

- Direct main, copied Control Encoder, ZeroConv taps, loss, sampler, evaluator, MedDINO source/weights, Fast64 split and seed were unchanged.
- Main and MedDINO remained frozen; L506, external memory and neighbor NDCT opens were zero.
- Conclusions are limited to this Fast64 implementation and matched checkpoints. No Full-L333, 8-patient, or test evaluation was run.
