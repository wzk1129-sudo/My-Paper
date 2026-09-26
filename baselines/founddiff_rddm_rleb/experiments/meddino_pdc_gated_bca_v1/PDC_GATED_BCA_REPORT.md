# MedDINO-PDC-Gated-BCA-V1

PiDiNet-Tiny carv4 PDC features from center LDCT are projected, anatomy-gated and added to the unchanged E12 H128/H64 hints.
The E12-BCA window attention (Q=RDDM encoder, K/V=fused structure) and trainable main backbone are unchanged.

## Full L333 (610 slices, step6000 EMA)

| Arm | PSNR | SSIM | RMSE HU | MAE HU |
|---|---:|---:|---:|---:|
| PDC-gated BCA | 40.854364 | 0.948517 | 18.229297 | 12.039938 |
| E12-BCA | 40.856054 | 0.948571 | 18.225127 | 12.011960 |

Delta PSNR -0.001690 dB; slice bootstrap 95% CI [-0.002117,-0.001270]; positive fraction 0.374.
The CI is within one L333 patient and is not patient-level significance.
beta128=0.092798, beta64=0.089676; alpha128=0.201746, alpha64=0.164211.
Train time 0.445s/step; inference 141.2s; peak inference VRAM 0.80 GiB.
No training during Full-L333; L506, external memory and neighbor targets untouched.
