# RDDM vs matched LDCT-only residual regression V1

Both arms use the same 13,130,945-parameter UnetRes initialization, 24,000 paired augmented training samples, Adam/EMA settings, and frozen Mayo Fast protocol.
Control retains the unchanged RDDM residual diffusion. Regression uses [LDCT, LDCT], fixed zero time, and direct residual MSE. No pretrained checkpoint was used.

## Fast64 EMA checkpoints

| Step | Control PSNR | Regression PSNR | Delta dB | Control SSIM | Regression SSIM | Control RMSE HU | Regression RMSE HU | Control MAE HU | Regression MAE HU |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2000 | 38.47870 | 39.06579 | +0.58709 | 0.926035 | 0.930594 | 23.9292 | 22.3673 | 15.7929 | 14.7798 |
| 4000 | 40.14539 | 40.27009 | +0.12470 | 0.941559 | 0.943183 | 19.7640 | 19.4835 | 13.0245 | 12.7744 |
| 6000 | 40.38720 | 40.46496 | +0.07776 | 0.944026 | 0.944758 | 19.2251 | 19.0579 | 12.5807 | 12.5241 |
| 8000 | 40.48457 | 40.53827 | +0.05370 | 0.944869 | 0.945451 | 19.0149 | 18.9003 | 12.4597 | 12.4243 |
| 10000 | 40.52205 | 40.57519 | +0.05314 | 0.945066 | 0.945769 | 18.9336 | 18.8221 | 12.4228 | 12.3702 |
| 12000 | 40.57651 | 40.60593 | +0.02943 | 0.945714 | 0.946039 | 18.8192 | 18.7570 | 12.3521 | 12.3330 |

## Full L333 (610 slices, step 12000 EMA)

RDDM: PSNR 40.58002 dB, SSIM 0.945784, RMSE 18.8103 HU, MAE 12.3431 HU; validation 139.02 s; training 3818.3 s (0.3182 s/step), peak 4.16 GiB.
Direct regression: PSNR 40.61033 dB, SSIM 0.946114, RMSE 18.7466 HU, MAE 12.3251 HU; validation 55.01 s; training 3830.1 s (0.3192 s/step), peak 4.17 GiB.

Paired regression minus control PSNR: mean +0.03031 dB, median +0.02663 dB, IQR [+0.01721,+0.04056], win fraction 0.989, within-L333 block-bootstrap 95% interval [+0.02624,+0.03429].
This interval describes stability within one L333 CT stack; it is not patient-population significance.
Engineering flag: **DYNAMIC_DIFFUSION_VALUE_SMALL_OR_TIED**.
Condition Reliance Probe asks what an already trained model uses at forward time. This training ablation asks whether dynamic diffusion training improves final restoration under this matched backbone/data/optimizer.
Conclusions are limited to the FoundDiff-derived RDDM, this Mayo Fast protocol, and this residual objective.
L506 and external memory were never opened; no formal baseline source or checkpoints were changed.
