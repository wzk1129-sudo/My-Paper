# MedDINOv3 four-arm RDDM ControlNet study

Fast64 EMA evaluation only. The archived C3 final-layer arm was not retrained.
All arms use the same trained Direct 2.5D RDDM; no performance-based early stopping.

| Arm | 500 | 1k | 2k | 4k | 6k PSNR | 6k SSIM | 6k RMSE HU | 6k MAE HU | 6k soft PSNR | 6k soft SSIM | Δ vs C3 | Δ vs D1 | Positive slices vs D1 | Trainable params | Peak VRAM GiB | Sec/step |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| D1_MULTI_512 | 40.776075 | 40.786610 | 40.797164 | 40.806988 | 40.813156 | 0.948272 | 18.3154 | 12.0776 | 28.500742 | 0.852532 | +0.010955 | +0.000000 | NA | 6776736 | 4.878 | 0.290 |
| D2_MULTI_ADD_512 | 40.775563 | 40.785313 | 40.795532 | 40.804686 | 40.810844 | 0.948241 | 18.3200 | 12.0812 | 28.499364 | 0.852351 | +0.008643 | -0.002312 | 0.344 | 6777632 | 4.988 | 0.292 |
| D3_MULTI_1024 | 40.775378 | 40.785971 | 40.798994 | 40.810688 | 40.817221 | 0.948301 | 18.3065 | 12.0763 | 28.504287 | 0.852798 | +0.015019 | +0.004065 | 0.844 | 6776736 | 4.948 | 0.520 |
| D4_MULTI_FT_512 | 40.776075 | 40.786610 | 40.797164 | 40.805830 | 40.812100 | 0.948264 | 18.3175 | 12.0776 | 28.499515 | 0.852533 | +0.009899 | -0.001056 | 0.297 | 35125152 | 5.379 | 0.357 |

D4 500/1k values are the shared D1 pre-finetune history, not independent D4 training.
C3 is ARCHIVED_REFERENCE; its 2k/4k/6k PSNR is 2000: 40.788209, 4000: 40.797496, 6000: 40.802201.

## Slice-level 6k comparisons

- D1_MULTI_512_vs_C3_FINAL: mean +0.010955, median +0.010686, positive 0.969; p10/p25/p50/p75/p90 = +0.002614/+0.005676/+0.010686/+0.015191/+0.020194.
- D2_MULTI_ADD_512_vs_D1_MULTI_512: mean -0.002312, median -0.001914, positive 0.344; p10/p25/p50/p75/p90 = -0.007722/-0.005444/-0.001914/+0.001326/+0.003073.
- D3_MULTI_1024_vs_D1_MULTI_512: mean +0.004065, median +0.005425, positive 0.844; p10/p25/p50/p75/p90 = -0.003500/+0.000878/+0.005425/+0.007023/+0.010365.
- D4_MULTI_FT_512_vs_D1_MULTI_512: mean -0.001056, median -0.000748, positive 0.297; p10/p25/p50/p75/p90 = -0.003221/-0.002041/-0.000748/+0.000162/+0.000872.

## Four scientific questions

- Partial fine-tuning vs frozen at 6k: D4-D1 = -0.001056 dB; 4k delta = -0.001158 dB.
- Positive Fast64 slices for D4 vs D1: 0.297; not a majority.
- 1024 resolution vs 512 frozen at 6k: D3-D1 = +0.004065 dB; D4-D3 = -0.005121 dB.
- Next-step priority under Fast64: resolution has the stronger signal (D3-D1 +0.004065 dB, 0.844 positive slices) than partial fine-tuning (D4-D1 -0.001056 dB, 0.297 positive slices), although D3 training costs 1.79x seconds/step. The absolute gain is small and needs full-L333 confirmation before any formal claim; no next experiment is started.

## D4 matched metrics vs D1

- 4000: PSNR 40.805830 (-0.001158 dB), SSIM 0.948192 (-0.000021), RMSE 18.3303 (+0.0023 HU), MAE 12.0855 (+0.0028 HU), soft PSNR 28.493564 (-0.001590 dB), soft SSIM 0.852401 (-0.000036).
- 6000: PSNR 40.812100 (-0.001056 dB), SSIM 0.948264 (-0.000009), RMSE 18.3175 (+0.0021 HU), MAE 12.0776 (+0.0000 HU), soft PSNR 28.499515 (-0.001227 dB), soft SSIM 0.852533 (+0.000001).

## Scope and interpretation

The prior D4 attempt failed before any optimizer step because audit state_hash converted a BFloat16 tensor directly to NumPy. The retry casts audit tensors to float32 first. This instrumentation fix did not change model, checkpoint, loss, optimizer, AMP or data protocol.
D4 training reached step 6000 before a report-only configuration error. Feature diagnostics now load L333-Fast64 through the canonical frozen_config(); no optimizer step was repeated.
Matched Fast64 curves, feature activity, VRAM, and training time are archived in CSV/JSON.
D4 blocks 0–7 hash unchanged; blocks 8–11 changed with finite nonzero gradients.
No Full-L333, 8-patient formal run, L506, external memory, or neighbor NDCT access occurred.
These observations apply to this implementation and Fast64 protocol; the report does not establish a final paper method.
