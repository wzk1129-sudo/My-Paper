# ProSAC-CT APGC Fast Reproduction: M0 vs M1 (loss 1:1:1)

Completed independent 8k runs of M0_LOSS111 and M1_APGC_LOSS111. The only functional arm difference is APGC. The loss was changed identically in both arms to MSE residual + MSE noise + MSE reconstruction. Equal 1:1:1 weights are a reproduction choice; the paper did not publish these values.

## Fast64 EMA validation

| Step | M0 PSNR | M1 PSNR | Delta PSNR | M0 SSIM | M1 SSIM | Delta SSIM | Delta RMSE HU | Delta MAE HU | Positive slices |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2000 | 25.958314 | 25.969402 | +0.011089 | 0.789502 | 0.792761 | +0.003259 | -0.0275 | -0.3597 | 47/64 |
| 4000 | 26.088585 | 26.103781 | +0.015195 | 0.793665 | 0.793501 | -0.000164 | -0.0358 | -0.0612 | 41/64 |
| 8000 | 26.285884 | 26.181275 | -0.104609 | 0.798129 | 0.795994 | -0.002134 | +0.2291 | +0.1629 | 0/64 |

| Step | M0 RMSE HU | M1 RMSE HU | M0 MAE HU | M1 MAE HU | M0 edge RMSE HU | M1 edge RMSE HU | M0 edge MAE HU | M1 edge MAE HU |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2000 | 20.2541 | 20.2265 | 10.8134 | 10.4537 | 32.5708 | 32.5692 | 24.9368 | 24.9374 |
| 4000 | 19.9736 | 19.9378 | 10.2795 | 10.2183 | 32.1691 | 32.0333 | 24.4018 | 24.3391 |
| 8000 | 19.5324 | 19.7615 | 9.8833 | 10.0463 | 31.2462 | 31.6092 | 23.5856 | 23.9719 |

At 2k and 4k, APGC gains were small (+0.0111 and +0.0152 dB). At 8k it was worse by 0.1046 dB, with 0/64 slices improved. This is not a stable or obvious APGC gain under this reproduction choice; no extension or parameter sweep was started.

## Training and diagnostics

| Arm | Step | Median loss_res | Median loss_noise | Median loss_recon | Median loss_total |
|---|---:|---:|---:|---:|---:|
| M0_LOSS111 | 2000 | 0.00744 | 0.02169 | 0.00720 | 0.03582 |
| M0_LOSS111 | 4000 | 0.00678 | 0.03300 | 0.00693 | 0.04674 |
| M0_LOSS111 | 8000 | 0.00643 | 0.01683 | 0.00643 | 0.03296 |
| M1_APGC_LOSS111 | 2000 | 0.00723 | 0.01977 | 0.00730 | 0.03558 |
| M1_APGC_LOSS111 | 4000 | 0.00690 | 0.03302 | 0.00690 | 0.04590 |
| M1_APGC_LOSS111 | 8000 | 0.00658 | 0.01779 | 0.00666 | 0.03417 |

The loss table contains medians over the 100 optimizer steps ending at each validation node; it is not validation loss.
M0 peak allocated VRAM 22.010 GiB and median step time 0.631 s; M1 21.504 GiB and 0.671 s.
At M1 step 8k: alpha mean/std 1.069348/0.053473; gamma mean/std 1.069348/0.053473; AEFM mean/std -0.215312/0.880073.
M0 had 6 finite pre-clip gradient spikes >100 (maximum 1770.3, near step 2k); M1 had 0. Both used the frozen gradient clip and finished with finite losses and metrics. The spike is a comparability caveat.

## Protocol and integrity

Eight frozen training patients; 3-slice LDCT context and center NDCT target; L333 frozen Fast64 (64 slices); HU clip [-160, 240] -> [0,1] and RDDM internal domain; 512x512; Adam lr 2e-4; physical batch 4; T=1000; sum_scale=1; frozen 10-step sampler and evaluator.
Smoke shared-backbone initial SHA256: d429070140ac394c654af01945a49485b2139882a3d1d8c3ad0e283f5a709d4a. Full 8k training stream hashes match at every step (final 90cdf493d87a04c4aa4074a76d0148289b951ebca6092b45e7340dcfa0f6ee0d). Source/config hashes match the launch record; checkpoints exist at 2k/4k/8k for both arms. EMA swap-in and restoration were checked in the validation source.
All recorded validation rows identify L333 and match the frozen Fast64 indices. The dataset constructors use only the configured eight training patients and L333 validation; no external-memory path is used. Pre-run audit recorded zero forbidden-test opens. There was no OS-level file-open tracing during training, so zero L506 opens cannot be independently certified from runtime instrumentation.
No training beyond 8k, Full-L333 evaluation, parameter sweep, or new experiment was performed.

Wall time: 3.04 hours from formal launch to completion. See M0_VS_M1_RESULTS.csv and FINAL_AUDIT_LOSS111.json for exact values and provenance.
