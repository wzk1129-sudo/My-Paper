# Train–sampling consistency diagnosis

This is inference-only on the existing 4k and 8k EMA checkpoints of experiment/rddm-res-noise-beta-time-v1. Each path uses the same fixed L333-Fast64 slice indices, per-slice seed (20260829 + index), Gaussian draw, weights, frozen evaluator and 2-step schedule. The formal model source and checkpoints were not changed. The only intervention is in this diagnostic wrapper.

## Finding

The dominant cause of the single-digit joint PSNR is the mismatch between training and sampling noise scales. At t=999 the actual training schedule has B_999=0.999979794, while the current conditional sampler starts from LDCT+sqrt(0.01)*z = LDCT+0.1*z. The effective epsilon standard deviation measured over Fast64 is 0.1000 at t=999 and 0.1050 at t=499. Training used standard normal epsilon with target standard deviation 1. Adding the missing noise term to the intermediate update alone does not repair the result. Using B_999*z and the original RDDM two-term update restores 8k PSNR from 7.9103 to 38.8837 dB, although that remains 1.6009 dB below the matched residual-only Fast baseline (40.4846 dB).

The learned noise branch is usable on a scale-matched state: at 8k/t=499 its MSE against effective noise falls from 0.423429 to 0.000059 and Pearson correlation rises from 0.993521 to 0.999973. The former learned-noise standard deviation near 0.75 is therefore not evidence of an isolated intrinsic scale-learning failure. It was measured on a sampling state whose effective noise standard deviation was only about 0.105.

## Actual schedule and code

From the instantiated ResidualDiffusion tensors, with T=1000, sampling_timesteps=2, eta=0 and sum_scale=0.01:

| t | A_t | B_t | abs(A_t-1) |
|---:|---:|---:|---:|
| 999 | 0.9936471581 | 0.9999797940 | 0.0063528419 |
| 499 | 0.7196658850 | 0.9599024653 | 0.2803341150 |

B_999/sqrt(sum_scale)=9.99979794; sqrt(sum_scale)/B_999=0.10000202; sqrt(sum_scale)/B_499=0.10417725. The training q_sample is x_t=x0+A_t*(LDCT-NDCT)+B_t*epsilon. The current sampler initialization is LDCT+0.1*z. The absolute A_999 discrepancy is recorded; the GT-free FULL_CONSISTENT_JOINT initialization LDCT+B_999*z omits (A_999-1)*r because r is unknown at inference. The separately labeled oracle path supplies that term from validation GT only for a mathematical check.

The current FoundDiff-derived eta=0 intermediate update subtracts delta_A*pred_res and omits delta_B*pred_noise. The original RDDM image-restoration sampler subtracts both. Its conditional initialization also uses sqrt(sum_scale)*z, so the initialization issue is not unique to FoundDiff-derived code. Exact source locations and the official URL are in SAMPLER_CODE_AUDIT.md.

## Counterfactual Fast64 results

All metric columns are the original per-slice metric implementation averaged over 64 L333 slices. CURRENT_JOINT and RESIDUAL_ONLY reproduce the previously archived controls within 1e-5 on every metric. The oracle path is not a deployable method or a performance result.

| Checkpoint | Path | PSNR dB | SSIM | RMSE HU | MAE HU |
|---:|---|---:|---:|---:|---:|
| 4k | CURRENT_JOINT | 9.581861 | 0.028313 | 663.6666 | 481.4207 |
| 4k | RESIDUAL_ONLY | 38.908921 | 0.923741 | 22.7580 | 15.0253 |
| 4k | RDDM_UPDATE_ONLY | 9.728545 | 0.029498 | 652.5578 | 472.2737 |
| 4k | FULL_CONSISTENT_JOINT | 35.678978 | 0.871450 | 32.9724 | 22.3570 |
| 4k | ORACLE_INIT_DIAGNOSTIC_ONLY | 35.679277 | 0.871457 | 32.9712 | 22.3557 |
| 8k | CURRENT_JOINT | 7.910282 | 0.015799 | 804.4845 | 592.7572 |
| 8k | RESIDUAL_ONLY | 39.729321 | 0.935574 | 20.7197 | 13.5724 |
| 8k | RDDM_UPDATE_ONLY | 7.817940 | 0.015291 | 813.0833 | 598.6153 |
| 8k | FULL_CONSISTENT_JOINT | 38.883658 | 0.927580 | 22.8237 | 15.5044 |
| 8k | ORACLE_INIT_DIAGNOSTIC_ONLY | 38.884035 | 0.927585 | 22.8227 | 15.5032 |

At 4k, update-only gains 0.146684 dB and full-consistent gains 26.097116 dB over current joint. At 8k, update-only changes by -0.092342 dB and full-consistent gains 30.973376 dB. The oracle-vs-GT-free difference is only 0.000377 dB at 8k, so omitting the unavailable (A_999-1)*r term has little effect in this specific validation set.

## Trajectory evidence

Numbers below are means of per-slice spatial statistics across 64 slices. The full 1280-row table includes both timesteps, all five paths, both checkpoints, A_t/B_t, delta_A/delta_B, noise MSE/correlation, residual MSE and current x0 PSNR.

| Checkpoint/path/t | pred_noise std | epsilon_eff std | noise MSE | noise corr | x0 PSNR |
|---|---:|---:|---:|---:|---:|
| 4k current/999 | 0.553887 | 0.099999 | 0.207459 | 0.989796 | 9.540099 |
| 4k current/499 | 0.574930 | 0.104856 | 0.222080 | 0.993370 | 9.581861 |
| 4k full/999 | 0.998622 | 0.999965 | 0.000222 | 0.999893 | 35.563636 |
| 4k full/499 | 0.996093 | 1.000020 | 0.000241 | 0.999891 | 35.678978 |
| 8k current/999 | 0.734065 | 0.099999 | 0.404294 | 0.989762 | 7.838553 |
| 8k current/499 | 0.754434 | 0.105004 | 0.423429 | 0.993521 | 7.910282 |
| 8k full/999 | 0.998066 | 0.999965 | 0.000054 | 0.999976 | 38.719420 |
| 8k full/499 | 1.000958 | 1.000064 | 0.000059 | 0.999973 | 38.883658 |

Here epsilon_eff=(x_t-NDCT-A_t*(LDCT-NDCT))/B_t is computed with GT only after prediction for diagnosis; it is never model input. The full-consistent 8k joint result is below the 8k residual-only reconstruction (39.729321 dB) and below the matched Fast baseline (40.484568 dB). Thus sampler repair explains the catastrophic failure but does not establish a useful restoration gain from joint residual/noise training. Per the experiment's stop criteria, no further residual+noise training or parameter sweep is recommended.

## Integrity

Both checkpoints matched their archived SHA256 values and were loaded with EMA state_dict, with no optimizer step. The formal source, evaluator and L333-fast64 split hashes remained unchanged. The data-access guard recorded only L333, all 64 frozen indices, zero L506 opens, zero external-memory opens and zero blocked attempts. No Full-L333, training, lambda sweep or other experiment ran. See FINAL_DIAGNOSIS.json and DATA_ACCESS_AUDIT.json for machine-readable provenance and access counts.
