# RDDM Fast Screen V1

FoundDiff RDDM+RLEB backbone adapted to our fixed Mayo-2016 protocol. This is a Fast-dev method screen, not a final paper result. All eight candidates started independently from seed 20260829 and trained on L067/L143. EMA validation used the same frozen 64 L333 indices at 2k/4k/6k/8k, the same HU metrics, 2-step sampler and sum_scale=0.01. No external memory or L506 image was accessed.

## Completion and baseline

All eight 50-step smoke checks passed, and all eight separate runs completed 8000 optimizer steps. No non-finite loss, gradient or validation metric was found. Queue runtime: 6:13:20.158456.

| Step | Baseline PSNR | SSIM | RMSE HU | MAE HU |
|---:|---:|---:|---:|---:|
| 2000 | 38.478700 | 0.926035 | 23.9292 | 15.7929 |
| 4000 | 40.145386 | 0.941559 | 19.7640 | 13.0245 |
| 6000 | 40.387202 | 0.944026 | 19.2251 | 12.5807 |
| 8000 | 40.484568 | 0.944869 | 19.0149 | 12.4597 |

## Full Fast64 checkpoint curve

Each cell is PSNR dB (Delta PSNR versus the fixed Fast baseline at the same step).

| Candidate | 2k | 4k | 6k | 8k |
|---|---:|---:|---:|---:|
| F1_noise_050 | 39.1511 (+0.6724) | 40.2859 (+0.1405) | 40.4343 (+0.0471) | 40.4820 (-0.0025) |
| F1_noise_025 | 38.9252 (+0.4465) | 40.0978 (-0.0476) | 40.3661 (-0.0211) | 40.4636 (-0.0209) |
| F1_noise_000 | 17.9096 (-20.5691) | 13.2874 (-26.8579) | 12.5730 (-27.8142) | 12.6902 (-27.7944) |
| F2_multiscale_ldct | 38.3392 (-0.1395) | 40.2096 (+0.0642) | 40.3999 (+0.0127) | 40.4840 (-0.0006) |
| F3_rleb2conv | 36.9299 (-1.5488) | 39.1635 (-0.9819) | 40.1050 (-0.2822) | 40.3182 (-0.1663) |
| F4_window_010 | 38.2957 (-0.1830) | 40.1388 (-0.0066) | 40.3712 (-0.0160) | 40.4608 (-0.0238) |
| F4_window_025 | 38.6937 (+0.2150) | 40.1857 (+0.0404) | 40.3828 (-0.0045) | 40.4452 (-0.0394) |
| F4_window_050 | 38.7858 (+0.3071) | 40.1939 (+0.0485) | 40.3957 (+0.0085) | 40.4890 (+0.0044) |

The per-candidate `validation_metrics.csv` files and `FAST_SCREEN_V1_LONG.csv` retain every checkpoint's SSIM, RMSE/MAE HU, validation loss, training loss, pre-clipping gradient norm, learning rate, parameter count, peak VRAM, seconds/step and validation time.

## 8k secondary metrics and cost

| Candidate | SSIM | RMSE HU | MAE HU | Parameters | Peak VRAM GiB | Seconds/step |
|---|---:|---:|---:|---:|---:|---:|
| F1_noise_050 | 0.944797 | 19.0200 | 12.4224 | 13,130,945 | 4.163 | 0.309 |
| F1_noise_025 | 0.944649 | 19.0583 | 12.4731 | 13,130,945 | 4.163 | 0.306 |
| F1_noise_000 | 0.071454 | 464.1183 | 326.5214 | 13,130,945 | 4.163 | 0.317 |
| F2_multiscale_ldct | 0.944939 | 19.0164 | 12.4321 | 13,628,165 | 4.710 | 0.351 |
| F3_rleb2conv | 0.943284 | 19.3758 | 12.7097 | 19,477,697 | 5.419 | 0.407 |
| F4_window_010 | 0.944616 | 19.0688 | 12.5155 | 13,130,945 | 4.163 | 0.312 |
| F4_window_025 | 0.944205 | 19.1035 | 12.5913 | 13,130,945 | 4.163 | 0.311 |
| F4_window_050 | 0.944972 | 19.0070 | 12.4662 | 13,130,945 | 4.163 | 0.309 |
| Fast baseline | 0.944869 | 19.0149 | 12.4597 | 13,130,945 | 4.163 | 0.313 |

## Observations

- At 8k, F1a (−0.0025 dB), F2 (−0.0006 dB) and F4c (+0.0044 dB) are numerically close to baseline (within 0.01 dB). This is a descriptive range, not a statistical equivalence test.
- F1b (−0.0209 dB), F4a (−0.0238 dB) and F4b (−0.0394 dB) are lower at 8k. F3 is lower by 0.1663 dB; F1c collapses to 12.6902 dB (−27.7944 dB).
- Only F4c is positive at both 6k (+0.0085 dB) and 8k (+0.0044 dB). No candidate reaches the +0.05 dB 8k promotion heuristic. F4c improves 8k SSIM and RMSE slightly, while MAE is slightly worse than baseline.
- F1a is ahead at 2k/4k/6k but not at 8k. A smaller training noise amplitude did not show a stable 8k gain. F1c finished normally with finite loss, gradient and metrics; its collapse is a performance failure, not a numerical or runtime failure.
- F2 zero-gated multiscale LDCT has active learned branches: gamma levels start at 0 and end (raw) at +0.398101, +0.024579, −0.001180 and −0.003062; optimizer moments are nonzero and all 16 condition-conv/projection tensors changed from the seed reconstruction. Feature magnitudes were not recorded. The optimizer evidence rules out an entirely untrained branch, but the branch's output contribution magnitude was not recorded. Near-baseline PSNR does not prove raw LDCT is redundant; optimization and the 8k budget may also matter.
- F3 trails baseline at every checkpoint, but its Delta PSNR narrows from −1.5488 at 2k to −0.1663 at 8k. Under the fixed 8k screen it did not win; longer-horizon convergence cannot be inferred from these four points. F3 uses 19.48M parameters and 5.419 GiB peak VRAM versus 13.13M and 4.163 GiB for baseline.
- F4 8k Delta PSNR is −0.0238 (lambda 0.1), −0.0394 (0.25), +0.0044 (0.5): there is no monotonic degradation with larger window-loss weight in the tested range, nor a gain meeting the promotion threshold. F4 training loss includes an added window term, so its numeric loss should not be directly compared with residual-only candidates.
- The screen does not support a claim that any tested change materially improves the backbone under this Fast-dev protocol. It does not select a final paper backbone. No further experiment was launched.

## Integrity and scope

Formal source SHA256: `e8ed93223fbdf4a0e2521f73c7a4f213561f6a191a1a7b2a5f95934b85ebf574`. The protected formal 60k checkpoint, baseline result files, Fast protocol, three split files and all tracked baseline files retained their audited hashes. Each candidate source and config SHA256 is in `provenance.json` and its training summary. All patient/image access reports show optimization reads only from L067/L143 and fixed L333-fast64 validation; L506 opens = 0, external-memory opens = 0, blocked attempts = 0. No full-L333 or test evaluation was performed in this queue.

Exact runtime source and configs are archived under `fast_dev_variants/` and `experiments/fast_dev/candidates/<job>/CONFIG.json`; a second copy of each config and its full checkpoint-level results is under this study directory. Original aggregate files are preserved as `FAST_SCREEN_V1_ORIGINAL.json` and `FAST_SCREEN_V1_ORIGINAL_REPORT.md`. No checkpoint, medical image or large training log is included.
