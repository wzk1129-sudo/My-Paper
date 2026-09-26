# MedDINO Full-L333 Validation V1

Inference-only L333 development validation: all 610 ordered slices, EMA step-6000 checkpoints, two-step RDDM sampler with sum_scale=0.01 and slice-wise seed 20260829+index.
Wide-window [-1000,1000] HU is primary; [-160,240] HU is **SECONDARY_SOFT_TISSUE_METRIC** on the same predictions.

| Model | PSNR | SSIM | RMSE HU | MAE HU | Soft PSNR | Soft SSIM | Soft RMSE HU | Positive slices vs reference | Mean wall sec/slice | Peak VRAM GiB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| C3_FINAL | 40.810725 | 0.948292 | 18.3178 | 12.0756 | 28.496345 | 0.852282 | 15.1685 | NA | 0.221 | 0.953 |
| D1_MULTI_512 | 40.821165 | 0.948359 | 18.2968 | 12.0647 | 28.505206 | 0.852421 | 15.1537 | 0.956 | 0.165 | 0.955 |
| D3_MULTI_1024 | 40.825761 | 0.948390 | 18.2868 | 12.0631 | 28.509422 | 0.852686 | 15.1459 | 0.851 | 0.385 | 0.955 |

| Comparison | Mean Delta PSNR | Median | Slice-bootstrap 95% CI | Positive slices | Mean Delta Soft PSNR |
|---|---:|---:|---:|---:|---:|
| C3_D1 | +0.010439 | +0.009463 | [+0.009851, +0.011037] | 0.956 | +0.008861 |
| D1_D3 | +0.004597 | +0.005468 | [+0.004178, +0.005024] | 0.851 | +0.004216 |
| C3_D3 | +0.015036 | +0.014366 | [+0.014472, +0.015610] | 0.993 | +0.013077 |

## Paired interpretation

- D1 Multi-512 vs C3 Final: stable positive on this L333 volume = True. All-three-thirds positive = True.
- D3 Multi-1024 vs D1 Multi-512: stable positive on this L333 volume = True. All-three-thirds positive = True.
- D3/D1 inference wall-time ratio = 2.34x; peak VRAM ratio = 1.00x. The observed PSNR gain and compute cost should both determine whether D3 remains a candidate.

## Slice-position thirds

- C3_D1 first_third (indices 0-203, n=204): +0.007834 dB.
- C3_D1 middle_third (indices 204-406, n=203): +0.009774 dB.
- C3_D1 last_third (indices 407-609, n=203): +0.013723 dB.
- D1_D3 first_third (indices 0-203, n=204): +0.005154 dB.
- D1_D3 middle_third (indices 204-406, n=203): +0.006929 dB.
- D1_D3 last_third (indices 407-609, n=203): +0.001704 dB.

## Candidate priority

C3 Final remains the archived final-layer comparator.
D1 Multi-512 is the primary candidate for any later formal 8-patient test: its full-L333 paired gain over C3 is +0.010439 dB with 0.956 positive slices.
D3 Multi-1024 is a conditional compute-cost candidate, not the first formal-training choice: it adds only +0.004597 dB over D1 at 2.34x inference wall time.
Peak VRAM here means peak CUDA allocated memory measured by PyTorch, not process-level nvidia-smi reservation.

## Scope and safety

The 10,000 paired-bootstrap draws resample L333 slices only. Their intervals are not cross-patient generalization intervals.
No training, optimizer step, external memory, neighbor NDCT, or L506 access occurred. Model weights, checkpoints, formal source, and frozen evaluator were unchanged.
This is development validation, not a test-set or final paper result. No formal 8-patient training was started.
