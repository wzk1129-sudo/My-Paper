# Residual+noise beta-time Fast-dev V1

One shared RLEB U-Net is called twice on the same input: alpha-time for residual, beta-time for noise. Physical batch 2, accumulation 1. The 100-step lambda calibration is separate from fresh 8k training.

Calibration: lambda_noise=0.004944047, lambda_recon=0.004796126. Parameters: 13,131,010.

| Step | Joint PSNR | Delta vs matched residual baseline | SSIM | RMSE HU | MAE HU |
|---:|---:|---:|---:|---:|---:|
| 2000 | 9.476718 | -29.001982 | 0.023386 | 671.810 | 486.852 |
| 4000 | 9.581861 | -30.563525 | 0.028313 | 663.667 | 481.421 |
| 6000 | 8.414259 | -31.972944 | 0.019380 | 759.133 | 555.361 |
| 8000 | 7.910282 | -32.574286 | 0.015799 | 804.484 | 592.757 |

| Step | Joint PSNR | Residual-only PSNR | Residual-only delta vs matched baseline | Learned/analytic noise MSE at t=499 | Pearson r | Learned/analytic std at t=499 |
|---:|---:|---:|---:|---:|---:|---:|
| 4000 | 9.581861 | 38.908921 | -1.236464 | 0.222824 | 0.994363 | 0.5749/0.1039 |
| 8000 | 7.910282 | 39.729321 | -0.755247 | 0.424465 | 0.994491 | 0.7544/0.1041 |

At 8k, learned noise mean at t=499 is -0.0256 versus analytic 0.0001; the std remains strongly inflated. Paired sampling latent states are identical between learned and residual-only paths.

Stop decision: 8k joint PSNR is below the matched baseline, learned noise remains mismatched, and residual-only reconstruction remains below the matched baseline. End the residual+noise direction; no Full-L333, lambda sweep, or additional noise-head experiment.

No optimizer step after 8k, no checkpoint/source/split/evaluator modification. L506/external-memory/blocked opens are zero. Raw images and checkpoints remain local and excluded from this lightweight archive.

Mean optimizer-step time 0.493 s; peak allocated VRAM 7.56 GiB; largest preclip gradient norm 46.226.
