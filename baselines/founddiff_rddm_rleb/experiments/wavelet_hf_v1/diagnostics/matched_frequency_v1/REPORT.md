# Matched frequency diagnosis: Wavelet HF V1 vs 2P5D Direct Concat

This is inference only. Existing 2k/4k/6k/8k EMA checkpoints were read without training or modification. Both arms used the same L333-fast64 slice indices, original corediff environment (Python 3.8.13, PyTorch 1.10.1+cu111, CUDA 11.1), fixed per-slice validation seed, two-step sampler, sum_scale 0.01, HU preprocessing and [-1000,1000] HU evaluator. The original Haar implementation SHA256 was verified as 5f2e8446512bbdea89e164c566ba9741384b62cc742220900be709991ff7c737.

For each slice, the residual is the **last sampling-step clipped pred_res**, and the target is 2*(center LDCT - center NDCT) in the diffusion [-1,1] domain. This inference-time measure differs from Wavelet V1's earlier frequency_metrics.csv, which measured training-loss predictions at a sampled diffusion timestep. The newly recomputed PSNR/SSIM/RMSE/MAE matched both archived validation histories within 1e-5 at every node before frequency results were interpreted.

| Step | Direct mean HF MSE | Wavelet mean HF MSE | HF improvement | Delta LL MSE | Delta spatial residual MSE | Delta PSNR |
|---:|---:|---:|---:|---:|---:|---:|
| 2k | 0.0000825199 | 0.0000654376 | 20.7008% | -0.0000888464 | -0.0000350233 | +1.069041 dB |
| 4k | 0.0000539280 | 0.0000523619 | 2.9041% | +0.0000006927 | -0.0000010014 | +0.072875 dB |
| 6k | 0.0000509130 | 0.0000504481 | 0.9130% | +0.0000028222 | +0.0000003569 | +0.008430 dB |
| 8k | 0.0000499192 | 0.0000496441 | 0.5511% | +0.0000007690 | -0.0000000141 | -0.004856 dB |

At 8k, LH/HL/HH MSE all decreased versus Direct by 1.9887e-7, 2.8210e-7 and 3.4431e-7 respectively. Mean HF MSE decreased by 2.7509e-7 (0.5511%). LL MSE increased by 7.6896e-7 (about 0.36%); full spatial residual MSE was effectively unchanged, decreasing by 1.4080e-8 (about 0.016%). This is a very small LL/HF redistribution, not a clear overall spatial-error penalty.

On paired 8k slices, HF MSE improved in 61/64 (95.3125%), while PSNR improved in 27/64 (42.1875%). Pearson correlation between per-slice HF-error improvement and PSNR delta was +0.793716; this is an association, not a causal estimate. Mean PSNR still decreased by 0.004856 dB. The high-frequency advantage is too small at 8k to ensure a positive reconstruction result; the slight LL change and per-slice PSNR losses can offset it. The evaluator's clipped HU window and mean per-slice PSNR are also not numerically equivalent to full-range mean residual MSE.

**Final classification: CASE B.** The large early HF and PSNR gains contract to a nearly identical 8k frequency solution. The evidence supports early convergence acceleration, rather than a durable high-frequency advantage. It does not establish a meaningful frequency trade-off of CASE C, because 8k spatial residual MSE did not worsen. The current Wavelet auxiliary-loss route should stop under its unchanged 8k promotion rule. These results alone do not justify advancing to an LF/HF decoupled architecture; that would require a separate, stronger bottleneck hypothesis and new authorization.

No optimizer step, 10k/12k continuation, Full-L333 evaluation, lambda sweep or additional experiment was performed. Data-access audit: L506=0, external memory=0, neighbor NDCT=0, blocked attempts=0.
