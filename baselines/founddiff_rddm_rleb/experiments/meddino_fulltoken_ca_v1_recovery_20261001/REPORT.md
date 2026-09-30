# MedDINO full-token global cross-attention matched continuation

Direct 12k EMA is the common initialization. Control and candidate use fresh matched Adam/EMA.
The only candidate feature is frozen final-layer MedDINO [1024,768] dense patch tokens.
Independent global CA at encoder post-downsample 128 and 64; zero-initialized scalar gates.
No Sobel, Laplacian, Wavelet, SALF, HAF, PDC, ControlNet, structure loss or external memory.

| Extra step | Control PSNR | Candidate PSNR | Delta PSNR | Positive /64 |
|---:|---:|---:|---:|---:|
| 0 | 40.767627 | 40.767627 | +0.000000 | 0 |
| 1000 | 40.777333 | 40.777813 | +0.000479 | 39 |
| 2000 | 40.792144 | 40.788759 | -0.003385 | 9 |
| 4000 | 40.809092 | 40.806493 | -0.002599 | 27 |
| 6000 | 40.818550 | 40.817322 | -0.001229 | 30 |

Decision: STOP_DELTA_6K_NONPOSITIVE

L506, external memory and neighbor NDCT were not opened by the guarded data pipeline.

Recovery note: Control resumed from an intact step-2000 checkpoint into a new output root. The original partial run remains unchanged. First-2000 file-open audit was not persisted; the guarded data loader and train manifest were verified.
