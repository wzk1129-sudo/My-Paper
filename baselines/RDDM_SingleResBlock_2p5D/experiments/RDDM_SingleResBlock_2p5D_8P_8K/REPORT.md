# RDDM SingleResBlock 2.5D 8P 8K
Completed 8000 optimizer steps and four fixed EMA validations.

## Provenance and architecture
Official repository: https://github.com/nachifur/RDDM
Official commit: 524caea34cedb3733aa5bf6c27dc1cd6e0469216
Official source: experiments/2_Image_Restoration_deraing_raindrop_noise1/src/residual_denoising_diffusion_pytorch.py
Snapshot SHA256: b7c41d56313a5073a88c85cedd9ae2b65167515f6294f42288bba52831ea5c3c
Project commit at creation: adc8f138e1e02a1ac567177d6508436c2b5c6f36
Official building blocks are copied verbatim and AST-verified against the immutable snapshot.
Original single U-Net: 19 full ResBlocks (8 encoder + 2 bottleneck + 8 decoder + 1 final), 8 LinearAttention, 1 Global Attention, 8 encoder-decoder skips, 1 stem skip.
New single U-Net: 10 full ResBlocks (4+1+4+1), 8 LinearAttention, 1 Global Attention, 4 encoder-decoder skips, 1 stem skip.
Total/trainable parameters: 20505217/20505217.
Input: [I_t, LDCT_prev, LDCT_center, LDCT_next], output: center residual [B,1,512,512].
Only the new experiment directory was created. No pretrained weights, extra losses, or extra modules.
Each complete ResBlock retains its internal block1 and block2 convolution subblocks; there is no second ResBlock per scale.
MLSF can subsequently be integrated at explicit encoder/decoder interfaces, but none is present in this experiment.

## Protocol
Stored values -> float32 HU = stored - 1024; clip [-1024,3072]; u=(HU+1024)/4096; diffusion maps u to 2u-1 once.
Linear beta: 0.0001 to 0.02 over 1000 steps; P=cumprod(1-beta); A=1-sqrt(P); B=sqrt(1-P), float32 buffers.
Training: R=Y-X0; X_t=X0+A_t R+B_t epsilon; only residual MSE. t uniform integer [0,999].
Time input always A_t*1000 to native time MLP.
Sampling: X999=Y+0.1 epsilon; clip predicted residual to [-1,1]; X499=X999-(A999-A499)R999; final clamp(Y-R499,-1,1). Model times 999,499. No predicted-noise update.
Training B999=0.9999797940254211; inference initial standard deviation=0.1. This mismatch is retained as requested.
Adam lr=2e-4, betas=(0.9,0.99), weight_decay=1e-9; batch=2, accumulation=1, FP32, AMP/TF32 off; gradient clip=1; no scheduler.
EMA decay=.995 every optimizer step, zero warmup. Seed=20260829. Augmentation uses isolated seed=10 RNG, paired horizontal flip p=.5 and uniform quarter rotations. No crop.
Training patients: ['L067', 'L096', 'L109', 'L143', 'L192', 'L286', 'L291', 'L310']; 4800 pairs. Validation L333 Fast64, exact saved order, batch=1, isolated CUDA noise generator seed=20260829+slice_index.
L506 accesses: 0 (allowlisted enumeration and reads only).
Checkpoint contains model/EMA/optimizer, all requested RNG states, shuffle permutation and cursor, configuration and source metadata.
Environment: {"python": "3.11.16 (main, Sep  2 2026, 23:39:52) [GCC 15.3.0]", "torch": "2.8.0+cu128", "cuda": "12.8", "numpy": "2.3.2", "einops": "0.8.1", "skimage": "0.25.2", "gpu": "NVIDIA GeForce RTX 3090", "tf32": false, "amp": false, "cpu_threads": 2}

## Metrics
Each slice is independently windowed and evaluated, then 64 slice metrics are arithmetically averaged.
Main window [-1000,1000], data_range=2000; soft window [-160,240], data_range=400.
skimage SSIM win_size=7, gaussian_weights=False, use_sample_covariance=True, channel_axis=None, K1=.01,K2=.03.
| Step | PSNR | SSIM | RMSE HU | MAE HU | Soft PSNR | Soft SSIM | Soft RMSE HU | Soft MAE HU |
|---|---|---|---|---|---|---|---|---|
| 2000 | 38.677645 | 0.928052 | 23.391685 | 15.793290 | 26.906709 | 0.832618 | 18.175574 | 8.994741 |
| 4000 | 40.280271 | 0.943224 | 19.460985 | 12.849770 | 28.067635 | 0.836018 | 15.918234 | 7.845166 |
| 6000 | 40.486846 | 0.945056 | 19.003522 | 12.571548 | 28.212228 | 0.839484 | 15.658637 | 7.721877 |
| 8000 | 40.655352 | 0.946525 | 18.644329 | 12.303540 | 28.354164 | 0.839725 | 15.410262 | 7.609621 |

Best EMA checkpoint: /home/xujian/research/My-Paper/baselines/RDDM_SingleResBlock_2p5D/experiments/RDDM_SingleResBlock_2p5D_8P_8K/checkpoints/step_8000.pt (ema_state_dict).
Training-step time: 4450.05 seconds; total including saves/validation: 4535.47 seconds.
Peak allocated GPU memory: 12.755 GiB; peak reserved: 14.086 GiB.
Reference supplied by user: FoundDiff TimeOnly 40.846046 dB; difference of best score: -0.190694 dB.
That historical score is not independently re-evaluated here: differences in architecture, initialization, training schedule, fixed evaluation noise or preprocessing limit causal comparison.
This is one seed and one validation patient; validation checkpoint selection is not a held-out test result.
No additional ablations or noise settings were tested. Inspect loss trend and soft-tissue scores before adding MLSF/detail modules.
