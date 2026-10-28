# RDDM SingleResBlock 2.5D — Mayo 8 patients / 8000 steps

A single-U-Net, residual-only CT baseline built from the official RDDM image-restoration implementation. All results below are measured with EMA weights on L333 Fast64; these are validation results, not held-out test results.

## Architecture
- Input: [I_t, LDCT_previous, LDCT_center, LDCT_next], 4 x 512 x 512.
- Output: one center-slice residual; reconstruction is center LDCT minus predicted residual.
- Native RDDM dim=64, dim_mults=(1,2,4,8).
- 10 complete two-convolution ResBlocks, 8 LinearAttention, 1 Global Attention.
- Four encoder-decoder skips plus one stem skip.
- 20,505,217 total/trainable parameters.
- No MedDINO, MLSF, SS2D, DACB, frequency module, or auxiliary loss.
- Official single U-Net before modification has 19 full ResBlocks and eight encoder-decoder skips.
- Official blocks are verbatim extracted and AST-verified. The immutable snapshot and provenance are in src/official_rddm_original.py and SOURCE.json.

## Verified results
Main HU window [-1000,1000]; per-image metrics averaged over the fixed 64 slices.

| Step | PSNR dB | SSIM | RMSE HU | MAE HU |
|---|---:|---:|---:|---:|
| 2000 | 38.677645 | 0.928052 | 23.391685 | 15.793290 |
| 4000 | 40.280271 | 0.943224 | 19.460985 | 12.849770 |
| 6000 | 40.486846 | 0.945056 | 19.003522 | 12.571548 |
| 8000 | **40.655352** | **0.946525** | **18.644329** | **12.303540** |

At 8000 steps, soft window [-160,240]: PSNR 28.354164 dB, SSIM 0.839725, RMSE 15.410262 HU, MAE 7.609621 HU.
Training-step time 74.17 min; program time including saves and validation 75.59 min. Peak allocated GPU memory 12.755 GiB, reserved 14.086 GiB.
Completed exactly 8000 optimizer steps and 8000 EMA updates; process exit code 0.
The supplied historical FoundDiff TimeOnly reference is 40.846046 dB (difference -0.190694 dB); protocols were not independently matched, so this is contextual rather than a controlled architecture comparison.

## Data and numerical protocol
Data is not distributed. Obtain authorized Mayo 2016 paired NPY files, each 512x512:
PATIENT_25_INDEX_img.npy and PATIENT_target_INDEX_img.npy, using zero-padded slice indices.
Training patients: L067, L096, L109, L143, L192, L286, L291, L310 (4800 pairs).
Validation: L333, fixed order in splits/L333_fast64.txt. L506 is excluded by the loader allowlist.
Each patient boundary duplicates the center LDCT; no neighboring NDCT is used.
Stored values are converted to HU by subtracting 1024, clipped to [-1024,3072], then mapped once to [0,1]. Diffusion maps these to [-1,1] exactly once.
Paired horizontal flip p=.5 and uniform 0/90/180/270 rotation use a separate RNG seed 10. No crop.

For beta=linspace(.0001,.02,1000), P=cumprod(1-beta), A=1-sqrt(P), B=sqrt(1-P):
R=Y-X0, X_t=X0+A_t*R+B_t*epsilon, loss=MSE(pred_R,R).
Time conditioning is float32 A_t*1000, not the integer t.
Sampling starts at Y+0.1*epsilon. Predict/clamp residual at 999, update X499=X999-(A999-A499)*R999, then predict/clamp residual at 499 and return clamp(Y-R499,-1,1).
Training B999 is approximately 1, while inference initial noise std is .1; this requested mismatch is retained.
Validation uses a separate CUDA generator with seed 20260829+slice_index and batch=1.
SSIM: skimage, win_size=7, gaussian_weights=False, use_sample_covariance=True, K1=.01, K2=.03. Metrics are computed independently per slice before averaging.

## Reproduction
Recorded environment: Python 3.11.16, PyTorch 2.8.0+cu128, NumPy 2.3.2, einops 0.8.1, scikit-image 0.25.2, RTX 3090 24GB. No package upgrades were performed.
The training implementation imports only its extracted official blocks; importing the complete upstream snapshot is unnecessary and would require unrelated upstream dependencies.

This directory is a completed-run archive: commands intentionally refuse to overwrite existing artifacts.
For a fresh run, make a NEW experiment directory outside this archive containing the Python files, src/, splits/, and config.json. Set config.json data_root to your authorized data location and give name a new identifier. Create experiments/<name>/checkpoints, logs, and reports before running:
1. python -B -u preflight.py
2. python -B -u train.py
Use the available founddiff environment and a remote tmux session for long training. Original exact server invocation is preserved in RUN_INSTRUCTIONS_ORIGINAL.md and the run manifest. run.sh is the original machine-specific launcher, not a portable launcher.
Common setup fixes CPU threads=2, disables AMP/TF32/cudnn benchmark, and sets CUBLAS_WORKSPACE_CONFIG=:4096:8.
Adam lr=.0002, betas=(.9,.99), weight_decay=1e-9; batch=2, accumulation=1, clip norm=1, EMA=.995 every step with no warmup, no LR scheduler, seed=20260829.
Explicit resume: python -B -u train.py --resume /absolute/checkpoint.pt. Model, optimizer, EMA, all RNG states, shuffle permutation and cursor are restored. Never duplicate a live run.
Known recovery limitation: interrupted validation may leave an existing per-slice CSV; inspect it before resuming because output files are exclusively created and are not silently overwritten.

## Artifacts
See [full report](experiments/RDDM_SingleResBlock_2p5D_8P_8K/REPORT.md), [metrics](experiments/RDDM_SingleResBlock_2p5D_8P_8K/validation_summary.csv), and [configuration](config.json).
The experiment directory includes four per-slice CSVs, all 8000 training log records, preflight evidence, run manifest, start/completion metadata and console logs.
The first preflight failed due to filename-regex transmission escaping, which was fixed before training; the successful retry and failure log are both retained. A scalar-conversion warning in training logging did not interrupt execution.
Full checkpoints remain on the research server; CHECKPOINT_INFO.json records the best checkpoint path, SHA256 and size. No image pixels, weight tensors or raw datasets are included.
MLSF/detail modules may be evaluated in a separate future experiment; no such ablation was run here.

## Upstream attribution
Official RDDM: https://github.com/nachifur/RDDM
Pinned commit: 524caea34cedb3733aa5bf6c27dc1cd6e0469216.
Liu et al., Residual Denoising Diffusion Models, CVPR 2024, pp. 2773–2783.
The upstream snapshot retains its original contents. No new license is asserted over upstream code.
