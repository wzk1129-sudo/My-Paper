# Official RDDM 2.5D + MedDINO hierarchical edge V2

This branch adds the V2 experiment to the already published matched MedDINO baseline branch, experiment/rddm-official-2p5d-meddino-condition-v1. The inherited `experiments/rddm_official_2p5d_meddino_condition_v1/` directory supplies the unchanged official restoration RDDM backbone, 2.5D context loader, MedDINO loader, diffusion/sampler, normalization, and matched evaluator. The new `experiments/rddm_2p5d_meddino_hier_edge_v2/` directory contains V2 and its measured results.

## Implementation
- Frozen MedDINOv3 ViT-B/16 layers [2,5,8,11]. Each 768-channel map passes through the existing Conv1x1 768→64 adapter. Layers 2/5 make `A_local`; 8/11 make `A_semantic`; each 128-channel field remains at 32×32.
- Center LDCT alone yields four fixed edge maps: Gaussian sigma 0.8, 1.6, 3.2 followed by Scharr magnitude, plus sigma 0.8 followed by signed Laplacian. Each map receives per-sample spatial RMS normalization (epsilon 1e-6), Conv3 1→32 + GN8 + SiLU. Concatenation and Conv3 128→128 + GN8 + SiLU produce full 512×512 edge feature `E`. Separate Conv1x1 128→128 + GN8 heads after 32×32 resize produce `E_local` and `E_semantic`.
- Exact multiplicative sources: `S_local = 0.1 * (A_local * E_local)`; `S_semantic = 0.1 * (A_semantic * E_semantic)`; `S_mid = 0.5*S_local + 0.5*S_semantic`. No learned lambda or anatomy residual fusion.
- Stages E0/E1/D2/D3 use local; E2 uses mid; E3/MID/D0/D1 use semantic. Every stage bilinearly resizes its 128-channel source to the feature resolution with `align_corners=False`. The existing FiLM form remains `F + gamma(S)*GN(F) + beta(S)`, GN8 affine=False and zero final projection. Former global Linear adapters apply independently at each spatial position to preserve full spatial information.
- Original restoration topology is preserved: 19 ResnetBlocks, 8 LinearAttention blocks, bottleneck full Attention and both skips per stage. Backbone forward AST and all 9 attention input links passed readiness. Activation recomputation applies only to the new edge/FiLM trainable functions to fit B=2; it preserves RNG.

## Matched protocol and private dependencies
- Both V2 and the 40.908237 matched baseline started from the same official RDDM 2D 8k EMA checkpoint (SHA256 2cc1d6aeb785bd0b7a5786801e3fa9e0e3ba4427bf3a43c86fc894f3486ad669), with fresh Adam and fresh EMA. Shared initial model hash, initial RNG hash, train/validation manifests and all 8000 augmented tensor/timestep/noise stream digests matched. Sanity 200-step weights were discarded.
- Train L067/L143; EMA Fast64 validation L333; seed20260829; B=2, accumulation1; Adam lr2e-4, betas(0.9,0.99), weight decay1e-9; EMA0.995; grad clip1; AMP=False; CUBLAS_WORKSPACE_CONFIG=:4096:8. MSE pred_res, T=1000, sampling_timesteps=2, sum_scale0.01; no sampler or loss change.
- Normalize Mayo stored pixels as HU=stored-1024, clip[-1024,3072], map to[0,1]. Input prev/center/next LDCT, center NDCT supervision only. HU metric window[-1000,1000], PSNR range2000; skimage SSIM win7, non-Gaussian, sample covarianceTrue, K1=.01, K2=.03.
- This publication contains source, configuration, metrics and hashes. Private Mayo arrays, parent/EMA/MedDINO checkpoints, full prediction arrays and CT visualizations are excluded. `model.py` in the inherited baseline uses original absolute paths to those private inputs; replace paths only in a separate reproduction environment and verify the hashes in the manifests. A reader can audit the exact implementation and results from this branch without those files.

## L333-fast64 EMA result
All rows use the same 64 centers and evaluator. Deltas in the validation CSV compare each V2 node against the fixed matched baseline 8k.
| Step | PSNR dB | SSIM | RMSE HU | MAE HU | Delta PSNR vs baseline 8k |
|---:|---:|---:|---:|---:|---:|
| 0 | 40.623375 | 0.946056 | 18.719123 | 12.301865 | -0.284863 |
| 2000 | 40.765716 | 0.947535 | 18.415240 | 12.127258 | -0.142521 |
| 4000 | 40.855088 | 0.948397 | 18.228122 | 12.023777 | -0.053150 |
| 6000 | 40.881759 | 0.948590 | 18.173293 | 12.002924 | -0.026478 |
| 8000 | 40.899279 | 0.948758 | 18.137594 | 11.976010 | -0.008959 |
| Matched baseline @8k | 40.908237 | 0.948847 | 18.119234 | 11.986737 | 0 |

Predeclared outcome: **STOP**. V2 6k and 8k PSNR deltas are non-positive; at 8k ΔPSNR=-0.008959 dB and ΔSSIM=-0.000089. Edge-region RMSE is higher by +0.026289 HU; paired baseline-minus-V2 95% slice-bootstrap CI [-0.04197300598025322, -0.010507071018218994]. No 10k/12k/Full-L333 extension.

## Final-only L506 evaluation
Only after saving the V2@8k EMA and completing its original Fast64/region report, a separate read-only pass inferred all 524 valid L506 centers. The model received original normalized inputs. Both HU windows were scored post-hoc from the **same saved float32 predictions**; no L506 metric selected a checkpoint or altered training.
| Model and L506 evaluation window | PSNR dB | SSIM | RMSE HU | MAE HU |
|---|---:|---:|---:|---:|
| V2@8k [-1000,1000] | 41.621091 | 0.955856 | 16.879333 | 11.158063 |
| Official CoreDiff 150k EMA unified [-1000,1000] | 41.609371 | 0.955808 | 16.915365 | 11.236898 |
| V2@8k [-160,240] | 29.637760 | 0.883079 | 13.501280 | 6.355840 |
| Official CoreDiff 150k EMA [-160,240] | 29.601209 | 0.887514 | 13.566654 | 6.396306 |

V2 minus CoreDiff main-window PSNR: +0.011720 dB; soft-window PSNR: +0.036551 dB, soft SSIM: -0.004435. This is a numeric comparison between different models/training/sampling protocols, not a causal architectural attribution. The CoreDiff main-window row uses its **unified** protocol; its separate official dynamic-range SSIM is not substituted.

## Audit and source navigation
- `model_v2.py`: V2 fixed kernels, feature hierarchy, multiplicative fusion, spatial FiLM and unchanged RDDM forward. `readiness_v2.py`: B2 real-batch topology/gradient/zero-init audits.
- `train_v2.py`, `runs/2P5D_MEDDINO_HIER_EDGE_V2_8K/EFFECTIVE_CONFIG.json`, `TRAIN_LOG_100STEP.csv`: exact training loop, effective options and light loss/gradient/time trace. Inherited `model.py`, `context_data.py`, `med_loader.py`, `evaluate.py`, `src/official_rddm_ct.py` in the preceding package contain the reused core.
- `finalize_v2.py`, `REGION_SUMMARY.json`, `PAIRED_REGION_DELTAS.csv`, `CONDITION_STRENGTH.csv`: exact 8k checkpoint reload, fixed center-LDCT edge mask and matched region results.
- `evaluate_l506_final_v2.py`, `L506_FINAL_EVAL_8K/EVALUATION_CONFIG.json`, both L506 per-slice CSVs and `L506_FINAL_EVAL_8K/SUMMARY.json`: final-only one-pass evaluation and comparison. Saved prediction file SHA256 is fe0654656ea8a5e0f00390c8cf2c4810ec1b315c2c637222605f5d801517a0db; the private 549 MB NPY file is excluded. V2 EMA checkpoint SHA256 is c8c291a8b78f87f6809565ea00808ff28ff8e3182f9a3206c6acfc4ab7a38684.
- `READINESS.json`, `VISUAL_REVIEW.json`, `SANITY_SUMMARY.json`, `SUMMARY.json`, `TRAINING_SUMMARY.json`, `L506_FINAL_EVAL_8K/DATA_ACCESS_AUDIT.json`, source/run manifests and checkpoint reload audits document frozen weights, actual data opens, parameters, VRAM, stream matching and failure gates.
- Full method/metric discussion: `REPORT.md` and `L506_FINAL_EVAL_8K/REPORT.md`. Per-slice Fast64 metrics and exact node activity are in selected `runs/2P5D_MEDDINO_HIER_EDGE_V2_8K/` CSV/JSON files. The full training JSONL remains only on the private remote machine; SHA256 e78dc5c8e3aab21abc78393d4bd6eb618baba58c3e7e65eddcbfdda987093a8b.

## Publication boundary
GitHub branch: `experiment/rddm-2p5d-meddino-hier-edge-v2` in private `wzk1129-sudo/My-Paper`. Its base is the previously published matched MedDINO branch. Actual experiment source was the separate CoreDiff worktree branch with base commit `1c258e6706a1d0fe23c8319a3ffad18738594278`, before publication. Source hashes in the audit files tie the published texts to that execution. No CT images, DICOM, NPY, checkpoints, visualizations or large feature caches are in this commit.
