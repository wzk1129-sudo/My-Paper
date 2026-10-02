# MedDINO-Hierarchical-Edge V2 Fast-Dev report

Conclusion: **STOP**. 8000 steps completed; no extension scheduled.

## Provenance
- Branch: experiment/rddm-2p5d-meddino-hier-edge-v2
- Base Git commit: 1c258e6706a1d0fe23c8319a3ffad18738594278; V2 implementation is an uncommitted independent worktree change (see source manifests).
- Matched baseline checkpoint: /home/xujian/projects/CoreDiff_official_2p5d_meddino_condition_v1/experiments/rddm_official_2p5d_meddino_condition_v1/runs/OFFICIAL_RDDM_2P5D_MEDDINO_CONDITION_retry1/checkpoints/step_8000.pt
- Baseline checkpoint SHA256: 1ea09e9d0a4f2d56a636a62dc4a02afe80ff6342f506628077dd0003af748ce1
- V2 checkpoint SHA256: c8c291a8b78f87f6809565ea00808ff28ff8e3182f9a3206c6acfc4ab7a38684
- Parent initialization checkpoint: /home/xujian/projects/CoreDiff_official_rddm_ct_adapt_v1/experiments/rddm_official_ct_adapt_v1/runs/candidate_8k_20261001/checkpoints/step_8000.pt
- Parent SHA256: 2cc1d6aeb785bd0b7a5786801e3fa9e0e3ba4427bf3a43c86fc894f3486ad669
- Official RDDM restoration source commit: 524caea34cedb3733aa5bf6c27dc1cd6e0469216

## Exact protocol and implementation
- Exact V2 constants: CONFIG.json; actual training protocol snapshot: runs/2P5D_MEDDINO_HIER_EDGE_V2_8K/EFFECTIVE_CONFIG.json.
- Inherited config model metadata describes the historical stripped implementation and is not used to build V2. Actual V2 topology is audited in READINESS.json: 19 ResnetBlocks, 8 LinearAttention, 1 bottleneck full Attention; unchanged original skips and diffusion AST.
- MedDINO layers [2,5,8,11] -> four 768-to-64 Conv1x1 adapters; [2,5] local and [8,11] semantic -> separate Conv1x1(128,128)+SiLU, spatial 32x32 throughout.
- Fixed center-LDCT Gaussian sigma [0.8,1.6,3.2], radius ceil(3*sigma)=[3,5,10], separable normalized kernels; reflect padding; Scharr /32 magnitude; signed four-neighbor Laplacian after sigma0.8.
- Raw maps use per-sample spatial RMS normalization with epsilon1e-6; each Conv3(1,32)+GN8+SiLU; concat -> Conv3(128,128)+GN8+SiLU at full512x512; resize32 -> independent Conv1x1(128,128)+GN8 local/semantic.
- S_local=0.1*(A_local*E_local); S_semantic=0.1*(A_semantic*E_semantic); S_mid=0.5*S_local+0.5*S_semantic. No residual anatomy fusion or learnable lambda.
- Stage map: {"E0": "S_local", "E1": "S_local", "E2": "S_mid", "E3": "S_semantic", "MID": "S_semantic", "D0": "S_semantic", "D1": "S_semantic", "D2": "S_local", "D3": "S_local"}
- Every stage resizes bilinearly with align_corners=False; original GN8 affine=False and adapter/projection dimensions preserved. Former global Linear adapters apply per spatial position; no global pooling. FiLM = F+gamma*GN(F)+beta; zero final projections.
- Activation recomputation only for edge encoder and FiLM affine functions during training, non-reentrant and preserve_rng_state=True. No backbone change, AMP, batch, or formula change.
- Train L067/L143 only; validation fixed64 L333 centers with patient-local prev/next LDCT. Center NDCT supervision only. No external memory, L506, or Full-L333.
- Stored pixel -1024 -> clip[-1024,3072] -> [0,1]; paired flip0.5 and rotations0/90/180/270. Input [x_t,prev LDCT,center LDCT,next LDCT]; output center residual.
- pred_res, original MSE/q_sample/residual/sampler; T=1000, sampling2, sum_scale0.01. Adam lr2e-4, betas(0.9,0.99), wd1e-9; B2 accum1; EMA0.995 each step; AMP=False; clip1.0; no scheduler.
- Seed20260829; CPUthreads2; CUBLAS_WORKSPACE_CONFIG=:4096:8; deterministic algorithms and cuDNN deterministic, benchmarkFalse, TF32False. Fresh optimizer/EMA for sanity and formal; both initialized from matched2D8kEMA, not prior MedDINO8k.
- Evaluation: EMA, fixed validation per-slice seed+slice_index; HUclip[-1000,1000], PSNRrange2000; skimage SSIM win7, non-Gaussian, sample covarianceTrue, K1=.01 K2=.03; per-slice HU RMSE/MAE and arithmetic means.

## Validation
| Step | PSNR | SSIM | RMSE HU | MAE HU | Delta PSNR vs baseline8k | Delta SSIM |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 40.623375 | 0.946056 | 18.719123 | 12.301865 | -0.284863 | -0.002792 |
| 2000 | 40.765716 | 0.947535 | 18.415240 | 12.127258 | -0.142521 | -0.001312 |
| 4000 | 40.855088 | 0.948397 | 18.228122 | 12.023777 | -0.053150 | -0.000450 |
| 6000 | 40.881759 | 0.948590 | 18.173293 | 12.002924 | -0.026478 | -0.000257 |
| 8000 | 40.899279 | 0.948758 | 18.137594 | 11.976010 | -0.008959 | -0.000089 |

Baseline8k: PSNR 40.908237245; SSIM 0.948847464; RMSE 18.119234214 HU; MAE 11.986736715 HU.
All node deltas compare with the fixed baseline8k above, not a same-step control. Full deltas: VALIDATION_METRICS.csv. Per-slice values: run L333_fast64_step_*_metrics.csv.

## Region diagnostics
Mask fixed before training: center LDCT sigma0.8 Gaussian+Scharr magnitude, divided by per-sample spatial RMS+1e-6; edge>=2.0, flat=complement. No target-derived mask. Both EMA8k checkpoints evaluated on exactly the same64 slices and masks. Region metrics are arithmetic means of per-slice HU-clipped metrics.
| Region | Baseline RMSE | V2 RMSE | Baseline MAE | V2 MAE |
|---|---:|---:|---:|---:|
| edge | 22.137846 | 22.164135 | 15.891828 | 15.902725 |
| flat | 17.829472 | 17.847208 | 11.743596 | 11.731520 |

Paired mean baseline-minus-V2 edge RMSE improvement: -0.026289 HU; bootstrap95%CI [-0.04197300598025322, -0.010507071018218994], seed20260829,10000 replicates. This interval describes Fast64 slice resampling, not independent patient replication.

## Condition strength at 8k
Definition: RMS(gamma*GN(F)+beta)/RMS(F), mean of128 sampler observations across64 slices. CONDITION_STRENGTH.csv retains every validation node and gamma/beta RMS.
| Stage | Delta/base | Gamma RMS | Beta RMS |
|---|---:|---:|---:|
| E0 | 0.06398792 | 0.02599022 | 0.03915858 |
| E1 | 0.10626849 | 0.05001979 | 0.06358842 |
| E2 | 0.03038833 | 0.01280747 | 0.02239144 |
| E3 | 0.00919737 | 0.00368514 | 0.00751185 |
| MID | 0.00598736 | 0.00387657 | 0.00118253 |
| D0 | 0.00536774 | 0.00395426 | 0.00264229 |
| D1 | 0.01042401 | 0.00850136 | 0.00672516 |
| D2 | 0.02970047 | 0.02420345 | 0.01509029 |
| D3 | 0.02078555 | 0.02305349 | 0.01095677 |

## Integrity and resources
- Readiness PASS: realB2 forward/backward, native attention gradients finite/nonzero, feature/edge gradients finite/nonzero after zero projection begins updating; step0 output difference 0.0.
- Visual review PASS with limitations: Laplacian has visible granular noise and coherent contours; no kernel changed. The step0 E heatmap is untrained and provides no performance claim.
- Sanity PASS200: nonfinite0; median 1.4074 sec/step; peak 17.115 GiB.
- Formal nonfinite0; all parameter audits finite; peak 17.120 GiB; mean 1.3920, median 1.4163 sec/step (training timing includes step audits, excludes node validation).
- Trainable parameters 38,131,393; total 123,773,377; baseline trainable 37,948,353; delta +183,040.
- MedDINO checkpoint file SHA256492b2b96c00e9d3b93ba3b07a8d387aa2196b5e04698378fcc23049b447966d2; loaded state before/after a614e65a580989fd83255ce74e4064af4e96863d8766afd74ce94d56b576232f / a614e65a580989fd83255ce74e4064af4e96863d8766afd74ce94d56b576232f; frozen/eval/gradNone; fixed kernels unchanged.
- All8000 per-step augmented tensors, sample metadata, timestep and noise bytes match the baseline cumulative stochastic stream; initialization and manifests also match. RANDOM_STREAM_AUDIT.json and train.jsonl retain evidence.
- Reload PASS for baseline and V2 checkpoints; reproduced whole-image metrics within1e-5. Both SHA256s recorded; checkpoint files remain private and unmodified.
- All data audits: L506=0, external_memory=0, neighbor_NDCT=0, blocked_attempts=0. Only authorized train patients and Fast64 centers/context opened.

## Decision and scope
- Decision STOP; PROMOTE requires deltaPSNR>=.03, SSIM improvement and positive lower bootstrapCI for edge RMSE. STOP requires both6k and8k deltaPSNR<=0. Other outcomes retained for diagnosis.
- No10k/12k/Full-L333 scheduled. Evidence is a single seeded two-training-patient Fast-Dev experiment, not a generalization claim.

## Artifacts and reproducibility
- Code: model_v2.py, train_v2.py, finalize_v2.py, bootstrap.py; inherited unchanged model, loader, diffusion and evaluator are tracked in the sibling baseline package.
- Launch commands/environment/output/log/time and Git state: *_LAUNCH_AUDIT.json. ENVIRONMENT.json contains actual runtime versions. Checkpoints under runs/*/checkpoints remain excluded from Git.
- Audits: READINESS.json, VISUAL_REVIEW.json, SANITY_SUMMARY.json, TRAINING_SUMMARY.json, REGION_* and *_RELOAD_AUDIT.json, source manifests, run INITIAL_STATE/RANDOM_STREAM_AUDIT/DATA_ACCESS_AUDIT/MEDDINO_FROZEN_AUDIT.
- Logs: logs/sanity_200.log, logs/formal_8k.log; per-step log runs/2P5D_MEDDINO_HIER_EDGE_V2_8K/train.jsonl.
- Diagnostic image remains private: visualizations/L333_fast64_309_edges.png. No CT data, checkpoints, or large outputs uploaded.
- No software installation, Git commit, or push performed in this task.

## Post-hoc L506 final-only evaluation

V2 step8000 EMA, full valid L506 centers, one saved prediction array, two post-hoc HU windows. See L506_FINAL_EVAL_8K/REPORT.md, SUMMARY.json and per-slice CSVs. No L506 result was used for training or checkpoint selection.
