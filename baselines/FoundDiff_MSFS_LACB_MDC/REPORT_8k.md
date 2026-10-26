# FoundDiff-MSFS-LACB-MDC

Completed 8000 optimizer steps and all four EMA L333-Fast64 validations from random seed10 initialization. No old diffusion weights loaded. MedDINO frozen pretrained weights only.

|Step|Baseline PSNR|Candidate PSNR|Delta PSNR|SSIM|RMSE HU|MAE HU|
|---:|---:|---:|---:|---:|---:|---:|
|2000|40.26604638|39.86870050|-0.39734588|0.93901455|20.41044092|13.77905004|
|4000|40.69278263|40.74317238|0.05038975|0.94763056|18.46375516|12.53946275|
|6000|40.79711561|40.85308429|0.05596868|0.94839645|18.23207669|12.33213024|
|8000|40.84604585|40.89735173|0.05130588|0.94873618|18.14114612|12.29543646|

|Step|Soft PSNR|Soft SSIM|Soft RMSE HU|
|---:|---:|---:|---:|
|2000|27.72870879|0.83483356|16.55546969|
|4000|28.44453110|0.84200331|15.25659444|
|6000|28.52690681|0.84288617|15.11464556|
|8000|28.55993071|0.84313972|15.05932867|

Candidate8k versus TimeDACB8k: PSNR +0.05130588 dB; SSIM +0.00042401; RMSE -0.10630146 HU; MAE -0.14282177 HU. Exceeds TimeDACB@8k PSNR: True. Best candidate EMA step=8000, PSNR=40.89735173.

Total parameters 110,587,073; trainable 24,945,089; frozen 85,641,984. Peak allocated 8.003 GiB, peak reserved 10.275 GiB. Training+validation runtime 4.529 hours. Runtime does not include preflight/final audit.

## Implementation
model.py implements MSFS, ScaleFusion, FAM, LACB, LinearAttention, MDC and ConditionUnet.forward. src/DADiff.py is byte-identical to the baseline; all original RDDM math, 10 single-convolution ResnetBlocks, down/up/skip/head modules retained. Input [I_t,previous,center,next]; output single-channel center residual. Both prior branches see center LDCT only; MedDINO uses model center*1000 HU, then (HU-65.1084213256836)/178.01663208007812 repeated3.

MSFS: frozen MedDINO zero-based blocks2/5/8/11, four patch grids768@32; CNN center-only 3x3/GN8/SiLU stages64@512,64@256,128@128,256@64. Each CNN pooled32 queries corresponding Med layer with projected KV, SDPA4heads x32dim. Attention128@32 upsampled, concatenated with native CNN and fused by3x3/GN8/SiLU. Four FAM3x3 concatenate same-scale64/64/128/256 U-Net and structure, before encoder ResBlock.

Global condition M11 GAP768 -> Linear256; concatenate original time_mlp256 -> Linear512/256 -> SiLU -> Linear256. One joint vector feeds all nine independent LACBs.

LACB: CPE1 depthwise3x3 -> non-affine LN eps1e-6 + condition shift/scale -> ACL in_proj/dwc/SiLU/ELU+1 linear attention with LePE -> multiply act_proj/SiLU branch -> out_proj/residual -> CPE2/residual -> LN/FFN(C,2C,C GELU)/residual = H. Output x+gate*(H-x); exactly one shift/scale/gate projection, zero initialized. Four LA heads at each scale; ACL norm2 remains default LayerNorm. No additional gates, stochastic depth, Mamba or SS2D. LA token matrix is never NxN; KTV normalization matches pinned ACL code.

MDC: parallel depthwise3x3 dilation2 padding2 and5x5 dilation2 padding4, each GN8+SiLU, concatenate ->1x1 -> input residual. Six locations E0/E1/E2 and D1/D2/D3. Encoder FAM -> original ResBlock -> LACB -> optional MDC -> identical tensor to skip and downsample. Mid original ResBlock -> LACB. Decoder concat skip -> original ResBlock -> LACB -> optional MDC -> upsample. E3/MID/D0 are64x64; E3 ends same-resolution3x3, not an extra downsample. Decoder channel sequence512/256/128/64; original stem skip/final ResBlock/1x1 head unchanged.

## Protocol and validation
config.json inherits the original eight-patient training and fixed Fast64 order; seed10, batch2 x accumulation2, 512, FP32/TF32off, Adam2e-4 betas(.9,.99) wd0, clip1, EMA.995 every step, residual MSE only, diffusion1000/sample2/sumscale.01, no augmentation or L506. frozen_metrics.py unchanged. Per-slice means: main[-1000,1000], PSNRrange2000, SSIMwin7; soft[-160,240]. Only new trainable modules use activation checkpointing; resolution/batch/precision unchanged.

Preflight status PASS; ACL LA forward max difference=0.0, input gradients equal=True. Nine LACBs, six MDCs, four FAMs and full dataflow checked. Time and global conditions influence updated outputs; MSFS/LACB/MDC finite gradients and optimizer updates checked. Frozen MedDINO SHA and eval/no-grad verified. Realbatch2 training and two-step sampling passed. No hidden auxiliary losses or extra experiments.

## Interpretation and limitations
This compares the complete candidate architecture with the baseline at matched step counts; it cannot identify an individual module's contribution. A single seed and one validation patient's correlated64 slices do not establish statistical significance or generalization. Assess both main and soft windows and the saved per-slice errors before selecting a new architecture. Next useful direction: inspect where errors concentrate and whether high-resolution structure fusion preserves fine detail; no further variant or search was run.

## Artifacts and reproduction
Best local EMA checkpoint: /home/xujian/projects/FoundDiff_Lite_TimeDACB/experiments/FoundDiff_MSFS_LACB_MDC_8k/checkpoints/step_8000.pt, key ema_state_dict. Frozen MedDINO weights are not duplicated in checkpoints and must match med_loader.py SHA.
Run in a fresh copy of this directory: CUDA_VISIBLE_DEVICES=0 conda run --no-capture-output -n flexict python preflight.py; then bash run.sh. Existing train logs/checkpoints are protected. Configure external data/model paths in config.json and med_loader.py for another machine. Exact launch/environment/source records are in RUN_MANIFEST.json and INITIAL_STATE.json.

ACL reference: https://github.com/ClimBin/ACLNet/blob/905f6d04530d8b6ca4b5a962ef59fd5f3ea8c6b1/models/layers.py. Baseline: https://github.com/wzk1129-sudo/My-Paper/tree/adc8f138e1e02a1ac567177d6508436c2b5c6f36/baselines/FoundDiff_Lite_TimeDACB.
