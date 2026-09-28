# Sampler code audit

Scope: inference-only comparison on the existing 4k/8k EMA checkpoints. No baseline source or checkpoint is edited.

## Current FoundDiff-derived implementation

Source: src/rddm_rleb.py, SHA256 e8ed93223fbdf4a0e2521f73c7a4f213561f6a191a1a7b2a5f95934b85ebf574.

- Lines 451-459: conditional sum_scale=0.01 forces ddim_sampling_eta=0; convert_to_ddim=True and the 1000-step linear beta schedule is used.
- Lines 477-480 and 582-594: A_t=1-sqrt(alpha_cumprod_t) and B_t=sqrt(1-alpha_cumprod_t) are stored as alphas_cumsum and betas_cumsum.
- Lines 867-873: training uses q_sample=x0+A_t*r+B_t*epsilon, with r=LDCT-NDCT and epsilon~N(0,I).
- Lines 765-784: two-step sampling uses (999,499),(499,-1) and initializes img=LDCT+sqrt(sum_scale)*z, where z~N(0,I).
- Lines 642-665: the joint prediction calls the shared network with alpha-time and beta-time, clips pred_res to [-1,1], and forms/clips x0=img-A_t*pred_res-B_t*pred_noise.
- Lines 808-830: at eta=0, sigma2=0; the active use_pred_noise branch updates the intermediate state as img_next=img-(A_t-A_next)*pred_res. The pred_noise term is absent in this branch.
- Lines 830-840: an alternate use_x_start branch is present but inactive.

## Original RDDM image-restoration implementation

Official source: https://github.com/nachifur/RDDM/blob/main/experiments/2_Image_Restoration_deraing_raindrop_noise1/src/residual_denoising_diffusion_pytorch.py

In its ddim_sample, use_pred_noise at eta=0 subtracts both (A_t-A_next)*pred_res and (B_t-B_next)*pred_noise. Its explicit special_eta_0 branch has the same two-term update. Its conditional initialization also uses LDCT+sqrt(sum_scale)*z; therefore the initialization-scale issue is shared by this particular official code path and is not a FoundDiff-only divergence. The actual schedule tensor values are recorded in SCHEDULE_AUDIT.json.

The diagnostic wrapper changes only inference state updates and/or initialization. ORACLE_INIT_DIAGNOSTIC_ONLY uses validation NDCT solely to construct an exact initial q-sample for a mathematical upper-bound check; its result is never a deployable method or performance claim.
