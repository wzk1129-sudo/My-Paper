# ProSAC-CT APGC reconstruction boundary

Source: https://arxiv.org/html/2607.01756, section 2.2, equations (7)-(8).

This is an APGC structure-provider transfer probe on the existing frozen RDDM ControlNet. It is not a reproduction of the complete ProSAC-CT model. RFDDS, TD3, and ProSAC's residual+noise objective are absent.

The official MedDINOv3 ViT-B/16 CT-3M teacher is loaded from the pinned author checkpoint and source. The model takes only a copy of center LDCT. Main RDDM input and preprocessing remain unchanged. MedDINO receives HU clipped to [-1000,1000], normalized by the official mean 65.1084213256836 and std 178.01663208007812, repeated to three channels. The source demo uses 2048 in its example, but this experiment explicitly fixes native Mayo 512x512 to preserve the CT grid and reduce computation. Official DINOv3 produces RoPE for the observed 32x32 patch grid; it does not interpolate a learned absolute position table. The final normalized patch tokens x_norm_patchtokens already exclude CLS and four storage tokens, then reshape to [B,768,32,32]. No intermediate layers or extra CNN encoder are used.

C3: project that single AFM with separate 1x1 convolutions to 32/64/128 channels, then bilinearly resize to 256/128/64. Projection before interpolation is equivalent to interpolation before an affine 1x1 projection for these settings and saves memory.

C4: use the same AFM. For an explicit Sobel map, use sqrt(gx^2+gy^2+1e-12) with standard 3x3 Sobel X/Y kernels divided by 8. Use the 4-neighbor 3x3 Laplacian kernel [[0,1,0],[1,-4,1],[0,1,0]]. The input is center LDCT HU clipped to [-1000,1000] and scaled to [0,1]; there is no Gaussian smoothing. Resize both edge maps to each target scale. At each scale, EFM_l = Conv1x1(concat(Sobel_l,Laplacian_l)); S_l = (alpha_l * AFM_l) * (gamma_l * EFM_l), with independent learned channel vectors initialized to one.

RECONSTRUCTION_CHOICE_NOT_EXPLICIT_IN_PAPER: paper Eq.(7) actually writes F_EFM = F_Sob + F_Lap. The experiment request specifies concat plus trainable Conv1x1 instead. Therefore C4 is explicitly a ProSAC-style reconstruction, not paper-exact APGC. The paper also leaves fixed edge kernels and edge-input scaling unspecified, so these are recorded implementation choices.

The existing 2.5D Control Encoder, three structure-hint injection scales, four zero-initialized residual taps, frozen Direct main, RDDM residual loss, two-step sampler and Fast64 evaluator remain unchanged. No additional gate, FiLM, attention, or residual anatomy fusion is introduced.

