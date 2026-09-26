# E12-guided Backbone Cross-Attention (E12-BCA) V1

This isolated Fast6k experiment tests whether the RDDM encoder can use E12 structure features directly. The canonical Direct Concat 12k EMA weights initialize the main RDDM trunk, exactly as for D1/E12-HAF Fast6k arms. No D1/E12-HAF 6k trained weights initialize this arm.

Actual E12-HAF source uses frozen MedDINO layers 2/5/8/11, SALF, and zero-initialized hierarchical reassembly. It has **no Sobel/Laplacian/Edge branch**. E12-HAF's cross-attention is within its Control Encoder; this experiment removes that ControlNet path and inserts new window cross-attention into the RDDM encoder.

At 128², the post-downsample RDDM feature has 128 channels and E12 structure has 64 channels; 8×8 aligned windows use four 32-channel heads. At 64², channels are 256 and 128; aligned windows are 16×16. The backbone feature is Q, E12 structure is K/V. Separate trainable alphas begin at 0.1; output projections use ordinary initialization. The original U-Net blocks, timestep, diffusion formulation, residual objective, neighboring LDCT context, skip connections, and 2-step sampler are unchanged.

Fast training uses L067/L143, batch 2, Adam 2e-4, AMP off, EMA .995, seed 20260829, 6000 optimizer steps, and L333 Fast64 at 0/500/1000/2000/4000/6000. Full evaluation uses step6000 EMA on all 610 L333 slices with the frozen evaluator. L506 and external-memory patients are forbidden.

**Comparison limitation:** D1/E12-HAF freeze the RDDM main modules, whereas the present request requires the BCA RDDM main to train. Hence an observed delta against those arms reflects both injection architecture and a trainable-main difference. It cannot isolate the attention mechanism alone.

The 20-step smoke is separate from formal training. The formal train starts anew from the canonical initialization. Results, checkpoints, and logs are written only under this experiment's new output directory. Git archives lightweight source/config/metrics/audits/report; checkpoints, images, data, and large logs stay remote and untracked.
