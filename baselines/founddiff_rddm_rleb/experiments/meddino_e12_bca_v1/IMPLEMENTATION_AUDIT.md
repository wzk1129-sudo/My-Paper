# E12-BCA implementation audit

This describes the code that produced the completed Fast6k and Full-L333 run. The experiment branch is `experiment/rddm-meddino-e12-bca-v1`; the parent commit is `b9e31c586f88b3796173190f0ed5f38f698f2f5c`. The formal diffusion source `src/rddm_rleb.py` was not edited.

## Actual structure path

```text
center LDCT in the conditional diffusion input
  -> SALFProvider.forward: HU reconstruction/windowing and MedDINO input preparation
  -> frozen MedDINOv3 intermediate blocks 2/5/8/11 (four 768-channel maps)
  -> per-scale SALF learned softmax scores over four layers
  -> weighted concatenation, 1x1 projection, spatial interpolation
  -> A64 [B,128,64,64], A128 [B,64,128,128], A256 [B,32,256,256]
  -> HierarchicalSALFProvider: H64=A64; H128=A128+zero-initialized reassembly(H64);
     H256=A256+zero-initialized reassembly(H128)
  -> BCA consumes S128=H128 and S64=H64
```

The actual committed E12-HAF extractor has **no Sobel/Laplacian/Edge branch**. Its 256-scale reassembly runs as part of the unchanged E12 provider topology, but BCA does not consume S256; consequently its final `up128_256` path has no direct loss path in BCA (source-level inference). E12-HAF's separate local/cross attention is inside its Control Encoder and is not instantiated here. SALF and hierarchical modules are freshly initialized for this independent arm from the same canonical Direct state used by other Fast6k arms; no trained D1/E12-HAF 6k checkpoint was used as a warm start.

## BCA definition and exact location

`meddino_e12_bca_v1/model.py` defines `BCAUnet.forward`. It retains the Direct Concat diffusion input `[x_t, LDCT(z-1), LDCT(z), LDCT(z+1)]` and the original time MLP, down blocks, mid block, up blocks, and skip count/order. Each encoder iteration first executes its original residual block, saves that block's skip, and executes its original downsample. **Immediately after downsample i=1** it applies BCA128 to the 128×128 feature; **immediately after downsample i=2** it applies BCA64 to the 64×64 feature. The next original encoder block then receives the modified feature. The skip captured in that same iteration precedes BCA, so BCA affects the subsequent encoder path rather than rewriting that saved skip.

| Tap | Backbone H input | E12 S input | Window | Heads × dim/head | Q | K and V | Output |
|---|---|---|---:|---:|---|---|---|
| post-downsample i=1 | [B,128,128,128] | [B,64,128,128] | 8×8 | 4×32 | H | S128 | [B,128,128,128] |
| post-downsample i=2 | [B,256,64,64] | [B,128,64,64] | 16×16 | 4×32 | H | S64 | [B,256,64,64] |

The shapes above were measured with B=2 and recorded in `STRUCTURE_AUDIT.json`. In each aligned, non-shuffled spatial window, separate PreNorm and learned linear projections form Q from H and K/V from S. Scaled dot-product cross-attention uses no dropout; a normally initialized output linear map returns to the backbone channel count. Each stage has its own trainable scalar alpha initialized to 0.1:

```text
H_out = H + alpha_stage * output_projection(cross_attention(Q(H), K(S), V(S)))
```

There is no attention ZeroConv. The original RDDM timestep, residual-diffusion formulation/objective, conditional restoration, loss, and 2-step sampler remain unchanged.

The inherited config retains `training.validation_every=2000` and `recovery_checkpoint_every=1000`. The actual BCA training script uses its explicit `STEPS=(0,500,1000,2000,4000,6000)` for EMA validation and saves resumable checkpoints at 2000/4000/6000. The source and `VALIDATION_HISTORY.csv` are authoritative for this run schedule; the inherited interval fields do not fully encode it.

## Frozen and trainable modules

| Module | This BCA run |
|---|---|
| MedDINOv3 feature extractor | Frozen, eval mode; no optimizer gradients |
| E12 SALF scorers and projections | Trainable, fresh initialization |
| E12 hierarchical reassembly | Trainable topology; H256 output is unused by BCA |
| E12-HAF Control Encoder / ControlNet | Absent |
| RDDM U-Net main modules | Trainable |
| BCA Q/K/V/output projections and alpha128/alpha64 | Trainable |

The smoke and formal gradient audits record nonzero gradients in both BCA stages, both alphas, the backbone, and the active structure path. The 6k EMA alphas are 0.2021737844 and 0.1617987007. Full-L333 last-slice weighted attention residual RMS values are 0.0926482677 at 128 and 0.0487138703 at 64; these demonstrate a nonzero conditioning path but are not a causal ablation by themselves.

## Functional differences from controls

- **Versus D1_MULTI_512:** D1 uses the original multi-layer MedDINO provider and a ControlNet-style injection with a frozen RDDM main. BCA uses SALF plus E12 hierarchy, removes the Control Encoder, inserts two direct local cross-attention taps into the RDDM encoder, and trains the RDDM main.
- **Versus E12-HAF:** the E12 feature-provider topology is shared. E12-HAF sends its features through Control Encoder local/cross fusion and residual injection while freezing the RDDM main. BCA omits that Control Encoder, uses RDDM backbone features as Q and E12 features as K/V at 128/64, and trains the RDDM main. E12-HAF's 256-scale Control tap is absent.
- **Shared:** fixed Fast6k Mayo patients/split, seed, conditional residual diffusion, optimizer, batch, EMA, sampling, and L333 evaluation protocol.

D1 and E12-HAF freeze the RDDM backbone, whereas E12-BCA trains it. The observed Full-L333 PSNR gains of +0.034890 dB against D1 and +0.036915 dB against E12-HAF therefore **cannot be attributed to Cross-Attention alone**. They measure this complete BCA implementation and training-parameter choice.
