# MedDINO-PDC-Gated-BCA-V1 implementation audit

Parent: E12-BCA commit 48a78da568be3bb35ec33d9ca864ccb9b795c4c5. New arm starts from the same canonical Direct 12k EMA; it does not warm-start the trained E12-BCA 6k weights. PiDiNet source: https://github.com/hellozhuo/pidinet, commit d21aa881ed9c628571636fad39acfe1fad517ebd. See PIDINET_UPSTREAM_AUDIT.json for file hashes and license. Eight forward/gradient cases matched official PDC ops exactly.

## Actual forward path

1. Diffusion calls inherited BCAUnet with [x_t, center_LDCT]. DirectUnet.neighbors supplies same-patient prev/next LDCT. The U-Net init convolution actually receives [x_t, prev, center, next] (4 channels); q-sample and NDCT target use only the center pair.
2. Unchanged E12 anatomy: center LDCT -> frozen MedDINOv3 layers 2/5/8/11 -> SALF -> hierarchical reassembly -> H128 [B,64,128,128] and H64 [B,128,64,64]. H256 is still computed by the inherited provider but unused by BCA.
3. PDC shape input is center LDCT only. From diffusion input [-1,1], HU=(center+1)*2048-1024; clip [-1000,1000] HU; divide by 1000. No RGB normalization, neighbor target, NDCT, segmentation or external CT enters PDC.
4. PiDiNet-Tiny carv4, base width 20: x1 [B,20,512,512], x2 [B,40,256,256], x3 [B,80,128,128], x4 [B,80,64,64]. The adapted official PDC Conv2d, cd/ad/rd/cv, depthwise PDC -> ReLU -> pointwise residual PDCBlock and MaxPool2d stride transition are in pdc.py. CT init is one channel and random. Natural-image weights, edge classifier/side heads, CDCM, CSAM and official training pipeline are excluded.
5. x3 projects 80->64; x4 projects 80->128 by 1x1 Conv + GELU. At each scale anatomy Conv1x1(H->1) concatenates with projected shape; GroupNorm(1,C+1) -> Conv1x1 -> GELU -> Conv1x1 -> Sigmoid produces G. GroupNorm replaces unstable ordinary BatchNorm for batch 2.
6. shape_gated=projected_shape*(1+G). S128=H128+beta128*shape_gated128; S64=H64+beta64*shape_gated64. Each beta initializes to 0.1.
7. Unchanged E12-BCA inserts after encoder downsample i=1 and i=2. Q=RDDM encoder, K/V=fused S128/S64. Windows 8/16, 4 heads, dim_head 32, PreNorm, scaled_dot_product_attention, linear output, alpha init 0.1, residual addition. No mid BCA, dropout or attention ZeroConv.

## Trainability and observed activity

Frozen: MedDINO backbone. Trainable: SALF, active hierarchy, PDC, both shape projections, both gates, betas, BCA128/64 and alphas, RDDM main. Total 99,737,237 parameters; trainable 14,095,253; frozen 85,641,984. Added versus E12-BCA: 94,182 = PDC 72,780 + projections 15,552 + gates 5,848 + two betas. All 13 tracked trainable gradient groups had nonzero finite gradients in every one of 6000 steps. MedDINO gradients stayed None.

Last Full-L333 sample with step6000 EMA: at 128, anatomy/raw-PDC/projected/gated/weighted/fused RMS = 0.983927/0.108225/0.043397/0.062684/0.005817/0.984219; gate mean/std/spatial std = 0.444169/0.026108/0.026108; beta=0.092798; alpha=0.201746; BCA weighted residual RMS=0.092194. At 64, the corresponding RMS = 0.643304/0.082843/0.032486/0.048839/0.004380/0.643406; gate mean/std/spatial std = 0.503276/0.016914/0.016914; beta=0.089676; alpha=0.164211; BCA weighted residual RMS=0.049811. These are last-sample diagnostics, not volume-level gate distributions. They show that PDC/gates/BCA are active.

## Outcome and limits

Fast64@6000 EMA PSNR 40.846740 dB, delta to matched E12-BCA -0.001802 dB. Full L333 610-slice EMA: PSNR 40.854364 dB, SSIM 0.948517, RMSE 18.229297 HU, MAE 12.039938 HU. Relative to E12-BCA, delta PSNR -0.001690 dB, paired slice bootstrap 95% CI [-0.002117,-0.001270] dB, positive fraction 0.373770. This one-patient slice CI is not patient-level significance. PDC, gating and fusion were introduced together; their individual effects are not isolated. The active combined path did not improve E12-BCA.

Original EXPERIMENT_CONFIG.json is untouched. CONFIG_DISCREPANCY_AUDIT.json clarifies its two-channel diffusion API field versus actual four-channel U-Net init and inherited generic K/V label; runtime K/V is fused S128/S64. L506, external memory, neighbor NDCT target and blocked attempts were all zero. No medical images or checkpoints are archived to Git.
