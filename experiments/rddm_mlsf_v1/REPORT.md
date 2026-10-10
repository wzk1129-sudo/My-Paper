# MLSF-RDDM matched 8k result

## Model and audit

MLSF keeps the Official RDDM residual diffusion, native ResnetBlocks, nine native Attention sites, 2.5D stem, frozen MedDINO L2/L5/L8/L11 Adapter/Fusion, and nine V1 FiLM sites. Added LocalDetailEncoder, one aligned 3x3 L8/L11 cross-attention at 128x128, and zero-initialized gamma/beta corrections inside E1/E2/D1/D2 FiLM. No extra loss or training arm.

Shared V1 state tensors exact: 498; added trainable parameters: 336,032; original native Attention sites: 9.
Shapes: L256 [B,32,256,256], L128/F128 [B,64,128,128], M32 [B,64,32,32], F256 [B,32,256,256]. Each 128x128 query attends to exactly 9 clamped 32x32 MedDINO tokens; no global attention matrix.
The four corrected FiLM sites compute the original gamma/beta and the spatial delta in one FiLM formula before native Attention. Other five sites use V1 unchanged.

Step 0 Fast64 PSNR/SSIM/RMSE/MAE matched V1 exactly; first-batch train stream and all checkpoint stream hashes matched V1. The real batch=2 two-step readiness passed with frozen MedDINO and finite native/new gradients. L506 and external-memory opens were zero.

## EMA Fast64 trajectory

| Step | V1 PSNR | MLSF PSNR | Delta PSNR | V1 SSIM | MLSF SSIM | Delta SSIM | V1 RMSE HU | MLSF RMSE HU | V1 MAE HU | MLSF MAE HU |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 40.623375 | 40.623375 | +0.000000 | 0.946056 | 0.946056 | +0.000000 | 18.7191 | 18.7191 | 12.3019 | 12.3019 |
| 1000 | 40.678683 | 40.818798 | +0.140115 | 0.946614 | 0.948051 | +0.001437 | 18.6005 | 18.3029 | 12.2378 | 12.0840 |
| 2000 | 40.775686 | 40.863531 | +0.087845 | 0.947631 | 0.948429 | +0.000798 | 18.3950 | 18.2107 | 12.1409 | 12.0285 |
| 4000 | 40.862841 | 40.900376 | +0.037535 | 0.948459 | 0.948703 | +0.000244 | 18.2124 | 18.1343 | 12.0409 | 12.0153 |
| 6000 | 40.892207 | 40.924381 | +0.032174 | 0.948718 | 0.948917 | +0.000199 | 18.1519 | 18.0855 | 11.9962 | 11.9785 |
| 8000 | 40.908237 | 40.939682 | +0.031445 | 0.948847 | 0.949028 | +0.000181 | 18.1192 | 18.0542 | 11.9867 | 11.9596 |

## Paired 8k findings

PSNR improved on 62/64 slices (96.9%). Mean paired delta PSNR = +0.031445 dB; fixed-seed 10,000-replicate paired-slice bootstrap 95% CI = [+0.027782, +0.035037] dB. This is within-patient slice uncertainty, not cross-patient significance.
Soft-tissue RMSE HU: V1 24.1206, MLSF 24.0561; high-gradient RMSE HU: V1 26.6213, MLSF 26.3524. Masks exactly reuse the previous MedDINO feature audit: center LDCT HU foreground > -500, soft [-160,240] within foreground, and Gaussian sigma 0.8 + Scharr top 10% within foreground.
Peak training allocation 18.41 GiB; mean training step 0.962 s; sampling only inference per slice V1 0.364 s, MLSF 0.360 s.

## Decision and limits

Full-L333 gate eligible: **True** (requires delta PSNR >= +0.03 dB, >=60% slices improved, and nondecreasing SSIM).
The conclusion is limited to one L333 patient's fixed Fast64 slices. More frequency energy or module activity alone would not establish real anatomical improvement. No L506 or external-memory data were accessed.

## Full-L333 gated validation

A single authorized Full-L333 evaluation (610 slices) gave PSNR 40.946909 dB, SSIM 0.949096, RMSE 18.0377 HU, MAE 11.9491 HU. No L506 or external-memory data were opened.

## Follow-up MLSF module ablations (Full-L333, 610 slices)

Subsequent independently trained 8k EMA arms A2/A3/A4 were evaluated on the same L333 validation patient and fixed evaluation protocol. V1 and full MLSF are reused from the matched evaluations above; these values are an addendum to the original MLSF experiment, not a rerun of the original model.

| 模型 | PSNR (dB) | SSIM | RMSE (HU) | MAE (HU) | PSNR 相对完整 MLSF (dB) |
|---|---:|---:|---:|---:|---:|
| MedDINO V1 | 40.915844 | 0.948920 | 18.101865 | 11.975785 | −0.031065 |
| **完整 MLSF** | **40.946909** | **0.949096** | **18.037721** | 11.949062 | — |
| A2：无局部 Attention | 40.942867 | 0.949043 | 18.045929 | 11.951393 | −0.004043 |
| A3：仅 F128 注入 | 40.922897 | 0.948931 | 18.087386 | 11.971644 | −0.024013 |
| A4：仅 F256 注入 | 40.943912 | 0.949069 | 18.043838 | **11.945893** | −0.002997 |

- **A2 / no Local Attention**: F128 = L128; remove only the new 3x3 MedDINO local cross-attention, while retaining LocalDetailEncoder, F256 fusion and all four MLSF FiLM corrections.
- **A3 / F128-only injection**: retain new MLSF FiLM corrections at E2/D1 only; E1/D2 use the original V1 modulation.
- **A4 / F256-only injection**: retain new MLSF FiLM corrections at E1/D2 only; E2/D1 use the original V1 modulation. F256 still depends on F128, so this is **not** a removal of the F128 feature path.

Full MLSF has the highest PSNR/SSIM and lowest RMSE; A4 has the lowest MAE. The observed Local Attention difference is about +0.004 dB relative to A2, while full MLSF exceeds A4 by about +0.003 dB. A3 versus A4 changes both the conditioning features and injection locations and does not isolate feature resolution. All comparisons are from one validation patient, not an independent cross-patient generalization test.

The complete follow-up run records are retained remotely under `experiments/rddm_mlsf_ablation_v1/` (`ABLATION_REPORT.md`, `ABLATION_RESULTS.csv`, checkpoints). This GitHub addendum records the supplied final metrics only; the ablation source, per-slice metrics, and model weights are not included here. No retraining or checkpoint modifications were made as part of this documentation update.
