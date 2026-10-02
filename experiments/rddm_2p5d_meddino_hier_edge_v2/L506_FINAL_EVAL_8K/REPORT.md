# V2@8k EMA final-only L506 evaluation

Status: COMPLETE. One read-only inference pass over all524 valid center slices, with both HU windows applied to the same saved float32 prediction array.

## Exact input and model protocol
- Checkpoint: /home/xujian/projects/CoreDiff_meddino_hier_edge_v2/experiments/rddm_2p5d_meddino_hier_edge_v2/runs/2P5D_MEDDINO_HIER_EDGE_V2_8K/checkpoints/step_8000.pt
- Checkpoint SHA256: c8c291a8b78f87f6809565ea00808ff28ff8e3182f9a3206c6acfc4ab7a38684
- V2 official RDDM restoration backbone, frozen MedDINOv3 layers[2,5,8,11], hierarchical edge condition; step8000 EMA.
- Mayo 2016 dose25 NPY series, full valid centers1..524; first/last slices0/525 used only as LDCT context, not target.
- Per center: prev/center/next LDCT, center NDCT target; stored pixel -1024 -> clip[-1024,3072] HU -> [0,1]. No soft-tissue input clipping.
- Unmodified V2 sampler, 2 sampling steps, sum_scale0.01; seed20260829+slice_index; batch1; no augmentation.

## Numeric results
Arithmetic means of 524 per-slice scores. Main and soft-tissue metrics reuse the identical prediction bytes.
| Model / window | PSNR dB | SSIM | RMSE HU | MAE HU |
|---|---:|---:|---:|---:|
| V2@8k L333-fast64 [-1000,1000] | 40.89927864 | 0.94875808 | 18.13759402 | 11.97600955 |
| V2@8k L506 [-1000,1000] | 41.62109150 | 0.95585624 | 16.87933262 | 11.15806265 |
| Official CoreDiff L506 unified [-1000,1000] | 41.60937146 | 0.95580814 | 16.91536520 | 11.23689846 |
| V2@8k L506 [-160,240] | 29.63775970 | 0.88307931 | 13.50128009 | 6.35583995 |
| Official CoreDiff L506 [-160,240] | 29.60120914 | 0.88751400 | 13.56665441 | 6.39630592 |

V2 minus official CoreDiff on L506, matching evaluation window:
| Window | Delta PSNR dB | Delta SSIM | Delta RMSE HU | Delta MAE HU |
|---|---:|---:|---:|---:|
| [-1000,1000] | +0.01172004 | +0.00004811 | -0.03603258 | -0.07883581 |
| [-160,240] | +0.03655056 | -0.00443469 | -0.06537432 | -0.04046596 |

CoreDiff reference is its official released150k EMA model, scored with the unified HU-window metrics for numeric comparison. The two models have different training and sampling protocols; these values alone do not isolate an architecture effect. No L506 result affected training, hyperparameters or checkpoint selection.

## Metric implementation
Both normalized prediction and target are converted to HU, then clipped to the stated post-hoc window. PSNR ranges2000 and400 HU. SSIM reuses the project frozen slice_metrics: skimage win7, gaussian_weights=False, use_sample_covariance=True, K1=.01, K2=.03. RMSE and MAE use clipped HU. Scores are per-slice then averaged.

## Integrity and artifacts
- Prediction: /home/xujian/projects/CoreDiff_meddino_hier_edge_v2/experiments/rddm_2p5d_meddino_hier_edge_v2/L506_FINAL_EVAL_8K/predictions_l506_524.npy; shape[524,1,512,512], float32 normalized [0,1] before HU windowing; SHA256 fe0654656ea8a5e0f00390c8cf2c4810ec1b315c2c637222605f5d801517a0db.
- Prediction stream SHA256: 85a7d6e81f11d0e49fb8890edcbf4936f55ddfec80792712d76168919cb17428; all524 reopened per-slice prediction hashes match both CSVs.
- Checkpoint before/after SHA256: c8c291a8b78f87f6809565ea00808ff28ff8e3182f9a3206c6acfc4ab7a38684 / c8c291a8b78f87f6809565ea00808ff28ff8e3182f9a3206c6acfc4ab7a38684. Model and MedDINO parameter hashes unchanged; no gradients.
- Data audit: only L506 views opened, center targets only, no other-patient reads or data writes. See DATA_ACCESS_AUDIT.json.
- Exact runtime config EVALUATION_CONFIG.json; environment ENVIRONMENT.json; metrics MAIN_1000_METRICS.csv and SOFT_TISSUE_160_240_METRICS.csv; summary SUMMARY.json.
- Prediction remains private on the remote machine and is ignored by Git. No training, tuning, OSL, L333 re-evaluation or other checkpoint evaluation.
