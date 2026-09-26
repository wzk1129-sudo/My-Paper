# PDC-StructSup-BCA V1 result

Independent 6000-step Fast protocol, center-NDCT-derived side structure supervision, step6000 EMA, frozen L333 evaluation. Final 20-step smoke PASS. Lambdas: 128=0.02, 64=0.02, clean-image L1=0.02. The original residual MSE remains unchanged.

| Step | Candidate Fast64 PSNR | PDC parent | Δ vs PDC | E12-BCA | Δ vs E12 |
|---:|---:|---:|---:|---:|---:|
| 0 | 40.765826 | 40.765826 | +0.000000 | 40.765823 | +0.000003 |
| 500 | 40.745025 | 40.755243 | -0.010218 | 40.755037 | -0.010012 |
| 1000 | 40.778430 | 40.782997 | -0.004567 | 40.782949 | -0.004519 |
| 2000 | 40.801410 | 40.805114 | -0.003704 | 40.806180 | -0.004769 |
| 4000 | 40.831291 | 40.831795 | -0.000504 | 40.834022 | -0.002731 |
| 6000 | 40.848106 | 40.846740 | +0.001366 | 40.848542 | -0.000436 |

Full L333 (610 EMA slices): PSNR 40.85553842 dB, SSIM 0.94854109, RMSE 18.226231 HU, MAE 12.003861 HU. ΔPSNR vs PDC +0.00117411 dB (paired slice bootstrap 95% CI [+0.00058940,+0.00174182], positive 60.0%); vs E12-BCA -0.00051608 dB (CI [-0.00088987,-0.00013866], positive 42.1%). Each bootstrap uses 10,000 draws over slices from one patient; not patient-level significance.

Structure side loss declines substantially; final beta128/64=0.096034/0.093251, BCA alpha128/64=0.200702/0.212096. Peak train VRAM 5.731 GiB, mean optimizer step 0.448s. `FINAL_AUDIT.json` PASS: nonfinite=0, L506/external-memory/neighbor-target/blocked opens=0, frozen formal source/split/evaluator hashes unchanged. The parent-relative gain is too small to treat as a useful performance improvement; see `STRUCTSUP_FAILURE_ANALYSIS.md`. Model source/EMA checkpoints and original parent results were not modified. The first import-only queue failure and clean retry are retained in audit records.
