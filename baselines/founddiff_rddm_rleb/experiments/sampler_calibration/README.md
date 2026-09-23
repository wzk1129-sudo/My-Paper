# 60k RDDM+RLEB sampler calibration

This is an inference-only calibration of the completed **FoundDiff RDDM+RLEB backbone adapted to our fixed Mayo-2016 protocol**. It uses `step_60000.pt` **EMA** weights, all 610 L333 validation slices, the original frozen HU preprocessing and metric implementation, and the original per-slice validation seeds. L506 was not accessed. No optimizer step or model change occurred.

The 2-step / `sum_scale=0.01` control reproduced the formal validation result exactly: **40.65805039 dB** PSNR. The best tested configuration was 8 steps / `sum_scale=0`: **40.67628429 dB**, a gain of **0.01823390 dB**. Relative to the control, SSIM changed by +0.00014571, RMSE by −0.03831 HU, and MAE by −0.01987 HU. Full validation took 284.3 s versus 119.6 s for the control (about 2.38×).

The gain is small for the added inference cost. Sampler settings are not treated as a major performance bottleneck or paper contribution. **The frozen Fast-dev comparison setting remains 2 steps / `sum_scale=0.01`** unless an experiment explicitly states otherwise. Selection here used L333 and is not an independent test result.

`sum_scale=0` was set on the loaded diffusion object at runtime. This changes the initial sampling noise only; the already loaded diffusion coefficient buffers remain unchanged. The archived `sweep_sampler.py` is an exact copy of the script used on the remote machine and retains its original absolute experiment paths for provenance. It is not part of the formal training entry point.

Files: `sampler_sweep.csv` and `sampler_sweep.json` contain the aggregate results and protocol record; `SAMPLER_CALIBRATION_REPORT.md` contains the full table. Per-slice metric files, logs, datasets, and checkpoint weights stay on the remote machine and are not archived in Git.
