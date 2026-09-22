# Unique RDDM fast baseline reference

Run: `rddm_fast_baseline_8k_20260922`. COMPLETE, verified 8000 optimizer steps.

This is an engineering-screening reference, not a final paper result.
No warm start: random initialization with the frozen seed; no smoke/formal checkpoint loaded.

| Step | PSNR | SSIM | RMSE HU | MAE HU |
|---:|---:|---:|---:|---:|
| 2000 | 38.478700 | 0.926035 | 23.929178 | 15.792909 |
| 4000 | 40.145386 | 0.941559 | 19.763954 | 13.024454 |
| 6000 | 40.387202 | 0.944026 | 19.225108 | 12.580673 |
| 8000 | 40.484568 | 0.944869 | 19.014858 | 12.459723 |

All rows: fixed L333-fast64, EMA, two-step deterministic sampling.
Best fast64 PSNR step: 8000.
Plateau report: {"status": "NOT_PLATEAU", "delta_psnr_6000_to_8000": 0.09736594163278767, "automatic_early_stop": false}.
Compute seconds: 2503.473052982241.
Wall seconds (training state through completion): 2583.005016604089.
Training peak allocated VRAM bytes: 4470513664.

Compare each candidate against this reference at the same optimizer steps. Do not compare a fast-method gain against the 8-patient/60k 40.65805 dB score. At 6k and 8k require positive Delta_PSNR; at 8k require >=0.05 dB and review SSIM/RMSE/MAE for material reverse degradation. This is a heuristic, not statistical significance.

Full-L333 inference was not run in this baseline task. Candidate promotion requires the same selected checkpoint and matching baseline checkpoint to be evaluated on all 610 L333 slices before formal 8-patient retraining. L506 remains untouched.

Weights, full train logs and per-slice records remain remote and are not included in Git.
