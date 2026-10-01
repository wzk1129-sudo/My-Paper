# Official RDDM restoration U-Net → Mayo CT (matched Fast-Dev)

This branch contains the exact [implementation, protocol, audits, and result report](experiments/rddm_official_ct_adapt_v1/REPORT.md) for comparing the existing simplified RDDM/RLEB backbone against the official RDDM restoration U-Net on Mayo 2016 paired LDCT/NDCT.

- Official source: [nachifur/RDDM](https://github.com/nachifur/RDDM) commit `524caea34cedb3733aa5bf6c27dc1cd6e0469216`.
- The two arms share the CT preprocessing, residual `pred_res` objective, diffusion and 2-step sampler, optimizer, seed, train order, and L333-fast64 evaluator.
- Train: L067/L143. Validation: fixed L333-fast64. L506 access: **0**.
- Control: 13,130,945 parameters. Official candidate: 36,248,513 parameters.
- Both arms completed 8,000 steps. No Full-L333 evaluation was run.

| EMA Fast64 step | Simplified PSNR (dB) | Official PSNR (dB) | Official − simplified (dB) |
|---:|---:|---:|---:|
| 0 | 13.0154 | 8.5760 | −4.4394 |
| 2,000 | 38.4787 | 40.0288 | +1.5501 |
| 4,000 | 40.1454 | 40.4472 | +0.3018 |
| 6,000 | 40.3872 | 40.5665 | +0.1793 |
| 8,000 | 40.4846 | 40.6234 | +0.1388 |

At 8,000 steps, the official candidate also changed SSIM by **+0.001186**, RMSE by **−0.295735 HU**, and MAE by **−0.157858 HU**. The small late advantage supports a possible effect of backbone simplification in this short, single-seed test; it does not establish a patient-level or full-training conclusion.

Start with [REPORT.md](experiments/rddm_official_ct_adapt_v1/REPORT.md). The main code is [simplified control](experiments/rddm_official_ct_adapt_v1/src/rddm_rleb.py), [official candidate](experiments/rddm_official_ct_adapt_v1/src/official_rddm_ct.py), [matched trainer](experiments/rddm_official_ct_adapt_v1/matched_train.py), and [readiness audit](experiments/rddm_official_ct_adapt_v1/readiness.py). Exact [metrics](experiments/rddm_official_ct_adapt_v1/results/fast_dev_metrics.csv) and [per-step training diagnostics](experiments/rddm_official_ct_adapt_v1/results/training_diagnostics.csv) are committed. CT images and checkpoints are excluded.

Remote CoreDiff worktree commit: `eb52321b50112ac9361095683530098fbf7e5df2`.
