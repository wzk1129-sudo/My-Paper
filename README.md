# FoundDiff clean RDDM + RLEB baseline

**FoundDiff RDDM+RLEB backbone adapted to our fixed Mayo-2016 protocol.**

This repository snapshot records the completed 60,000-step experiment from
2026-09-21. It is a reconstruction from the official FoundDiff source, with the
commented residual-block timestep MLP restored. It is **not an official FoundDiff
Table-V reproduction**. No DA-CLIP, DACB, attention, Mamba, SS2D, CSSM, dose or
anatomy conditioning is included in the executable baseline.

- [Experiment, protocol and final results](experiments/founddiff_rddm_baseline/EXPERIMENT.md)
- [Actual configuration](experiments/founddiff_rddm_baseline/CONFIG.json)
- [All validation checkpoints](experiments/founddiff_rddm_baseline/validation_metrics.csv)
- [Final training summary](experiments/founddiff_rddm_baseline/final_training_summary.json)
- [Exact source hashes](experiments/founddiff_rddm_baseline/source_manifest.json)
- [Code and execution notes](baselines/founddiff_rddm_rleb/README.md)

Best full-L333 EMA validation: **60k, PSNR 40.6581 dB, SSIM 0.946132,
RMSE 18.6489 HU, MAE 12.1817 HU**. The held-out test split was not accessed.

Only code, configurations, documentation and small aggregate results are included.
No dataset, image, prediction, per-slice result, training log or model weight is
distributed. Checkpoint manifests contain identifiers and hashes, not weights.
The original remote source and completed experiment remain unchanged.

## Provenance

Official source: [hao1635/FoundDiff](https://github.com/hao1635/FoundDiff),
commit `c85a3a0e87ae2e4587650f0fb7efc15f3311d8e2`.
See [source attribution](SOURCE_NOTICE.md). Historical audits are preserved as
dated evidence; their pre-training statements are superseded by the completed
results in `EXPERIMENT.md`.
