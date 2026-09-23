# Fast Screen V1 archive

This archive preserves the actual eight-candidate RDDM Fast-dev screening queue from the fixed Mayo-2016 protocol. It is a method-screening result, not a final paper result. Parent commit: `8fa99bf959da6051ec169bd3817b2bedce2788b6`.

The exact executed variant source is `baselines/founddiff_rddm_rleb/fast_dev_variants/`. Exact runtime configs and protected-artifact hash pins are retained at `baselines/founddiff_rddm_rleb/experiments/fast_dev/candidates/`; the eight subdirectories here repeat their configs and provide complete 2k/4k/6k/8k metrics, smoke summaries and access audits. `FAST_SCREEN_V1_LONG.csv` contains all 32 checkpoint-level records.

The original queue expects the completed Fast baseline summary and private dataset at the frozen remote paths; `baseline_reference.json` captures the required lightweight baseline reference. Checkpoint files, private medical images, per-step `train.jsonl` logs and secrets are intentionally absent from Git. The archived source and configs are byte-for-byte copies of those used in the completed run.

All candidates completed. Only F4c had positive Delta PSNR at both 6k and 8k, and none reached +0.05 dB at 8k. See `FAST_SCREEN_V1_REPORT.md` for the evidence and limits. L506 and external-memory image accesses were zero. No second-round experiment was started.
