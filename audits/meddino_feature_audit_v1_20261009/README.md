# MedDINO feature effectiveness audit V1

Read-only audit of Official RDDM 2.5D + MedDINOv3 Condition V1 at step 8000 EMA, using the fixed L333-fast64 validation set (64 slices). No training or model updates were performed.

## Findings

- The normal-condition baseline reproduces 40.908237 dB PSNR, 0.948847 SSIM, 18.119234 HU RMSE, and 11.986737 HU MAE.
- Raw MedDINO L11 has the strongest same-position cross-dose consistency (foreground cosine 0.9770); L8 has the largest hard-neighbor margin (0.0439). The trained Adapter/Fusion representation is weaker on this particular retrieval diagnostic.
- Turning off all nine FiLM condition sites reduces PSNR by 0.145259 dB. Replacing the condition with a distant slice reduces it by 0.244204 dB. Removing spatial variation while preserving channel means reduces it by 0.105847 dB. These are interventions on an already trained model, not matched training gains.
- The audit concludes that spatial MedDINO condition is used by V1, while its independently matched reconstruction gain remains small. See `AUDIT_REPORT.md` for caveats and complete analysis.

## Files

- `AUDIT_REPORT.md`: findings, methods, limitations, and conclusion.
- `run_audit.py`, `test_audit.py`: audit entry point and synthetic-data checks.
- `FEATURE_CROSSDOSE.csv`, `SPATIAL_MATCHING.csv`: per-slice feature diagnostics.
- `CONDITION_INTERVENTIONS.csv`, `PER_SLICE_INTERVENTIONS.csv`: aggregate and per-slice condition interventions.
- `FEATURE_SUMMARY.json`, `MATCHING_SUMMARY.json`, `BASELINE_RESULT.json`, `SMOKE_RESULT.json`: supporting summaries and checks.
- `AUDIT_MANIFEST.json`, `DATA_ACCESS_AUDIT.json`, `POSTRUN_INTEGRITY.json`, `LAUNCH_RECORD.json`: provenance, access, and integrity evidence.

The original code and results were produced in `/home/xujian/projects/CoreDiff_official_2p5d_meddino_condition_v1/experiments/meddino_feature_audit_v1/`. This package excludes checkpoints, weights, CT images, feature arrays, and the full execution log. The audited checkpoints remain on the remote machine.
