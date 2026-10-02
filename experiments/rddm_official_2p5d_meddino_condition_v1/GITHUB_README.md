# Official RDDM + 2.5D + MedDINOv3 Anatomy Condition V1

This branch publishes the actual implementation, effective protocol, scalar results and audit evidence for the completed matched 8k continuation experiment.

Start with [the experiment README](experiments/rddm_official_2p5d_meddino_condition_v1/README.md), [REPORT](experiments/rddm_official_2p5d_meddino_condition_v1/REPORT.md), [effective config](experiments/rddm_official_2p5d_meddino_condition_v1/EFFECTIVE_RUN_CONFIG.json), and [audit guide](experiments/rddm_official_2p5d_meddino_condition_v1/AUDIT_GUIDE.md).

| EMA L333-fast64 8k | Control: Official RDDM + 2.5D | Candidate: + frozen MedDINOv3 condition | Candidate − Control |
|---|---:|---:|---:|
| PSNR dB | 40.889946 | 40.908237 | +0.018291 |
| SSIM | 0.948667 | 0.948847 | +0.000180 |
| RMSE HU | 18.156012 | 18.119234 | −0.036778 |
| MAE HU | 11.984496 | 11.986737 | +0.002241 |

Both arms start from the same prior 2D 8k EMA and then train 8k additional steps with fresh Adam, matched sample/augmentation/timestep/noise streams. All 64 PSNR slice deltas are positive, but the +0.03 dB promotion gate was not met; Full-L333 was not run.

NORMAL / CONDITION_OFF / SHUFFLED_MEDDINO diagnostic PSNR: **40.908237 / 40.762979 / 40.902731 dB**. This is a single-seed, single-validation-patient result with a small gain.

Readiness, all 8000 matched stream hashes, frozen MedDINO, checkpoint reload, and forbidden-access audits passed. L506 / external-memory / neighbor-NDCT accesses are zero. All raw per-slice metric records and actual source snapshots are included.

Offline audit (stdlib only, no medical data/GPU/network):
~~~bash
cd experiments/rddm_official_2p5d_meddino_condition_v1
python3 verify_lightweight_artifacts.py
~~~

The publication contains code, configurations, dependency source snapshots, scalar metrics, hashes and provenance. Model checkpoints, DICOM, NPY medical arrays, medical images and feature caches remain private. See the README for pinned weight and data dependencies needed for GPU reproduction.
