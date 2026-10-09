# Official RDDM + 2.5D + MedDINOv3 V1 and MLSF-RDDM

This branch publishes the actual implementation, effective protocols, scalar results and audit evidence for the matched 8k V1 and MLSF-RDDM experiments. Start with the [V1 experiment](experiments/rddm_official_2p5d_meddino_condition_v1/README.md) and the [MLSF experiment](experiments/rddm_mlsf_v1/README.md).

Start with [the experiment README](experiments/rddm_official_2p5d_meddino_condition_v1/README.md), [REPORT](experiments/rddm_official_2p5d_meddino_condition_v1/REPORT.md), [effective config](experiments/rddm_official_2p5d_meddino_condition_v1/EFFECTIVE_RUN_CONFIG.json), and [audit guide](experiments/rddm_official_2p5d_meddino_condition_v1/AUDIT_GUIDE.md).

| EMA L333-fast64 8k | Control: Official RDDM + 2.5D | Candidate: + frozen MedDINOv3 condition | Candidate − Control |
|---|---:|---:|---:|
| PSNR dB | 40.889946 | 40.908237 | +0.018291 |
| SSIM | 0.948667 | 0.948847 | +0.000180 |
| RMSE HU | 18.156012 | 18.119234 | −0.036778 |
| MAE HU | 11.984496 | 11.986737 | +0.002241 |

Both V1 arms start from the same prior 2D 8k EMA and then train 8k additional steps with fresh Adam, matched sample/augmentation/timestep/noise streams. All 64 PSNR slice deltas are positive. Full-L333 was later evaluated for the formal MedDINO V1 at 8k to match MLSF; see the MLSF files below.

## MLSF-RDDM versus formal MedDINO V1

MLSF adds a local multi-scale fusion branch to the same formal V1 backbone. Its zero-initialized output preserves V1 at step 0. It starts from the same prior 2D 8k EMA, with an independent optimizer and the matched 8k training stream. See [MLSF source and audits](experiments/rddm_mlsf_v1/README.md), [training/report](experiments/rddm_mlsf_v1/REPORT.md), and the [610-slice matched comparison](experiments/rddm_mlsf_v1/FULL_L333_MATCHED_COMPARISON.md).

| EMA at 8k | Formal V1 | MLSF | MLSF − V1 |
|---|---:|---:|---:|
| Fast64 PSNR, dB | 40.908237 | 40.939682 | +0.031445 |
| Full-L333 PSNR, dB (610 slices) | 40.915844 | 40.946909 | +0.031065 |
| Full-L333 SSIM | 0.948920 | 0.949096 | +0.000177 |
| Full-L333 RMSE, HU | 18.101865 | 18.037721 | −0.064144 |
| Full-L333 MAE, HU | 11.975785 | 11.949062 | −0.026723 |

Full-L333 PSNR improved on 607/610 paired slices. The exact values, per-slice scalar metrics, checkpoint identities, access audits and training source are in the experiment directories.

NORMAL / CONDITION_OFF / SHUFFLED_MEDDINO diagnostic PSNR: **40.908237 / 40.762979 / 40.902731 dB**. This is a single-seed, single-validation-patient result with a small gain.

Readiness, all 8000 matched stream hashes, frozen MedDINO, checkpoint reload, and forbidden-access audits passed. L506 / external-memory / neighbor-NDCT accesses are zero. All raw per-slice metric records and actual source snapshots are included.

Offline audit (stdlib only, no medical data/GPU/network):
~~~bash
cd experiments/rddm_official_2p5d_meddino_condition_v1
python3 verify_lightweight_artifacts.py
~~~

The publication contains code, configurations, dependency source snapshots, scalar metrics, hashes and provenance. Model checkpoints, DICOM, NPY medical arrays, medical images and feature caches remain private. See the README for pinned weight and data dependencies needed for GPU reproduction.
