#!/usr/bin/env bash
set -euo pipefail
export CUBLAS_WORKSPACE_CONFIG=:4096:8
cd /home/xujian/projects/CoreDiff/experiments/rddm_res_noise_beta_time_v1_20260929/baselines/founddiff_rddm_rleb
PY=/home/xujian/miniconda3/envs/flexict/bin/python
ROOT=experiments/res_noise_beta_time_v1
CAL=$ROOT/runs/calibration_100step_20260929
RUN=$ROOT/runs/beta_time_fast8k_20260929
DIAG=$ROOT/diagnostics/beta_time_4k8k_20260929
exec > "$ROOT/launches/calibration_fast8k_diagnostic.log" 2>&1
echo "START $(date -Is) branch=$(git branch --show-current) head=$(git rev-parse HEAD)"
"$PY" -m res_noise_beta_time_v1.run --phase calibrate --output "$CAL" --device cuda:0
"$PY" -m res_noise_beta_time_v1.run --phase formal --calibration "$CAL/CALIBRATION.json" --output "$RUN" --device cuda:0
"$PY" -m res_noise_beta_time_v1.diagnose_noise_path --output "$DIAG" --device cuda:0
echo "COMPLETE $(date -Is)"
