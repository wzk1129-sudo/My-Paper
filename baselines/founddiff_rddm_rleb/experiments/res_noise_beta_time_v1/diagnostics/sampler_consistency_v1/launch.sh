#!/usr/bin/env bash
set -uo pipefail
cd /home/xujian/projects/CoreDiff/experiments/rddm_res_noise_beta_time_v1_20260929/baselines/founddiff_rddm_rleb
out=experiments/res_noise_beta_time_v1/diagnostics/sampler_consistency_v1
export PYTHONPATH=.
export CUDA_VISIBLE_DEVICES=0
/home/xujian/miniconda3/envs/flexict/bin/python "$out/run_consistency.py" > "$out/run.log" 2>&1
rc=$?
printf '%s\n' "$rc" > "$out/EXIT_STATUS"
exit "$rc"
