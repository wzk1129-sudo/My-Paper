#!/usr/bin/env bash
set -u
cd /home/xujian/projects/CoreDiff/experiments/rddm_knowledge_prior_serial_v1_20260929/baselines/founddiff_rddm_rleb
export CUBLAS_WORKSPACE_CONFIG=:4096:8
/home/xujian/miniconda3/envs/flexict/bin/python -m knowledge_prior_v1.smoke > experiments/knowledge_prior_serial_v1/SMOKE.log 2>&1
rc=$?
echo "$rc" > experiments/knowledge_prior_serial_v1/SMOKE.exit
exit "$rc"
