#!/usr/bin/env bash
set -u
cd /home/xujian/projects/CoreDiff/experiments/rddm_knowledge_prior_serial_v1_20260929/baselines/founddiff_rddm_rleb
export CUBLAS_WORKSPACE_CONFIG=:4096:8
/home/xujian/miniconda3/envs/flexict/bin/python -m knowledge_prior_v1.retry_init_check > experiments/knowledge_prior_serial_v1/A_INIT_RETRY.log 2>&1
rc=$?
echo "$rc" > experiments/knowledge_prior_serial_v1/A_INIT_RETRY.exit
exit "$rc"
