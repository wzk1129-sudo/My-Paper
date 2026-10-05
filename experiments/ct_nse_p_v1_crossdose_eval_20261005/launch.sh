#!/usr/bin/env bash
set -euo pipefail
cd /home/xujian/projects/CoreDiff
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export CUDA_VISIBLE_DEVICES=0
/home/xujian/miniconda3/envs/corediff/bin/python -u experiments/ct_nse_p_v1_crossdose_eval_20261005/pairing.py > experiments/ct_nse_p_v1_crossdose_eval_20261005/logs/pairing.log 2>&1
exec /home/xujian/miniconda3/envs/flexict/bin/python -u experiments/ct_nse_p_v1_crossdose_eval_20261005/evaluate_crossdose.py > experiments/ct_nse_p_v1_crossdose_eval_20261005/logs/eval_retry1.log 2>&1
