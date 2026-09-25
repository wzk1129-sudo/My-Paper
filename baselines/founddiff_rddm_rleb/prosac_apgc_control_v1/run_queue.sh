#!/usr/bin/env bash
set -euo pipefail
cd /home/xujian/projects/CoreDiff/experiments/rddm_prosac_apgc_control_v1_20260925/baselines/founddiff_rddm_rleb
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export CUDA_VISIBLE_DEVICES=0
exec /home/xujian/miniconda3/envs/flexict/bin/python -m prosac_apgc_control_v1.queue
