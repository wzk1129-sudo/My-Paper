#!/usr/bin/env bash
set -euo pipefail
cd /home/xujian/projects/CoreDiff/experiments/rddm_prosac_add_fusion_v1_20260926/baselines/founddiff_rddm_rleb
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export CUDA_VISIBLE_DEVICES=0
exec /home/xujian/miniconda3/envs/flexict/bin/python -m meddino_full_l333_v1.evaluate full >> experiments/meddino_full_l333_validation_v1/full_inference.log 2>&1
