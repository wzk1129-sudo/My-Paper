#!/usr/bin/env bash
set -euo pipefail
cd /home/xujian/projects/CoreDiff/experiments/rddm_prosac_add_fusion_v1_20260926/baselines/founddiff_rddm_rleb
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export CUDA_VISIBLE_DEVICES=0
exec /home/xujian/miniconda3/envs/flexict/bin/python -m meddino_four_arm_v1.queue
