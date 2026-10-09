#!/usr/bin/env bash
set -euo pipefail
cd /home/xujian/projects/CoreDiff_official_2p5d_meddino_condition_v1
export CUDA_VISIBLE_DEVICES=0
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS=2
/home/xujian/miniconda3/envs/flexict/bin/python experiments/rddm_mlsf_v1/evaluate_mlsf.py \
  --mode FAST \
  --output experiments/rddm_mlsf_v1/STEP0_AUDIT.json \
  > experiments/rddm_mlsf_v1/logs/step0.log 2>&1
