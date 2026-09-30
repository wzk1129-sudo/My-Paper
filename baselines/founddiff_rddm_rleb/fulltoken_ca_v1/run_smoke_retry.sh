#!/usr/bin/env bash
set -u
cd /home/xujian/projects/CoreDiff
cd experiments/rddm_meddino_fulltoken_ca_v1_20261001/baselines/founddiff_rddm_rleb
OUT=experiments/meddino_fulltoken_ca_v1
mkdir -p "$OUT"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
CUDA_VISIBLE_DEVICES=0 /home/xujian/miniconda3/envs/flexict/bin/python -m fulltoken_ca_v1.readiness --query-chunk 0 > "$OUT/SMOKE_RETRY.log" 2>&1
status=$?
printf '%s\n' "$status" > "$OUT/SMOKE_RETRY.exit"
exit "$status"
