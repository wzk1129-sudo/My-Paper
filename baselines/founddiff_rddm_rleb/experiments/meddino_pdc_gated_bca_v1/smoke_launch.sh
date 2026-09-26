#!/bin/bash
set -euo pipefail
export CUBLAS_WORKSPACE_CONFIG=:4096:8
cd /home/xujian/projects/CoreDiff/experiments/rddm_meddino_pdc_gated_bca_v1_20260926/baselines/founddiff_rddm_rleb && /home/xujian/miniconda3/envs/flexict/bin/python -u -m meddino_pdc_gated_bca_v1.smoke > /home/xujian/projects/CoreDiff/experiments/rddm_meddino_pdc_gated_bca_v1_20260926/baselines/founddiff_rddm_rleb/experiments/meddino_pdc_gated_bca_v1/SMOKE_20_RETRY.log 2>&1
