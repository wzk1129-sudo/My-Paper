#!/usr/bin/env bash
set -euo pipefail
cd /home/xujian/projects/CoreDiff/experiments/mayo_soft_tissue_metric_v1_20260925/baselines/founddiff_rddm_rleb
export CUBLAS_WORKSPACE_CONFIG=:4096:8
exec /home/xujian/miniconda3/envs/corediff/bin/python -m diagnostics.mayo_soft_tissue_metric_v1.eval_soft_tissue --device cuda:0 > diagnostics/mayo_soft_tissue_metric_v1/evaluation_retry1.log 2>&1
