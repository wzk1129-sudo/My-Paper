#!/usr/bin/env bash
set -euo pipefail
cd /home/xujian/projects/CoreDiff/experiments/rddm_cross_slice_error_adapter_v1_20260925/baselines/founddiff_rddm_rleb
export CUBLAS_WORKSPACE_CONFIG=:4096:8
exec /home/xujian/miniconda3/envs/corediff/bin/python -m cross_slice_adapter.experiment train --device cuda:0 > /home/xujian/projects/CoreDiff/experiments/rddm_cross_slice_error_adapter_v1_20260925/baselines/founddiff_rddm_rleb/experiments/cross_slice_error_adapter_v1/formal6k.log 2>&1
