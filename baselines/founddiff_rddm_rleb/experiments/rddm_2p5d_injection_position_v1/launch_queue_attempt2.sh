#!/usr/bin/env bash
set -euo pipefail
cd /home/xujian/projects/CoreDiff/experiments/rddm_2p5d_injection_position_v1_20260925/baselines/founddiff_rddm_rleb
export CUBLAS_WORKSPACE_CONFIG=:4096:8
exec /home/xujian/miniconda3/envs/corediff/bin/python -m experiments.rddm_2p5d_injection_position_v1.queue >> experiments/rddm_2p5d_injection_position_v1/attempt2/queue.log 2>&1
