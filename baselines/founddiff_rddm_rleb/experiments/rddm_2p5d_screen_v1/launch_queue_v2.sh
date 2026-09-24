#!/usr/bin/env bash
set -euo pipefail
cd /home/xujian/projects/CoreDiff/experiments/rddm_2p5d_screen_v1_20260924/baselines/founddiff_rddm_rleb
export CUBLAS_WORKSPACE_CONFIG=:4096:8
exec /home/xujian/miniconda3/envs/corediff/bin/python -m context_2p5d.queue > /home/xujian/projects/CoreDiff/experiments/rddm_2p5d_screen_v1_20260924/baselines/founddiff_rddm_rleb/experiments/rddm_2p5d_screen_v1/queue_attempt2.log 2>&1
