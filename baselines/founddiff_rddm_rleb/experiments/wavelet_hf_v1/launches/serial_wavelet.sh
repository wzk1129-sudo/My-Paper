#!/usr/bin/env bash
set -euo pipefail
cd /home/xujian/projects/CoreDiff/experiments/rddm_2p5d_wavelet_hf_v1_20260929/baselines/founddiff_rddm_rleb
out=experiments/wavelet_hf_v1
trap 'rc=$?; printf "%s\n" "$rc" > "$out/EXIT_STATUS_RETRY"' EXIT
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=.
/home/xujian/miniconda3/envs/corediff/bin/python -u wavelet_hf_v1/run.py --phase prepare > "$out/calibration_100/retry_corediff.log" 2>&1
/home/xujian/miniconda3/envs/corediff/bin/python -u wavelet_hf_v1/run.py --phase train > "$out/launches/formal_train.log" 2>&1
