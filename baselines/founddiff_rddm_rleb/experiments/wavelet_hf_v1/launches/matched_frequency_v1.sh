#!/usr/bin/env bash
set -euo pipefail
cd /home/xujian/projects/CoreDiff/experiments/rddm_2p5d_wavelet_hf_v1_20260929/baselines/founddiff_rddm_rleb
trap 'rc=$?; printf "%s\n" "$rc" > experiments/wavelet_hf_v1/launches/MATCHED_FREQ_EXIT_STATUS' EXIT
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=.
/home/xujian/miniconda3/envs/corediff/bin/python -u wavelet_hf_v1/matched_frequency.py > experiments/wavelet_hf_v1/launches/matched_frequency_v1.log 2>&1
