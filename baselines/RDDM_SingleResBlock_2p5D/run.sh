#!/bin/bash
set -u
set -o noclobber
cd /home/xujian/research/My-Paper/baselines/RDDM_SingleResBlock_2p5D
export CUDA_VISIBLE_DEVICES=0
export CUBLAS_WORKSPACE_CONFIG=:4096:8
/home/xujian/miniforge3/bin/conda run --no-capture-output -n founddiff python -B -u train.py > experiments/RDDM_SingleResBlock_2p5D_8P_8K/logs/train_console.log 2>&1
code=$?
printf '%s' "$code" > experiments/RDDM_SingleResBlock_2p5D_8P_8K/logs/exit_code.txt
exit "$code"
