#!/usr/bin/env bash
set -euo pipefail
cd /home/xujian/projects/CoreDiff_official_2p5d_meddino_condition_v1
export CUDA_VISIBLE_DEVICES=0
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS=2
PY=/home/xujian/miniconda3/envs/flexict/bin/python
EXP=experiments/rddm_mlsf_v1
while tmux has-session -t rddm_mlsf_v1_8k_20261009 2>/dev/null; do sleep 60; done
$PY -c 'import json; from pathlib import Path; p=Path("experiments/rddm_mlsf_v1/runs/MLSF_RDDM_8K_20261009/FINAL_STATE.json"); r=json.loads(p.read_text()); assert r["optimizer_step"]==r["ema_updates"]==8000'
while [ "$(/usr/lib/wsl/lib/nvidia-smi --query-compute-apps=pid --format=csv,noheader | awk 'NF>0' | wc -l)" -gt 0 ]; do sleep 60; done
$PY $EXP/postprocess_mlsf.py > $EXP/logs/postprocess.log 2>&1
if $PY -c 'import json,sys; r=json.load(open("experiments/rddm_mlsf_v1/FULL_L333_GATE.json")); sys.exit(0 if r["eligible"] else 1)'; then
  $PY $EXP/evaluate_mlsf.py --checkpoint $EXP/runs/MLSF_RDDM_8K_20261009/checkpoints/step_8000.pt --mode FULL --output $EXP/FULL_L333_RESULT.json > $EXP/logs/full_l333.log 2>&1
fi
$PY $EXP/finalize_report.py > $EXP/logs/finalize.log 2>&1
