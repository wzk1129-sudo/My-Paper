#!/usr/bin/env bash
set -euo pipefail
cd /home/xujian/projects/CoreDiff_official_rddm_ct_adapt_v1/experiments/rddm_official_ct_adapt_v1
export CUDA_VISIBLE_DEVICES=0
export CUBLAS_WORKSPACE_CONFIG=:4096:8
PY=/home/xujian/miniconda3/envs/corediff/bin/python
mkdir -p logs
"$PY" -c 'import json; r=json.load(open("readiness.json")); m=json.load(open("matched_readiness_audit.json")); assert r["forward_b2_1x512_pass"] and r["backward_finite"] and r["L506_access_count"]==0 and all(m["matches"].values())'
test ! -e runs/control_8k_20261001
test ! -e runs/candidate_8k_20261001
"$PY" matched_train.py --arm CONTROL_STRIPPED --steps 8000 \
  --output runs/control_8k_20261001 2>&1 | tee logs/control_8k_20261001.log
"$PY" matched_train.py --arm CANDIDATE_OFFICIAL_RDDM --steps 8000 \
  --output runs/candidate_8k_20261001 2>&1 | tee logs/candidate_8k_20261001.log
