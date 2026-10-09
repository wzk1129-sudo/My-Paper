#!/usr/bin/env bash
set -euo pipefail
cd /home/xujian/projects/CoreDiff_official_2p5d_meddino_condition_v1
v1_root=experiments/rddm_official_2p5d_meddino_condition_v1
result_name=FULL_L333_V1_8K_MLSF_MATCH
test ! -e "$v1_root/$result_name.json"
test ! -e "$v1_root/${result_name}_PER_SLICE.json"
export PYTHONPATH="$PWD/$v1_root${PYTHONPATH:+:$PYTHONPATH}"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
printf 'RUN_START %s\n' "$(date -Is)"
printf 'PROJECT %s\n' "$PWD"
printf 'GIT_HEAD %s\n' "$(git rev-parse HEAD)"
printf 'PYTHON %s\n' /home/xujian/miniconda3/envs/flexict/bin/python
printf 'GPU 0\n'
printf 'EVAL_CALL eval_checkpoint("OFFICIAL_RDDM_2P5D_MEDDINO_CONDITION","FULL","FULL_L333_V1_8K_MLSF_MATCH")\n'
/home/xujian/miniconda3/envs/flexict/bin/python -c 'from postprocess import eval_checkpoint; eval_checkpoint("OFFICIAL_RDDM_2P5D_MEDDINO_CONDITION","FULL","FULL_L333_V1_8K_MLSF_MATCH")'
printf 'RUN_FINISH %s\n' "$(date -Is)"
