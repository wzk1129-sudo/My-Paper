#!/usr/bin/env bash
set -euo pipefail
cd /home/xujian/projects/CoreDiff
worktree=/home/xujian/projects/RDDM_CVPR2024/experiments/prosac_apgc_reproduction_fast_v1_20260930
pkg="$worktree/experiments/prosac_apgc_fast_v1"
python_bin=/home/xujian/miniconda3/envs/flexict/bin/python
cd "$worktree"
test "$(git branch --show-current)" = experiment/prosac-apgc-reproduction-fast-v1
test ! -e "$pkg/runs/M0_LOSS111"
test ! -e "$pkg/runs/M1_APGC_LOSS111"
"$python_bin" - "$pkg" <<'PY'
import json,sys
from pathlib import Path
p=Path(sys.argv[1])/"sanity100"
a=json.loads((p/"M0_LOSS111"/"SUMMARY.json").read_text())
b=json.loads((p/"M1_APGC_LOSS111"/"SUMMARY.json").read_text())
assert a["status"]==b["status"]=="PASS"
assert a["optimizer_steps"]==b["optimizer_steps"]==100
assert a["train_stream_sha256"]==b["train_stream_sha256"]
for d in (a,b):
    assert d["peak_reserved_gib"] < 24
print("SANITY_GATE_PASS", flush=True)
PY
{
  echo "session=prosac_apgc_loss111_fast8k_20260930"
  echo "started=$(date -Is)"
  echo "host=$(hostname)"
  echo "user=$(whoami)"
  echo "worktree=$worktree"
  echo "branch=$(git branch --show-current)"
  echo "head=$(git rev-parse HEAD)"
  echo "environment=$python_bin"
  echo "gpu=0"
  echo "log=$pkg/formal_loss111_queue.log"
  echo "output=$pkg/runs/M0_LOSS111,$pkg/runs/M1_APGC_LOSS111"
  echo "commands=train_matched.py --arm M0_LOSS111 --max-step 8000; train_matched.py --arm M1_APGC_LOSS111 --max-step 8000"
  echo "exit_condition=both arms independently finish step 8000 and 2k,4k,8k validation; stop on first failure"
  echo "workspace_status:"
  git status --short
  echo "source_sha256:"
  sha256sum "$pkg/loss111.py" "$pkg/train_matched.py" "$pkg/apgc_model.py" "$pkg/CONFIG_M0_LOSS111.json" "$pkg/CONFIG_M1_APGC_LOSS111.json" "$worktree/experiments/7_RDDM_CT_Backbone_v1/rddm_ct.py"
  echo "environment_versions:"
  "$python_bin" -c 'import sys, torch; print(sys.version.split()[0],torch.__version__,torch.version.cuda)'
} > "$pkg/FORMAL_LOSS111_LAUNCH.txt"
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0
status="$pkg/FORMAL_LOSS111_STATUS.txt"
trap 'echo "FAILED $(date -Is)" > "$status"' ERR
echo "RUNNING_M0 $(date -Is)" > "$status"
"$python_bin" "$pkg/train_matched.py" --arm M0_LOSS111 --max-step 8000 > "$pkg/formal_M0.log" 2>&1
echo "RUNNING_M1 $(date -Is)" > "$status"
"$python_bin" "$pkg/train_matched.py" --arm M1_APGC_LOSS111 --max-step 8000 > "$pkg/formal_M1.log" 2>&1
echo "COMPLETE $(date -Is)" > "$status"
echo FORMAL_LOSS111_COMPLETE
