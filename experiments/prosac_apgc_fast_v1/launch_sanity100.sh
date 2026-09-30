#!/usr/bin/env bash
set -euo pipefail
cd /home/xujian/projects/CoreDiff
worktree=/home/xujian/projects/RDDM_CVPR2024/experiments/prosac_apgc_reproduction_fast_v1_20260930
pkg="$worktree/experiments/prosac_apgc_fast_v1"
python_bin=/home/xujian/miniconda3/envs/flexict/bin/python
cd "$worktree"
test "$(git branch --show-current)" = experiment/prosac-apgc-reproduction-fast-v1
test ! -e "$pkg/sanity100/M0_LOSS111"
test ! -e "$pkg/sanity100/M1_APGC_LOSS111"
{
  echo "session=prosac_apgc_loss111_sanity_20260930"
  echo "started=$(date -Is)"
  echo "host=$(hostname)"
  echo "user=$(whoami)"
  echo "worktree=$worktree"
  echo "branch=$(git branch --show-current)"
  echo "head=$(git rev-parse HEAD)"
  echo "environment=$python_bin"
  echo "gpu=0"
  echo "log=$pkg/sanity100_queue.log"
  echo "output=$pkg/sanity100"
  echo "commands=python sanity100.py --arm M0_LOSS111; python sanity100.py --arm M1_APGC_LOSS111"
  echo "exit_condition=both independent 100 optimizer steps complete or first failure"
  echo "workspace_status:"
  git status --short
  echo "source_sha256:"
  sha256sum "$pkg/loss111.py" "$pkg/sanity100.py" "$pkg/train_matched.py" "$pkg/apgc_model.py" "$pkg/CONFIG_M0_LOSS111.json" "$pkg/CONFIG_M1_APGC_LOSS111.json"
  echo "environment_versions:"
  "$python_bin" -c 'import sys, torch; print(sys.version.split()[0],torch.__version__,torch.version.cuda)'
} > "$pkg/SANITY100_LAUNCH.txt"
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0
"$python_bin" "$pkg/sanity100.py" --arm M0_LOSS111 > "$pkg/sanity100_M0.log" 2>&1
"$python_bin" "$pkg/sanity100.py" --arm M1_APGC_LOSS111 > "$pkg/sanity100_M1.log" 2>&1
echo SANITY100_COMPLETE
