#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
exec 9>RUN.lock
flock -n 9 || exit 73
test ! -e train.jsonl
exec > >(tee -a queue.log) 2>&1
trap 'code=$?; if [ "$code" -ne 0 ]; then printf "FAILED exit=%s time=%s\n" "$code" "$(date --iso-8601=seconds)" >> QUEUE_FAILED.txt; fi' EXIT
printf "WAIT_FOR_8K %s\n" "$(date --iso-8601=seconds)"
ready=0
for ((attempt=0; attempt<1440; attempt++)); do
    if test -f ../QUEUE_FAILED.txt; then echo PARENT_FAILED; exit 70; fi
    if test -f ../QUEUE_COMPLETE.txt; then ready=1; break; fi
    sleep 60
done
test "$ready" = 1
test -f ../checkpoints/step_8000.pt
test -f ../FINAL_AUDIT.json
sleep 2
used=$(/usr/lib/wsl/lib/nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -n 1)
if (( used > 1024 )); then echo "GPU busy; continuation not started"; exit 71; fi
export CUDA_VISIBLE_DEVICES=0
CONDA_BIN=/home/xujian/miniconda3/bin/conda
printf "CONTINUE_FROM_8K %s\n" "$(date --iso-8601=seconds)"
"$CONDA_BIN" run --no-capture-output -n flexict python -u train_continue.py > train.log 2>&1
"$CONDA_BIN" run --no-capture-output -n flexict python -u finalize_extended.py > finalize.log 2>&1
printf "COMPLETE_16K %s\n" "$(date --iso-8601=seconds)" > QUEUE_COMPLETE.txt
