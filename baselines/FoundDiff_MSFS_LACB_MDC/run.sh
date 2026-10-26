#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
exec 9>RUN.lock
flock -n 9 || exit 73
test ! -e train.jsonl
exec > >(tee -a queue.log) 2>&1
trap 'code=$?; if [ "$code" -ne 0 ]; then printf "FAILED exit=%s time=%s\n" "$code" "$(date --iso-8601=seconds)" >> QUEUE_FAILED.txt; fi' EXIT
export CUDA_VISIBLE_DEVICES=0
CONDA_BIN=/home/xujian/miniconda3/bin/conda
printf "START %s\n" "$(date --iso-8601=seconds)"
"$CONDA_BIN" run --no-capture-output -n flexict python -u train.py > train.log 2>&1
"$CONDA_BIN" run --no-capture-output -n flexict python -u finalize.py > finalize.log 2>&1
printf "COMPLETE %s\n" "$(date --iso-8601=seconds)" > QUEUE_COMPLETE.txt
