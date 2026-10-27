#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
exec 9>SERIAL.lock
flock -n 9 || exit 73
test -f PAIRED_PREFLIGHT.json
test ! -e experiments/Fast2P_TimeOnly_Control_8k/train.jsonl
test ! -e experiments/Fast2P_TimeOnly_MedCNN_8k/train.jsonl
exec > >(tee -a serial.log) 2>&1
trap 'code=$?; if [ "$code" -ne 0 ]; then printf "FAILED exit=%s time=%s\n" "$code" "$(date --iso-8601=seconds)" >> SERIAL_FAILED.txt; fi' EXIT
export CUDA_VISIBLE_DEVICES=0
CONDA_BIN=/home/xujian/miniforge3/bin/conda
printf "START %s\n" "$(date --iso-8601=seconds)"
for name in Fast2P_TimeOnly_Control_8k Fast2P_TimeOnly_MedCNN_8k; do
  printf "RUN %s %s\n" "$name" "$(date --iso-8601=seconds)"
  (cd "experiments/$name" && "$CONDA_BIN" run --no-capture-output -n founddiff python -u train.py > train.log 2>&1)
  "$CONDA_BIN" run --no-capture-output -n founddiff python -u finalize_pair.py "$name" > "experiments/$name/finalize.log" 2>&1
done
"$CONDA_BIN" run --no-capture-output -n founddiff python -u finalize_pair.py > final_comparison.log 2>&1
printf "COMPLETE %s\n" "$(date --iso-8601=seconds)" > SERIAL_COMPLETE.txt
