#!/usr/bin/env bash
set -euo pipefail
cd /home/xujian/research/My-Paper/baselines/FoundDiff_MedCNN_8patient_Aug_8k
exec 9>RUN.lock
flock -n 9 || exit 73
out=experiments/MedCNN_8patient_Aug_8k
test -d "$out/checkpoints"
test ! -e "$out/train.jsonl"
test ! -e "$out/checkpoints/step_0.pt"
exec > >(tee -a queue.log) 2>&1
trap 'code=$?; if [ "$code" -ne 0 ]; then printf "FAILED exit=%s time=%s\n" "$code" "$(date --iso-8601=seconds)" >> QUEUE_FAILED.txt; fi' EXIT
export CUDA_VISIBLE_DEVICES=0 PYTHONDONTWRITEBYTECODE=1
CONDA=/home/xujian/miniforge3/bin/conda
printf "START %s\n" "$(date --iso-8601=seconds)"
"$CONDA" run --no-capture-output -n founddiff python -B -u "$out/train.py" > "$out/train.log" 2>&1
"$CONDA" run --no-capture-output -n founddiff python -B -u finalize_pair.py > "$out/finalize.log" 2>&1
printf "COMPLETE %s\n" "$(date --iso-8601=seconds)" > QUEUE_COMPLETE.txt
