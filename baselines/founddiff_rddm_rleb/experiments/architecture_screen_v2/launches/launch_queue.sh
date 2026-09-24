#!/bin/bash
set -euo pipefail
cd /home/xujian/projects/CoreDiff/experiments/rddm_architecture_screen_v2_20260924/baselines/founddiff_rddm_rleb
exec /home/xujian/miniconda3/envs/corediff/bin/python -u -m architecture_screen_v2.queue >> /home/xujian/projects/CoreDiff/experiments/rddm_architecture_screen_v2_20260924/baselines/founddiff_rddm_rleb/experiments/architecture_screen_v2/launches/formal_queue.log 2>&1
