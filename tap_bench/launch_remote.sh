#!/usr/bin/env bash
set -e
cd ~/phase2_bundle/phase2
export HF_HOME=~/hf_cache
nohup bash run_all.sh cogvideox2b > run_all.log 2>&1 < /dev/null &
sleep 8
echo "--- run_all.log head ---"
head -20 run_all.log
