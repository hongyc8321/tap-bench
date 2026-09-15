#!/usr/bin/env bash
cd ~/phase2_bundle/phase2
export HF_HOME=~/hf_cache
nohup bash run_all.sh ltxvideo wan13b > run_all2.log 2>&1 < /dev/null &
sleep 5
head -3 run_all2.log
