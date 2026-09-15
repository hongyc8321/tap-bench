#!/usr/bin/env bash
cd /home/ubuntu/phase3
export HF_HOME=/home/ubuntu/hf_cache
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_DISABLE_XET=1
nohup bash train_and_eval.sh A0 > phase3_A0.log 2>&1 < /dev/null &
sleep 10
tail -5 phase3_A0.log
