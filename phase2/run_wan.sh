#!/usr/bin/env bash
cd /home/ubuntu/phase2_bundle/phase2
export HF_HOME=/home/ubuntu/hf_cache
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_DISABLE_XET=1
nohup /home/ubuntu/miniforge3/bin/python3 generate_probe.py --model wan13b \
  --subset curve --gt gt_curves.csv > gen_wan13b.log 2>&1 < /dev/null &
sleep 5
echo started
