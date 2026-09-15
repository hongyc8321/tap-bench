#!/usr/bin/env bash
cd /home/ubuntu/phase2_bundle/phase2
export HF_HOME=/home/ubuntu/hf_cache
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_DISABLE_XET=1
nohup /home/ubuntu/miniforge3/bin/python3 generate_probe.py --model ltxvideo \
  --mode i2v --subset i2vset --gt gt_curves.csv \
  > gen_ltxvideo_i2v.log 2>&1 < /dev/null &
sleep 5
echo started
