#!/usr/bin/env bash
cd /home/ubuntu/phase2_bundle/phase2
export HF_HOME=/home/ubuntu/hf_cache
setsid nohup /home/ubuntu/miniforge3/bin/python3 gen_closed.py \
  --endpoint bytedance/seedance-2.0/text-to-video --name seedance20 \
  --subset closed-signal --gt gt_curves.csv --out gen_closed \
  > gen_closed.log 2>&1 < /dev/null &
sleep 5
echo STARTED
