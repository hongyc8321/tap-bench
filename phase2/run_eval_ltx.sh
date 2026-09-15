#!/usr/bin/env bash
cd /home/ubuntu/phase2_bundle/phase2
export HF_HOME=/home/ubuntu/hf_cache
setsid nohup /home/ubuntu/miniforge3/bin/python3 eval_gen.py --gen-root gen_out \
  --gt gt_curves.csv --out eval_ltx > eval_ltx.log 2>&1 < /dev/null &
sleep 5
echo started
