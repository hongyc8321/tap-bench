#!/usr/bin/env bash
# CW-only rerun: generation (A0, A5) + compliance eval
export HF_HOME=/home/ubuntu/hf_cache
cd /home/ubuntu/phase3
PY=/home/ubuntu/miniforge3/bin/python3
P2=/home/ubuntu/phase2_bundle/phase2

echo "$(date) CW gen A0"
$PY $P2/generate_probe.py --model cogvideox2b --subset full \
  --gt gt_cw/gt_curves.csv --out gen_cw_A0 --lora lora_A0 > gen_cw_A0.log 2>&1
echo "$(date) CW gen A5"
$PY $P2/generate_probe.py --model cogvideox2b --subset full \
  --gt gt_cw/gt_curves.csv --out gen_cw_A5 --lora lora_A5 > gen_cw_A5.log 2>&1
echo "$(date) CW eval"
$PY eval_cw.py --gt gt_cw/gt_curves.csv --gen-root gen_cw_A0 --out cw_eval_A0 > cw_eval_A0.log 2>&1
$PY eval_cw.py --gt gt_cw/gt_curves.csv --gen-root gen_cw_A5 --out cw_eval_A5 > cw_eval_A5.log 2>&1
echo "$(date) CW ALL DONE"
