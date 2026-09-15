#!/usr/bin/env bash
# CW control: base model (no LoRA) on the 18 CW probe cells
export HF_HOME=/home/ubuntu/hf_cache
cd /home/ubuntu/phase3
PY=/home/ubuntu/miniforge3/bin/python3
P2=/home/ubuntu/phase2_bundle/phase2

echo "$(date) CW gen ORIG"
$PY $P2/generate_probe.py --model cogvideox2b --subset full \
  --gt gt_cw/gt_curves.csv --out gen_cw_ORIG > gen_cw_ORIG.log 2>&1
echo "$(date) CW eval ORIG"
$PY eval_cw.py --gt gt_cw/gt_curves.csv --gen-root gen_cw_ORIG --out cw_eval_ORIG > cw_eval_ORIG.log 2>&1
echo "$(date) CW ORIG DONE"
