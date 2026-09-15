#!/usr/bin/env bash
# Evaluate a LoRA arm on the probe curve subset and produce its Nyquist curve.
# Usage: bash eval_arm.sh A0
set -e
cd "$(dirname "$0")"
export HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com}
export HF_HUB_DISABLE_XET=1
export HF_HOME=~/hf_cache

ARM=${1:-A0}
P2=/home/ubuntu/phase2_bundle/phase2
PY=${PY:-$HOME/miniforge3/bin/python3}
$PY $P2/generate_probe.py --model cogvideox2b --subset curve \
  --gt $P2/gt_curves.csv --out gen_${ARM} --lora lora_${ARM}
$PY $P2/eval_gen.py --gen-root gen_${ARM} \
  --gt $P2/gt_curves.csv --out eval_${ARM}
echo "[$ARM] curve: eval_${ARM}/summary.txt + nyquist_curves.png"
