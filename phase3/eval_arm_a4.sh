#!/usr/bin/env bash
# A4 evaluation: two conditioning modes (sharp / blur), same probe cells.
# Usage: bash eval_arm_a4.sh
set -e
cd "$(dirname "$0")"
export HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com}
export HF_HUB_DISABLE_XET=1
export HF_HOME=~/hf_cache

P2=/home/ubuntu/phase2_bundle/phase2
PY=${PY:-$HOME/miniforge3/bin/python3}

$PY $P2/generate_probe.py --model cogvideox2b --subset curve \
  --gt $P2/gt_curves.csv --out gen_A4 --lora lora_A4 \
  --prompt-append ", exposed with a 0-degree (instantaneous) shutter"
$PY $P2/eval_gen.py --gen-root gen_A4 --gt $P2/gt_curves.csv --out eval_A4

$PY $P2/generate_probe.py --model cogvideox2b --subset curve \
  --gt $P2/gt_curves.csv --out gen_A4blur --lora lora_A4 \
  --prompt-strip "no motion blur" \
  --prompt-append ", exposed with a 180-degree shutter causing motion blur"
$PY $P2/eval_gen.py --gen-root gen_A4blur --gt $P2/gt_curves.csv --out eval_A4blur

echo "[A4] curves: eval_A4/ + eval_A4blur/"
