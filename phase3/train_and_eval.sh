#!/usr/bin/env bash
# Phase 3: LoRA fine-tune CogVideoX-2B on arm A0 / A3 and evaluate both.
# Usage: nohup bash train_and_eval.sh > phase3.log 2>&1 &
set -e
cd "$(dirname "$0")"
export HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com}
export HF_HUB_DISABLE_XET=1
export HF_HOME=~/hf_cache

ARM=${1:-A0}
DATA=train_${ARM}
OUT=lora_${ARM}

if [ ! -d diffusers_repo ]; then
  git clone --depth 1 https://github.com/huggingface/diffusers.git diffusers_repo
fi
PY=${PY:-$HOME/miniforge3/bin/python3}
"$PY" -m pip install -q -i https://pypi.tuna.tsinghua.edu.cn/simple peft

"$PY" diffusers_repo/examples/cogvideo/train_cogvideox_lora.py \
  --pretrained_model_name_or_path THUDM/CogVideoX-2b \
  --instance_data_root "$DATA" \
  --caption_column prompts.txt \
  --video_column videos.txt \
  --output_dir "$OUT" \
  --height 480 --width 720 --max_num_frames 49 --fps 8 \
  --rank 64 --lora_alpha 128 \
  --learning_rate 1e-4 --max_train_steps 2000 \
  --train_batch_size 1 --gradient_accumulation_steps 4 \
  --mixed_precision fp16 --gradient_checkpointing \
  --checkpointing_steps 500 --checkpoints_total_limit 2 \
  --validation_prompt "a grayscale wheel with 8 spokes rotating counterclockwise at exactly 2 revolutions per second, static camera, sharp focus, no motion blur, white background" \
  --validation_epochs 500 \
  --seed 42

echo "[$ARM] training done -> $OUT"
