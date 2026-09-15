#!/usr/bin/env bash
export HF_HOME=/home/ubuntu/hf_cache
cd /home/ubuntu/phase3

while [ ! -f lora_A1/pytorch_lora_weights.safetensors ]; do sleep 120; done
while pgrep -f train_cogvideox_lora > /dev/null; do sleep 60; done
sleep 30
echo "$(date) A1 weights ready -> eval"
bash eval_arm.sh A1 > eval_A1.log 2>&1
echo "$(date) A1 eval done"

while [ ! -f eval_A1/summary.txt ]; do sleep 60; done
echo "$(date) launching A2 training"
bash train_and_eval.sh A2 > phase3_A2.log 2>&1
echo "$(date) A2 training done -> eval"
bash eval_arm.sh A2 > eval_A2.log 2>&1
echo "$(date) A2 eval done; chain complete"
