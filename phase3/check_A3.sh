#!/usr/bin/env bash
L=/home/ubuntu/phase3/phase3_A3.log
echo "=== weights ==="
ls -la /home/ubuntu/phase3/lora_A3/pytorch_lora_weights.safetensors 2>/dev/null || echo NOT_YET
echo "=== process ==="
pgrep -af train_cogvideox_lora | head -2
echo "=== last lines ==="
tr "\r" "\n" < "$L" | grep -v "^\s*$" | tail -2 | cut -c1-110
echo "=== gpu ==="
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader
