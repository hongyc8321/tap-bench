#!/usr/bin/env bash
L=/home/ubuntu/phase3/phase3_A0.log
echo "=== process ==="
pgrep -af "train_cogvideox_lora" | head -2
echo "=== last lines (CR to NL) ==="
tr "\r" "\n" < "$L" | grep -v "^\s*$" | tail -3 | cut -c1-120
echo "=== killed? ==="
grep -ac "Killed" "$L"
echo "=== gpu/mem ==="
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader
free -m | sed -n 2p
