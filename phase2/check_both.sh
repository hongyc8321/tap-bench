#!/usr/bin/env bash
echo "=== seedance gen ==="
grep -E "^\[[0-9]+/" /home/ubuntu/phase2_bundle/phase2/gen_closed.log | tail -3
grep -c FAILED /home/ubuntu/phase2_bundle/phase2/gen_closed.log
pgrep -cf gen_closed || echo 0
echo "=== A3 train ==="
tail -c 300 /home/ubuntu/phase3/phase3_A3.log | tr "\r" "\n" | tail -1 | cut -c1-72
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader
