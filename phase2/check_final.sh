#!/usr/bin/env bash
echo "=== seedance ==="
bash /home/ubuntu/check_seed.sh | tail -3
echo "=== A3 eval progress ==="
tr "\r" "\n" < /home/ubuntu/phase3/eval_A3.log | grep -oE "^\[[0-9]+/154\]" | tail -1
if [ -f /home/ubuntu/phase3/eval_A3/summary.txt ]; then
  echo "=== A3 RESULTS ==="
  cat /home/ubuntu/phase3/eval_A3/summary.txt
fi
