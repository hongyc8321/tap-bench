#!/usr/bin/env bash
cd /home/ubuntu/phase3
setsid nohup /home/ubuntu/miniforge3/bin/python3 make_trainset_a4.py \
  --a0-root train_A0 --out-root . --workers 8 \
  > make_a4.log 2>&1 < /dev/null &
sleep 12
tail -2 make_a4.log
