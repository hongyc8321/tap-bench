#!/usr/bin/env bash
set -e
rm -rf /home/ubuntu/phase3/train_A0 /home/ubuntu/phase3/train_A1 \
       /home/ubuntu/phase3/train_A2 /home/ubuntu/phase3/train_A3
/home/ubuntu/miniforge3/bin/pip install -q -i https://pypi.tuna.tsinghua.edu.cn/simple peft 2>&1 | tail -1
cd /home/ubuntu/phase3
export HF_HOME=/home/ubuntu/hf_cache
setsid nohup /home/ubuntu/miniforge3/bin/python3 make_trainset.py --all --n 500 \
  --out-root . --workers 8 > make_trainset.log 2>&1 < /dev/null &
sleep 8
tail -2 make_trainset.log
