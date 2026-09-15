#!/usr/bin/env bash
cd ~/phase3
export HF_HOME=~/hf_cache
PY=${PY:-$HOME/miniforge3/bin/python3}
nohup $PY make_trainset.py --all --n 500 --out-root . --workers 8 \
  > make_trainset.log 2>&1 < /dev/null &
sleep 10
tail -3 make_trainset.log
