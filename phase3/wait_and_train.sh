#!/usr/bin/env bash
while ! grep -q "\[A3\] done" /home/ubuntu/phase3/make_trainset.log; do
  sleep 60
done
sleep 30
cd /home/ubuntu/phase3
export HF_HOME=/home/ubuntu/hf_cache
setsid nohup bash train_and_eval.sh A0 > phase3_A0.log 2>&1 < /dev/null &
echo "A0 training relaunched at $(date)"
