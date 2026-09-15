#!/usr/bin/env bash
cd /home/ubuntu/phase3
export HF_HOME=/home/ubuntu/hf_cache
setsid nohup bash train_and_eval.sh A1 > phase3_A1.log 2>&1 < /dev/null &
sleep 10
setsid nohup bash /home/ubuntu/chain_a1a2.sh > /home/ubuntu/chain_a1a2.log 2>&1 < /dev/null &
sleep 5
echo "A1 relaunched + chain armed"
tail -2 phase3_A1.log | cut -c1-70
