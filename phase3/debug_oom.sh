#!/usr/bin/env bash
echo "=== mem ==="
free -g | sed -n 2p
echo "=== oom evidence ==="
dmesg 2>/dev/null | grep -i "killed process" | tail -3
echo "=== trainset progress ==="
tail -1 /home/ubuntu/phase3/make_trainset.log
ls /home/ubuntu/phase3/train_A0/videos | wc -l
ls /home/ubuntu/phase3/train_A1/videos | wc -l
ls /home/ubuntu/phase3/train_A2/videos 2>/dev/null | wc -l
ls /home/ubuntu/phase3/train_A3/videos 2>/dev/null | wc -l
