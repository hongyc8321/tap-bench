#!/usr/bin/env bash
echo "=== python procs ==="
ps -eo pid,pcpu,pmem,etime,cmd | grep -E "python" | grep -v grep | head -5
echo "=== log last visible progress ==="
tr '\r' '\n' < /home/ubuntu/phase3/phase3_A0.log | tail -3
echo "=== log growth check ==="
s1=$(wc -c < /home/ubuntu/phase3/phase3_A0.log)
sleep 15
s2=$(wc -c < /home/ubuntu/phase3/phase3_A0.log)
echo "bytes: $s1 -> $s2"
