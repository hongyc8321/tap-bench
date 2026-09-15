#!/usr/bin/env bash
cd ~/phase2_bundle/phase2
n=$(find gen_out -name "*.mp4" 2>/dev/null | wc -l)
last=$(grep -o "\[[0-9]*/154\]" gen_*.log 2>/dev/null | tail -1)
running=$(pgrep -fc generate_probe || echo 0)
echo "mp4=$n $last running=$running"
if [ -f eval.log ]; then tail -3 eval.log; fi
