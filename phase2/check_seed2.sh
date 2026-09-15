#!/usr/bin/env bash
L=/home/ubuntu/phase2_bundle/phase2/gen_closed.log
echo "size: $(wc -c < $L)"
tr "\r" "\n" < "$L" | grep -v "^[[:space:]]*$" | tail -5 | cut -c1-110
echo "---"
ps -o pid,etime,stat,wchan:20 -p $(pgrep -f gen_closed.py | head -1) 2>/dev/null
