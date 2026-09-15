#!/usr/bin/env bash
echo "=== raw log tail ==="
tail -c 3000 /home/ubuntu/phase3/phase3_A0.log | sed "s/$(printf '\r')/$(printf '\n')/g" | grep -v "^$" | tail -6
echo "=== Killed marks ==="
grep -ac "Killed" /home/ubuntu/phase3/phase3_A0.log
echo "=== kernel oom ==="
journalctl -k --no-pager 2>/dev/null | grep -iE "out of memory|oom-kill" | tail -4
grep -aiE "oom|killed process" /var/log/kern.log 2>/dev/null | tail -4
echo "=== mem/swap ==="
free -m | sed -n 2p
swapon --show
echo "=== watcher log ==="
cat /home/ubuntu/wait_and_train.log
