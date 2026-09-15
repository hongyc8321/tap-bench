#!/usr/bin/env bash
L=/home/ubuntu/phase2_bundle/phase2/gen_closed.log
echo "=== recent ==="
grep -E "^\[[0-9]+/" "$L" | tail -5
echo "=== failures ==="
grep "FAILED" "$L"
echo "=== done count ==="
grep -c "fps=" "$L"
echo "=== alive ==="
pgrep -cf gen_closed.py || echo 0
