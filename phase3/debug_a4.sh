#!/usr/bin/env bash
cd /home/ubuntu/phase3
/home/ubuntu/miniforge3/bin/python3 make_trainset_a4.py --a0-root train_A0 \
  --out-root a4_dry --limit 2 --workers 1 2>&1 | grep -E "in (render_shutter|encode_clip|main)" | head -6
echo "=== tmp contents ==="
find a4_dry -type f | head -6
