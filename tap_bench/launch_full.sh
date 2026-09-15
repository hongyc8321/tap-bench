#!/usr/bin/env bash
cd /home/rhap/nkst/tap_bench
setsid nohup ../.venv/bin/python render_probe.py --grid full --out probe_set --workers 8 \
  > full_render.log 2>&1 < /dev/null &
echo "launched pid $!"
