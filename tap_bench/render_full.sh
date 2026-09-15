#!/usr/bin/env bash
cd /home/rhap/nkst/tap_bench
exec ../.venv/bin/python render_probe.py --grid full --out probe_set --workers 12 \
  > full_render.log 2>&1
