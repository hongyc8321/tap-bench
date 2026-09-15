#!/usr/bin/env bash
set -u
cd /home/rhap/nkst
echo "== pip packages =="
.venv/bin/pip list 2>/dev/null | grep -i -E 'numpy|pillow|opencv'
echo "== ffmpeg =="
command -v ffmpeg || echo NO_FFMPEG
echo "== cpu =="
nproc
