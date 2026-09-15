#!/usr/bin/env bash
me=$$
for pat in "run_all.sh" "generate_probe" "pip install"; do
  for pid in $(pgrep -f "$pat"); do
    [ "$pid" != "$me" ] && kill "$pid" 2>/dev/null
  done
done
sleep 3
pgrep -af "pip install" || echo "old pip gone"
bash /home/ubuntu/phase2_bundle/phase2/launch_remote.sh
