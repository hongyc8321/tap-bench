#!/usr/bin/env bash
cd /home/rhap/nkst/tap_bench/probe_set
for c in wheel grating ball; do
  echo -n "$c meta: "
  find $c -name meta.json | wc -l
done
echo "--- incomplete clips (frames != 120):"
for d in $(find . -name meta.json -exec dirname {} \;); do
  n=$(ls $d/frame_*.png 2>/dev/null | wc -l)
  if [ "$n" -ne 120 ]; then echo "$d $n"; fi
done
