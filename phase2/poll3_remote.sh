#!/usr/bin/env bash
echo "=== GPU ==="
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader
echo "=== gen mp4 counts ==="
for m in cogvideox2b ltxvideo wan13b; do
  n=$(find ~/phase2_bundle/phase2/gen_out/$m -name "*.mp4" 2>/dev/null | wc -l)
  echo "$m: $n"
done
echo "=== run_all2 tail ==="
tail -1 ~/phase2_bundle/phase2/run_all2.log | cut -c1-100
echo "=== trainset ==="
tail -1 ~/phase3/make_trainset.log
find ~/phase3 -name "*.mp4" 2>/dev/null | wc -l
echo "=== eval logs ==="
for e in ~/phase2_bundle/phase2/eval.log; do tail -2 "$e" 2>/dev/null; done
