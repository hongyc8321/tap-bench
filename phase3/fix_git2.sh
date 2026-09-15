#!/usr/bin/env bash
cd /home/ubuntu/phase3
mkdir -p diffusers_repo/examples/cogvideo
BASES=(
  "https://cdn.jsdelivr.net/gh/huggingface/diffusers@v0.34.0/examples/cogvideo"
  "https://raw.githubusercontent.com/huggingface/diffusers/v0.34.0/examples/cogvideo"
  "https://ghfast.top/https://raw.githubusercontent.com/huggingface/diffusers/v0.34.0/examples/cogvideo"
)
for base in "${BASES[@]}"; do
  for f in train_cogvideox_lora.py requirements.txt; do
    curl -fsSL --max-time 60 "$base/$f" -o "diffusers_repo/examples/cogvideo/$f" 2>/dev/null \
      && echo "OK $f from $base" || echo "FAIL $f from $base"
  done
  if [ -s diffusers_repo/examples/cogvideo/train_cogvideox_lora.py ]; then
    echo "SOURCE=$base"
    break
  fi
done
ls -la diffusers_repo/examples/cogvideo/
head -30 diffusers_repo/examples/cogvideo/train_cogvideox_lora.py 2>/dev/null | grep -E "arg|import" | head -10
