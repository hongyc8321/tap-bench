#!/usr/bin/env bash
cd /home/ubuntu/phase3
rm -rf diffusers_repo

if git clone --depth 1 --config http.version=HTTP/1.1 \
     https://github.com/huggingface/diffusers.git diffusers_repo 2>/dev/null; then
  echo "VIA github direct"
elif curl -fsSL --max-time 300 \
     "https://mirror.ghproxy.com/https://github.com/huggingface/diffusers/archive/refs/heads/main.tar.gz" \
     -o diffusers.tar.gz 2>/dev/null; then
  mkdir -p diffusers_repo && tar xzf diffusers.tar.gz -C diffusers_repo --strip-components=1
  echo "VIA ghproxy"
elif curl -fsSL --max-time 300 \
     "https://gitee.com/mirrors/huggingface_diffusers/repository/archive/main.tar.gz" \
     -o diffusers.tar.gz 2>/dev/null; then
  mkdir -p diffusers_repo && tar xzf diffusers.tar.gz -C diffusers_repo --strip-components=1
  echo "VIA gitee"
else
  echo "ALL_FAILED"
  exit 1
fi
ls diffusers_repo/examples/cogvideo/ | head -5
