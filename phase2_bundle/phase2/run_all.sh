#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"

export HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com}
export HF_HUB_ENABLE_HF_TRANSFER=1

PY=python3
if ! $PY -c "import torch" 2>/dev/null; then
  if [ ! -d .venv_cloud ]; then
    python3 -m venv .venv_cloud
  fi
  PY=.venv_cloud/bin/python
  $PY -m pip install -q torch --index-url https://download.pytorch.org/whl/cu124
fi
$PY -m pip install -q -r requirements.txt
echo "using PY=$PY ($($PY -c 'import torch; print(torch.__version__)'))"

MODELS=${@:-"cogvideox2b ltxvideo wan13b"}
for m in $MODELS; do
  echo "===== generating with $m ====="
  $PY generate_probe.py --model $m --subset curve --gt gt_curves.csv \
    2>&1 | tee gen_$m.log
done

echo "===== eval ====="
$PY eval_gen.py --gen-root gen_out --gt gt_curves.csv 2>&1 | tee eval.log
echo "results in eval_out/ (summary.txt, nyquist_curves.png, nyquist_bins.csv)"
