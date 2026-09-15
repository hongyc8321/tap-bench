#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"

export HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com}
export HF_HUB_ENABLE_HF_TRANSFER=1
export HF_HUB_DISABLE_XET=1

PY=""
for cand in "$CONDA_PREFIX/bin/python" "$HOME/miniforge3/bin/python" \
            "$HOME/miniconda3/bin/python" "$HOME/anaconda3/bin/python" \
            /opt/conda/bin/python python3; do
  if "$cand" -c "import torch" 2>/dev/null; then PY="$cand"; break; fi
done
if [ -z "$PY" ]; then
  if [ ! -d .venv_cloud ]; then
    python3 -m venv .venv_cloud
  fi
  PY="$PWD/.venv_cloud/bin/python"
  $PY -m pip install -q torch --index-url https://download.pytorch.org/whl/cu124
fi
PIP_INDEX=${PIP_INDEX:-https://pypi.tuna.tsinghua.edu.cn/simple}
$PY -m pip install -q -i "$PIP_INDEX" -r requirements.txt
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
