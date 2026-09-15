#!/usr/bin/env bash
# A5 orchestrator: trainset -> train -> curve eval -> CW capability probes.
export HF_HOME=/home/ubuntu/hf_cache
cd /home/ubuntu/phase3
PY=/home/ubuntu/miniforge3/bin/python3
P2=/home/ubuntu/phase2_bundle/phase2

echo "$(date) rendering A5 trainset (copy ccw + 500 cw twins)"
$PY make_trainset_a5.py > make_a5.log 2>&1
echo "$(date) A5 data ready -> training"
bash train_and_eval.sh A5 > phase3_A5.log 2>&1
echo "$(date) A5 trained -> curve eval"
bash eval_arm.sh A5 > eval_A5.log 2>&1
echo "$(date) A5 curve done -> CW generation"
$PY $P2/generate_probe.py --model cogvideox2b --subset full \
  --gt gt_cw/gt_curves.csv --out gen_cw_A0 --lora lora_A0 > gen_cw_A0.log 2>&1
$PY $P2/generate_probe.py --model cogvideox2b --subset full \
  --gt gt_cw/gt_curves.csv --out gen_cw_A5 --lora lora_A5 > gen_cw_A5.log 2>&1
echo "$(date) CW generation done -> CW eval"
$PY eval_cw.py --gt gt_cw/gt_curves.csv --gen-root gen_cw_A0 --out cw_eval_A0 > cw_eval_A0.log 2>&1
$PY eval_cw.py --gt gt_cw/gt_curves.csv --gen-root gen_cw_A5 --out cw_eval_A5 > cw_eval_A5.log 2>&1
echo "$(date) A5 ALL DONE"
