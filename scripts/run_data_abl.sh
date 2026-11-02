#!/bin/bash
# Data ablation lane (chunk planner, base stage, single-GPU budget = voxabl chunk-planner control).
#   run_data_abl.sh <code dir> <out root> <lane> <gpu> [extra train args...]   env: PY, SCMG_LOCAL_MIRROR, SCMG_NO_GLITCH
set -uo pipefail
CODE=$1; R=$2; o=$3; gpu=$4; shift 4
PY=${PY:-python}; D=/mnt/cpfs/zbl-cpfs-new/dataset/zehan/gesture/scmg
VAE=/mnt/cpfs/zbl-cpfs-new/CKPT/zehan/gesture-scmg/trained/vae-r3/final.pt
cd $CODE && export PYTHONPATH=$CODE${PYTHONPATH:+:$PYTHONPATH}
mkdir -p $R/$o
C="--stats $D/STATS_v36_1.json --text-cache $D/text-e5-v36_1 --vae $VAE --workers 6"
[ -f $R/$o/DONE.json ] || CUDA_VISIBLE_DEVICES=$gpu $PY -m scmg.train --stage base --out $R/$o $C --batch 128 --steps 8000 --lr 3e-4 \
  --warmup 1000 --eval-every 500 --snap-every 0 "$@" >> $R/$o/log.txt 2>&1 || { echo FAILED; touch $R/$o/FAILED; exit 1; }
CUDA_VISIBLE_DEVICES=$gpu $PY -m scmg.evaluate --ckpt $R/$o/best.pt --vae $VAE --stats $D/STATS_v36_1.json --text-cache $D/text-e5-v36_1 \
  --cfg-w 2.5 --out $R/$o/eval-best --n 43 --horizons inf,0 >> $R/$o/eval.log 2>&1 && touch $R/$o/EVAL_DONE
