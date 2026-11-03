#!/bin/bash
# 11-03: fine-tune lane from an existing checkpoint on our own GPUs, then probes + evals (text path / hold-fix validation).
#   ft_lane.sh <gpu> <out root> <lane> <init ckpt> <cfg-json (new switches)> <sc-json> [cfg-w=2.0]
# order: train -> planner_probe (text/audio reliance) -> semantic own/notext/swap (+score) -> rollout_probe -> FGD eval seed 0
set -uo pipefail
gpu=$1; R=$2; o=$3; init=$4; cf=$5; scj=$6; w=${7:-2.0}
CODE=$(cd "$(dirname "$0")/.." && pwd); cd $CODE; export PYTHONPATH=$CODE USE_TF=0 OMP_NUM_THREADS=${OMP_NUM_THREADS:-4} CUDA_VISIBLE_DEVICES=$gpu
PY=${PY:-python}; B=/mnt/cpfs/zbl-cpfs-new/CKPT/zehan/gesture-scmg; D=/mnt/cpfs/zbl-cpfs-new/dataset/zehan/gesture/scmg
VAE=$B/trained/vae-r3/final.pt; TOKC=$D/tok-qwen25-v36_1; SEMMAN=$B/semantic-recognizer-v1/manifest-n200; ST=${STEPS:-4000}
mkdir -p $R/$o
if [ ! -f $R/$o/DONE.json ]; then
  $PY -m scmg.train --out $R/$o --init $init --stats $D/STATS_v36_1.json --text-cache $D/text-e5-v36_1 --vae $VAE --workers ${WORKERS:-6} \
    --tok-cache $TOKC --tok-intent $B/models/intent-qwen25-v36_1 --intent-frames $B/models/intent-frames-v36_1 \
    --holdout $SEMMAN/MANIFEST.json --batch ${BATCH:-32} --stage base --steps $ST --lr ${LR:-1e-4} --warmup $((ST / 20)) \
    --eval-every $((ST / 8)) --snap-every 0 --cfg "$cf" --sc "$scj" >> $R/$o/log.txt 2>&1 || { echo "$(date -Is) TRAIN FAILED $o"; exit 1; }
  rm -f $R/$o/resume.pt
fi
[ -f $R/$o/planner_probe.json ] || $PY scripts/planner_probe.py $VAE $o=$R/$o/best.pt > $R/$o/planner_probe.json 2>> $R/$o/probe.log
so=$R/$o/semantic
if [ ! -f $so/DONE ]; then
  $PY scripts/gen_semantic.py $SEMMAN $so $o=$R/$o/best.pt@$w --variants own,notext,swap >> $R/$o/semantic.log 2>&1 && \
  for vv in own notext swap; do
    $PY scripts/semantic_eval.py score --gen-dir $so/$o/$vv --manifest $SEMMAN/MANIFEST.json --out $so/score-$vv.json >> $R/$o/semantic.log 2>&1
  done && touch $so/DONE
fi
[ -f $R/$o/rollout_probe.json ] || $PY scripts/rollout_probe.py $VAE $o=$R/$o/best.pt@$w > $R/$o/rollout_probe.json 2>> $R/$o/probe.log
[ -f $R/$o/eval-best/RESULTS.json ] || SCMG_SAVE_N=43 $PY -m scmg.evaluate --ckpt $R/$o/best.pt --vae $VAE --stats $D/STATS_v36_1.json \
  --text-cache $D/text-e5-v36_1 --tok-cache $TOKC --cfg-w $w --out $R/$o/eval-best --n 43 --horizons inf,0 --ndiv 20 --mmap >> $R/$o/eval.log 2>&1
echo "$(date -Is) LANE_DONE $o"
