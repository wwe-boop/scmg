#!/bin/bash
# 11-03 Claude: one SEMAUX lane on our own GPUs (quick validation while the DLC job queues). Same train/eval/semantic steps as
# pipeline_dlc.sh slane(), smaller batch (BATCH, default 32) and STEPS (default 16000).
#   local_lane.sh <gpu> <out root> <lane> <cfg-w> <cfg-json> <sc-json> [extra train args]
set -uo pipefail
gpu=$1; R=$2; o=$3; w=$4; cf=$5; scj=$6; shift 6
CODE=$(cd "$(dirname "$0")/.." && pwd); cd $CODE; export PYTHONPATH=$CODE USE_TF=0 OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
PY=${PY:-python}; B=/mnt/cpfs/zbl-cpfs-new/CKPT/zehan/gesture-scmg; D=/mnt/cpfs/zbl-cpfs-new/dataset/zehan/gesture/scmg
VAE=$B/trained/vae-r3/final.pt; TOKC=$D/tok-qwen25-v36_1; IFD=$B/models/intent-frames-v36_1
SEMMAN=$B/semantic-recognizer-v1/manifest-n200; ST=${STEPS:-16000}
mkdir -p $R/$o; export CUDA_VISIBLE_DEVICES=$gpu
if [ ! -f $R/$o/DONE.json ]; then
  $PY -m scmg.train --out $R/$o --stats $D/STATS_v36_1.json --text-cache $D/text-e5-v36_1 --vae $VAE --workers ${WORKERS:-6} \
    --tok-cache $TOKC --batch ${BATCH:-32} --stage base --steps $ST --lr 3e-4 --warmup $((ST / 8)) --eval-every $((ST / 16)) \
    --snap-every 0 --cfg "$cf" --sc "$scj" --intent-frames $IFD --holdout $SEMMAN/MANIFEST.json "$@" >> $R/$o/log.txt 2>&1 \
    || { echo "$(date -Is) TRAIN FAILED $o"; exit 1; }
fi
ev() { local tag=$1; shift; [ -f $R/$o/eval-best$tag/RESULTS.json ] || $PY -m scmg.evaluate --ckpt $R/$o/best.pt --vae $VAE \
  --stats $D/STATS_v36_1.json --text-cache $D/text-e5-v36_1 --tok-cache $TOKC --cfg-w $w --out $R/$o/eval-best$tag --n 43 \
  --horizons inf,0 --ndiv 20 --mmap "$@" >> $R/$o/eval.log 2>&1; }
so=$R/$o/semantic
if [ ! -f $so/DONE ]; then
  $PY scripts/gen_semantic.py $SEMMAN $so $o=$R/$o/best.pt@$w --variants own,notext,swap,gnull >> $R/$o/semantic.log 2>&1 && \
  for vv in own notext swap gnull; do [ -d $so/$o/$vv ] || continue
    $PY scripts/semantic_eval.py score --gen-dir $so/$o/$vv --manifest $SEMMAN/MANIFEST.json --out $so/score-$vv.json >> $R/$o/semantic.log 2>&1
  done && touch $so/DONE
fi
SCMG_SAVE_N=43 ev ""
SCMG_SAVE_N=43 SCMG_SEED0=1 ev -s1
[[ "$cf" != *voxcpm* ]] || SCMG_SAVE_N=43 ev -win --window-lm-text
rm -f $R/$o/resume.pt $R/$o/final.pt
echo "$(date -Is) LANE_DONE $o"
