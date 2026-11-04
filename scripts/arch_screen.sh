#!/bin/bash
# 11-04 module screen (rhythm / capacity): the r4 base recipe FROM SCRATCH with one architecture change (8-GPU DDP, batch
# 128 per GPU, 8000 steps = exactly dlc-scmg-train-r4-1102/base), then the full tier of best.pt in the SAME job
# (offline + engine x FGD + probes, inference seeds 0/1, CFG 2.5), directly comparable with baseline-1104/r4-main.
#   arch_screen.sh <out root> <label> <cfg-json>
set -uo pipefail
R=$1; lab=$2; cf=$3; CODE=$(cd "$(dirname "$0")/.." && pwd); cd $CODE; PY=${PY:-python}
B=/mnt/cpfs/zbl-cpfs-new/CKPT/zehan/gesture-scmg; D=/mnt/cpfs/zbl-cpfs-new/dataset/zehan/gesture/scmg; VAE=$B/trained/vae-r3/final.pt
NGPU=$(nvidia-smi -L | wc -l); O=$R/$lab; mkdir -p $O
export USE_TF=0 OMP_NUM_THREADS=4 PYTHONPATH=$CODE
log() { echo "$(date -Is) $*" >> $O/arch.log; }
if [ ! -f $O/DONE.json ]; then
  log "TRAIN_START $cf on $NGPU GPUs"
  $PY -m torch.distributed.run --standalone --nproc_per_node=$NGPU -m scmg.train --stage base --out $O --stats $D/STATS_v36_1.json \
    --text-cache $D/text-e5-v36_1 --vae $VAE --workers ${WORKERS:-6} --cfg "$cf" --batch 128 --steps 8000 --lr 3e-4 --warmup 1000 \
    --eval-every 500 --snap-every 2000 >> $O/log.txt 2>&1
fi
[ -f $O/DONE.json ] || { log "TRAIN_FAILED"; exit 1; }
log "TRAIN_DONE"
g=0; for P in offline engine; do for part in fgd probes; do
  NOVOX=1 bash scripts/baseline_eval.sh $g $O/eval $lab $O/best.pt $P $part 2.5 > $O/eval-$P-$part.log 2>&1 || log "EVAL_FAILED $P $part" &
  g=$((g + 1)); done; done; wait
log "ARCH_DONE"
