#!/bin/bash
# 11-04: BAgger (reverse, ONE round -- round 1 was the best on VoxCPM after the corrected evaluation) on the r4 main line.
#   bagger_r4.sh <out root>     (uses every visible GPU; env COUNT=1024 NGEN=15 CFGW=2.5 INIT SHARE=0.33 STEPS=6000)
# 1) collect COUNT reversed corrective trajectories with the INIT model, sharded over the GPUs (16-chunk rollouts so the
#    256-frame training window holds the whole reversed trajectory -- review fix); 2) train three lanes in parallel from
#    INIT on the same data: stream+intent+bagger, stream+bagger, bagger only; each lane validates while training (r4_lane.sh).
set -uo pipefail
R=$1; CODE=$(cd "$(dirname "$0")/.." && pwd); cd $CODE; PY=${PY:-python}
B=/mnt/cpfs/zbl-cpfs-new/CKPT/zehan/gesture-scmg; INIT=${INIT:-$B/dlc-scmg-train-r4-1102/base/best.pt}
export PYTHONPATH=$CODE${PYTHONPATH_EXTRA:+:$PYTHONPATH_EXTRA} USE_TF=0 OMP_NUM_THREADS=2
NG=$(nvidia-smi -L | wc -l); DATA=$R/bagger-data; mkdir -p $DATA
if [ ! -f $DATA/MANIFEST.json ]; then
  for g in $(seq 0 $((NG-1))); do
    CUDA_VISIBLE_DEVICES=$g $PY scripts/bagger_collect.py --ckpt $INIT --out $DATA --cond reverse --count ${COUNT:-1024} --n-gen ${NGEN:-15} \
      --seed 1000 --cfg-w ${CFGW:-2.5} --shard $g --shards $NG > $R/collect-$g.log 2>&1 &
  done; wait
  [ -f $DATA/MANIFEST.json ] || { echo "$(date -Is) COLLECT_FAILED"; exit 1; }
fi
echo "$(date -Is) COLLECT_DONE $(python3 -c "import json;print(len(json.load(open('$DATA/MANIFEST.json'))['rows']))")"
S='"p_stream": 0.67, "p_async": 1.0, "p_async_seg": 0.0'
export BAGGER_DIR=$DATA BAGGER_SHARE=${SHARE:-0.33}
SEMSHARE=0.6 bash scripts/r4_lane.sh 0 $R r4-stream-int-bag '{"adsc": true, "intent_aux": true}' "{$S, \"w_int\": 0.3, \"sem_w\": 2.0}" ${STEPS:-6000} > $R/lane-a.log 2>&1 &
bash scripts/r4_lane.sh $((1 % NG)) $R r4-stream-bag '{"adsc": true}' "{$S}" ${STEPS:-6000} > $R/lane-b.log 2>&1 &
bash scripts/r4_lane.sh $((2 % NG)) $R r4-bag '{}' '{}' ${STEPS:-6000} > $R/lane-c.log 2>&1 &
wait; echo "$(date -Is) BAGGER_R4_DONE"
