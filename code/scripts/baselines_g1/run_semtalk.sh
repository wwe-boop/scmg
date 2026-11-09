#!/bin/bash
# SemTalk (Speaker-2 weights) on ECHO-G val_common -> SMPL-X -> GMR -> G1 39-d -> official evaluator, on the 5090 host.
#   run_semtalk.sh [stem list]          env GPUS="0 1" (one process per GPU), NW=<GMR CPU workers>
# Outputs under $W (default ~/bulk/baselines-g1-1105/semtalk): smplx_seed0/<stem>.npz, g1_seed0/<stem>.pt,
# benchmark_g1_semtalk_seed0.json.  GMR conversion = gmr_convert.py (unmodified EMAGE/GestureLSM converter).
set -euo pipefail
D=$(cd "$(dirname "$0")" && pwd)
SPLIT=${1:-/mnt/bulk/zehan/echog-dataset/official-meta/splits/val_common.txt}
W=${W:-$HOME/bulk/baselines-g1-1105/semtalk}; mkdir -p $W/logs
ST=${SEMTALK_ROOT:-$HOME/bulk/semtalk-1105}
log() { echo "$(date -Is) $*" | tee -a $W/run.log; }
GL=(${GPUS:-0 1}); n=${#GL[@]}
log "SEMTALK_START gpus=${GL[*]} split=$SPLIT"
for i in $(seq 0 $((n - 1))); do
  CUDA_VISIBLE_DEVICES=${GL[$i]} OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 MODELSCOPE_OFFLINE=1 \
    PYTHONDONTWRITEBYTECODE=1 nice -n 5 $ST/env/bin/python $D/semtalk_infer.py --stems $SPLIT --out $W/smplx_seed0 \
    --shard $i/$n > $W/logs/infer_$i.log 2>&1 &
done; wait
log "SEMTALK_DONE $(ls $W/smplx_seed0 | wc -l) npz"
for i in $(seq 0 $((${NW:-8} - 1))); do
  env CUDA_VISIBLE_DEVICES="" MUJOCO_GL=egl OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 nice -n 10 $ST/gmrenv/bin/python \
    $D/gmr_convert.py --assets ${ASSETS:-$HOME/bulk/baselines-g1-1105/assets} -- --src $W/smplx_seed0 --out $W/g1_seed0 \
    --stems $SPLIT --shard $i/${NW:-8} --generator "semtalk speaker2 seed0" > $W/logs/gmr_$i.log 2>&1 &
done; wait
log "GMR_DONE $(ls $W/g1_seed0 | wc -l) pt"
$D/eval_official.sh $W/g1_seed0 $SPLIT semtalk+gmr $W/benchmark_g1_semtalk_seed0.json > $W/logs/eval.log 2>&1 && log EVAL_DONE || log EVAL_FAILED
