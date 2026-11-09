#!/bin/bash
# 11-04: BAgger (reverse, ONE round -- round 1 was the best on VoxCPM after the corrected evaluation) on the r4 main line.
#   bagger_r4.sh <out root>     (uses every visible GPU; env COUNT=1024 NGEN=15 CFGW=2.5 INIT SHARE=0.33 STEPS=6000)
# 1) collect COUNT reversed corrective trajectories with the INIT model, sharded over the GPUs (16-chunk rollouts so the
#    256-frame training window holds the whole reversed trajectory -- review fix); 2) train three lanes in parallel from
#    INIT on the same data: stream+intent+bagger, stream+bagger, bagger only; each lane validates while training (r4_lane.sh).
# 11-04 parameters (defaults = the EX002 run above, unchanged): PROFILE=g1 INIT=<G1 base> VAE=<G1 vae> (scripts/lane_profile.sh)
#   GPUS="0 1" physical GPUs (default: every visible GPU), LANES="b" subset of a b c (g1 supports b c; the G1 recipe is b =
#   stream+bagger with STEPS=2000), COLLECT_ONLY=1 stops after the collection.
set -uo pipefail
R=$1; CODE=$(cd "$(dirname "$0")/.." && pwd); cd $CODE; PY=${PY:-python}
source $CODE/scripts/lane_profile.sh
export PYTHONPATH=$CODE${PYTHONPATH_EXTRA:+:$PYTHONPATH_EXTRA} USE_TF=0 OMP_NUM_THREADS=2
GL=(${GPUS:-$(seq 0 $(($(nvidia-smi -L | wc -l) - 1)))}); NG=${#GL[@]}; DATA=$R/bagger-data; mkdir -p $DATA
LANES=${LANES:-$LANES_OK}
for l in $LANES; do case " $LANES_OK " in *" $l "*) ;; *) echo "lane $l not supported by PROFILE=${PROFILE:-ex002}"; exit 2;; esac; done
if [ ! -f $DATA/MANIFEST.json ]; then
  for g in $(seq 0 $((NG-1))); do
    CUDA_VISIBLE_DEVICES=${GL[$g]} $PY scripts/bagger_collect.py --ckpt $INIT "${COLLECT_ARGS[@]}" --out $DATA --cond reverse --count ${COUNT:-1024} --n-gen ${NGEN:-15} \
      --seed 1000 --cfg-w ${CFGW:-2.5} --shard $g --shards $NG > $R/collect-$g.log 2>&1 &
  done; wait
  [ -f $DATA/MANIFEST.json ] || { echo "$(date -Is) COLLECT_FAILED"; exit 1; }
fi
echo "$(date -Is) COLLECT_DONE $(python3 -c "import json;print(len(json.load(open('$DATA/MANIFEST.json'))['rows']))")"
[ "${COLLECT_ONLY:-0}" = 1 ] && exit 0
S='"p_stream": 0.67, "p_async": 1.0, "p_async_seg": 0.0'
export BAGGER_DIR=$DATA BAGGER_SHARE=${SHARE:-0.33}
on() { case " $LANES " in *" $1 "*) return 0;; esac; return 1; }
on a && { SEMSHARE=0.6 bash scripts/r4_lane.sh ${GL[$((0 % NG))]} $R r4-stream-int-bag '{"adsc": true, "intent_aux": true}' "{$S, \"w_int\": 0.3, \"sem_w\": 2.0}" ${STEPS:-6000} > $R/lane-a.log 2>&1 & }
on b && { bash scripts/r4_lane.sh ${GL[$((1 % NG))]} $R r4-stream-bag '{"adsc": true}' "{$S}" ${STEPS:-6000} > $R/lane-b.log 2>&1 & }
on c && { bash scripts/r4_lane.sh ${GL[$((2 % NG))]} $R r4-bag '{}' '{}' ${STEPS:-6000} > $R/lane-c.log 2>&1 & }
wait; echo "$(date -Is) BAGGER_R4_DONE"
