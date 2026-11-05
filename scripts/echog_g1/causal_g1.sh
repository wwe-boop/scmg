#!/bin/bash
# 11-05 训练匹配 G1 lanes: the selected recipe (sf0 = stream warm start, p_stream 0.67, ADSC text streaming, self_force_p 0,
# contact-aware decoder) trained with the TRAINING-MATCHED causal streaming inputs of the psync protocol (StageCfg
# causal_front + causal_spk + causal_text; scmg/causal_feat.py, View.load_causal of scripts/echog_g1/precompute_causal.py).
#   causal_g1.sh lane <out root> <label> <steps>   env SIDE dev|test, GPU, BENCH_PROTOS, G1_BENCH (dev 1 / test 0), SEEDS
#   causal_g1.sh ref  <out root> <protos>          dev only: the existing sf0 dev model under the given bench protocols
# dev = hos-base on the held-out-speaker view, benchmark on its 400 dev clips; test = fix-base on the full view (its test
# inference runs only when G1_BENCH=1 is given explicitly). Paths: B / C / CZ (precompute, campplus, words on CPFS).
set -uo pipefail
cmd=$1; O=$2; CODE=$(cd "$(dirname "$0")/../.." && pwd); cd $CODE; mkdir -p $O
B=/mnt/cpfs/zbl-cpfs-new/CKPT/zehan/gesture-scmg; C=/mnt/cpfs/zbl-cpfs-new/dataset/zehan/gesture
CZ=${CZ:-$B/causal-train-1105}; DEC=${DEC:-$B/echog-g1-decft-1105/decft-w3d0m05/final.pt}
A=${HOS_ASSETS:-$B/echog-g1-hos-assets}; HV=$B/views/echog-g1-v1-hos
SC='"p_stream": 0.67, "p_async": 1.0, "p_async_seg": 0.0, "self_force_p": 0'
CSC='"causal_front": true, "causal_spk": true, "causal_text": true'
export PROFILE=g1 SNAP=1000000 TRAIN_SEED=${TRAIN_SEED:-0} CAMPPLUS=$CZ/campplus PSTATS=$CODE/scripts/echog_g1/PROSODY_STATS_echog_g1_hos_train.json
log() { echo "$(date -Is) [$cmd] $*" >> $O/causal.log; }
dev_env() { export G1_VIEW=$HV G1_BENCH_STEMS=$A/dev_stems.txt REF_DIR=$A/motion_39d_30fps WAV_DIR=$A/audio SPLIT=$A/dev_stems.txt \
  VIEW_TEST=$HV VSPLIT=val TEXT_TEST=$C/scmg/text-e5-echog-g1-v1 TEXT_CAUSAL_TEST=$B/models/text-e5-causal-echog-g1-v1 \
  WORDS_DIR=$CZ/words-dev CAUSAL_WAV=$A/audio; }
case $cmd in
  lane)
    lab=$3; st=$4; side=${SIDE:-dev}
    sc="{$SC, $CSC}"; ${PY:-python} -c "import json,sys; json.loads(sys.argv[1])" "$sc" || exit 2
    if [ $side = dev ]; then dev_env; INIT=${INIT:-$B/echog-g1-arch-1104/hos-base/best.pt}; export G1_BENCH=${G1_BENCH:-1}
    else INIT=${INIT:-$B/echog-g1-arch-1104/fix-base/best.pt}; export G1_BENCH=${G1_BENCH:-0} WORDS_DIR=$CZ/words-vc; fi
    log "lane $side/$lab steps $st gpu ${GPU:-0} sc $sc init $INIT bench $G1_BENCH protos ${BENCH_PROTOS:-offline psync} seeds ${SEEDS:-0}"
    INIT=$INIT VAE=$DEC CAUSAL_DIR=$CZ/pre-train-slim BENCH_PROTOS="${BENCH_PROTOS:-offline psync}" G1_BENCH_SEEDS="${SEEDS:-0}" \
      bash scripts/r4_lane.sh ${GPU:-0} $O/$side $lab '{"adsc": true}' "$sc" $st > $O/lane-$side-$lab.log 2>&1
    log "LANE_END $side/$lab rc $?";;
  ref)
    dev_env; CK=${REF_CK:-$B/echog-g1-sf-1105/dev/sf0/final.pt}
    for p in $3; do
      P=$O/dev/sf0ref/bench/$p/seed_000; [ -f $(dirname $P)/benchmark_seed_000.json ] && continue
      PY=${BENCH_PY:-${PY:-python}} GPUS="${GPUS:-0}" CFGW=2.5 bash scripts/echog_g1/bench.sh scmg $CK $DEC $P $p 0 >> $O/ref.log 2>&1 \
        && log "REF_DONE $p" || log "REF_FAILED $p"
    done;;
  *) echo "usage: causal_g1.sh lane|ref ..."; exit 2;;
esac
