#!/bin/bash
# 11-09 SEM_BRIDGE quick lanes F4 / F6 (exp/text-path docs/criteria/SEMBRIDGE_F4F6_1109.md): GPU0 f4 (w_f4 0.1), GPU1 f6 (w_f6 0.05,
# margin 2), czm2kws recipe, 2000 steps, seed 0; QUICK100 per_clip seed 0 offline + offline no-text; pair_win with the ctrl arm of
# g1-quick-1109 (waits for its benches), liveliness.
#   g1_quick_f4f6.sh <out root> <g1-quick-1109 run dir>
set -uo pipefail
O=$1; R1=$2; CODE=$(cd "$(dirname "$0")/../.." && pwd); cd $CODE
B=/mnt/cpfs/zbl-cpfs-new/CKPT/zehan/gesture-scmg; C=/mnt/cpfs/zbl-cpfs-new/dataset/zehan/gesture
CZ=$B/causal-train-1105; A=$B/echog-g1-hos-assets; HV=$B/views/echog-g1-v1-hos; SB=$B/semantic-baseline-1105/g1
DEC=$B/echog-g1-decft-1105/decft-w3d0m05/final.pt; Q=${QUICK:-$B/g1-quick-1109/QUICK100.txt}
D=$B/baseline-repro-1104/echog/data_root; W=/mnt/cpfs/zbl-cpfs-new/USERS/zehan/baselines/echog-official
export PY=${PY:-/usr/local/bin/python}; export BENCH_PY=${BENCH_PY:-$PY} EVAL_PY=${EVAL_PY:-/mnt/cpfs/zbl-cpfs-new/USERS/zehan/envs/echog/bin/python}
export EG=$D OFF=$W/official MMAE=$D/eval_assets/mmae/g1_mmae_30body_30fps.npy FGD_CKPT=$W/weights/evaluation/g1_fgd_encoder.pt
export WORKERS=${WORKERS:-6} OMP_NUM_THREADS=2 USE_TF=0 SPK_REFRESH=grades CAUSAL_FEAT=inc SIDE=dev G1_BENCH=0
unset STEPS SCMG_LOCAL_MIRROR
mkdir -p $O; log() { echo "$(date -Is) $*" >> $O/quick.log; }
log "START commit $(cat GIT_COMMIT 2>/dev/null || git rev-parse HEAD) quick $(sha256sum $Q | cut -c1-16) ctrl from $R1"
if [ "${ONLY_EVAL:-0}" != 1 ]; then
  ( GPU=0 LANE_CFG='{"adsc": true, "f4_align": true}' EXTRA_SC='"w_safe": 300, "w_f4": 0.1' bash scripts/echog_g1/causal_g1.sh lane $O f4 2000 ) &
  ( GPU=1 EXTRA_SC='"w_safe": 300, "w_f6": 0.05, "f6_margin": 2.0' bash scripts/echog_g1/causal_g1.sh lane $O f6 2000 ) &
  wait $(jobs -p); log "TRAIN_DONE $(ls $O/dev/*/final.pt 2>/dev/null | wc -l) final.pt"
fi
export G1_VIEW=$HV REF_DIR=$A/motion_39d_30fps WAV_DIR=$A/audio SPLIT=$A/dev_stems.txt VIEW_TEST=$HV VSPLIT=val \
  TEXT_TEST=$C/scmg/text-e5-echog-g1-v1 TEXT_CAUSAL_TEST=$B/models/text-e5-causal-echog-g1-v1 WORDS_DIR=$CZ/words-dev CAUSAL_WAV=$A/audio \
  CAMPPLUS=$CZ/campplus PSTATS=$CODE/scripts/echog_g1/PROSODY_STATS_echog_g1_hos_train.json
bn() {
  local arm=$1 gpu=$2 proto=$3 tag=$4 extra=${5:-}; local R=$O/dev/$arm/bench/$tag
  [ -f $R/benchmark_seed_000.json ] && return 0
  [ -f $O/dev/$arm/final.pt ] || { log "NO_FINAL $arm"; return 1; }
  INFER_EXTRA="--seed-mode per_clip $extra" GPUS="$gpu $gpu" CFGW=2.5 PY=$BENCH_PY bash scripts/echog_g1/bench.sh scmg $O/dev/$arm/final.pt $DEC $R/seed_000 $proto 0 $Q \
    >> $O/dev/$arm/bench_quick.log 2>&1 && log "BENCH_DONE $arm $tag" || log "BENCH_FAILED $arm $tag"
}
( bn f4 0 offline offline; bn f4 0 offline offline-notext "--modality audio" ) &
( bn f6 1 offline offline; bn f6 1 offline offline-notext "--modality audio" ) &
wait $(jobs -p); log "BENCH_ALL_DONE"
for i in $(seq 360); do [ -f $R1/dev/ctrl/bench/offline-notext/benchmark_seed_000.json ] && break; sleep 30; done
sp=(gt-s0=- mis-s0=$R1/sem/mis_gt)
[ -d $R1/sem/mis_gt ] || { mkdir -p $O/mis_gt_wait; log "NO_MISGT_YET"; sp=(gt-s0=-); }
for arm in ctrl; do sp+=($arm-full-s0=$R1/dev/$arm/bench/offline/seed_000 $arm-notext-s0=$R1/dev/$arm/bench/offline-notext/seed_000); done
for arm in f4 f6; do
  [ -d $O/dev/$arm/bench/offline/seed_000 ] && sp+=($arm-full-s0=$O/dev/$arm/bench/offline/seed_000)
  [ -d $O/dev/$arm/bench/offline-notext/seed_000 ] && sp+=($arm-notext-s0=$O/dev/$arm/bench/offline-notext/seed_000)
done
SCORE_EXTRA="--allow-missing --stems $Q" PAIRS= bash scripts/semantic/sem_score_dirs.sh 0 $SB/data-sem-hos "$SB/rec-sem-hos-seed*" $O/sem dev "${sp[@]}" > $O/sem.log 2>&1
srcs=$(for s in "${sp[@]}"; do x=${s%%=*}; echo -n "${x%-s0} "; done)
$PY scripts/semantic/sem_multiseed.py --scores $O/sem --rec-prefix rec-sem-hos-seed --seeds 0 --sources $srcs \
  --pairs f4-full:ctrl-full,f6-full:ctrl-full,ctrl-full:ctrl-notext,f4-full:f4-notext,f6-full:f6-notext --out $O/sem/MULTI.json >> $O/sem.log 2>&1 \
  && log "SEM_DONE" || log "SEM_FAILED"
lv=(GT=$A/motion_39d_30fps ctrl-offline=$R1/dev/ctrl/bench/offline/seed_000)
for arm in f4 f6; do [ -d $O/dev/$arm/bench/offline/seed_000 ] && lv+=($arm-offline=$O/dev/$arm/bench/offline/seed_000); done
PYTHONPATH=$OFF/src $EVAL_PY scripts/echog_g1/g1_liveliness.py --norm-gt $A/motion_39d_30fps --norm-stems $A/dev_stems.txt --stems $Q \
  --out $O/LIVELY.json "${lv[@]}" > $O/lively.log 2>&1 && log "LIVELY_DONE" || log "LIVELY_FAILED"
chmod -R a+rwX $O 2>/dev/null; log "QUICK_DONE"
