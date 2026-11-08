#!/bin/bash
# 11-09 G1 encoder-only self-forcing FULL dev lane (exp/text-path docs/criteria/G1_SFSCOPE_1109.md "Full G1 dev lane"): arms sf0 / full / enc
# x training seeds 0-2 (czm2kws recipe, hos-base, 2000 steps), 400 dev clips, offline + psync x per_clip 0 / 1, pair_win (full set,
# sanity rows), liveliness, stutter. One pod, NG GPUs, a lock-file work queue (training first, a benchmark once its model exists).
#   g1_sf_full_1109.sh <out root>     env NG (default 8), PY / BENCH_PY / EVAL_PY
set -uo pipefail
O=$1; CODE=$(cd "$(dirname "$0")/../.." && pwd); cd $CODE; NG=${NG:-8}
B=/mnt/cpfs/zbl-cpfs-new/CKPT/zehan/gesture-scmg; C=/mnt/cpfs/zbl-cpfs-new/dataset/zehan/gesture
CZ=$B/causal-train-1105; A=$B/echog-g1-hos-assets; HV=$B/views/echog-g1-v1-hos; SB=$B/semantic-baseline-1105/g1
DEC=$B/echog-g1-decft-1105/decft-w3d0m05/final.pt; Q=$A/dev_stems.txt
D=$B/baseline-repro-1104/echog/data_root; W=/mnt/cpfs/zbl-cpfs-new/USERS/zehan/baselines/echog-official
export PY=${PY:-/usr/local/bin/python}; export BENCH_PY=${BENCH_PY:-$PY} EVAL_PY=${EVAL_PY:-/mnt/cpfs/zbl-cpfs-new/USERS/zehan/envs/echog/bin/python}
export EG=$D OFF=$W/official MMAE=$D/eval_assets/mmae/g1_mmae_30body_30fps.npy FGD_CKPT=$W/weights/evaluation/g1_fgd_encoder.pt
export WORKERS=${WORKERS:-6} OMP_NUM_THREADS=2 USE_TF=0 SPK_REFRESH=grades CAUSAL_FEAT=inc SIDE=dev G1_BENCH=0
unset STEPS SCMG_LOCAL_MIRROR
mkdir -p $O/locks; log() { echo "$(date -Is) $*" >> $O/full.log; }
log "START commit $(cat GIT_COMMIT 2>/dev/null || git rev-parse HEAD) NG $NG"
SF='"self_force_p": 0.5, "self_force_steps": 4, "self_force_rounds": 1, "self_force_cfg": 0.0, "self_force_mix": 0.0, "self_force_noise": 0.0'
declare -A XSC=([sf0]='"w_safe": 300' [full]="\"w_safe\": 300, $SF, \"self_force_scope\": \"all\"" [enc]="\"w_safe\": 300, $SF, \"self_force_scope\": \"encoders\"")
export G1_VIEW=$HV REF_DIR=$A/motion_39d_30fps WAV_DIR=$A/audio SPLIT=$A/dev_stems.txt VIEW_TEST=$HV VSPLIT=val \
  TEXT_TEST=$C/scmg/text-e5-echog-g1-v1 TEXT_CAUSAL_TEST=$B/models/text-e5-causal-echog-g1-v1 WORDS_DIR=$CZ/words-dev CAUSAL_WAV=$A/audio \
  CAMPPLUS=$CZ/campplus PSTATS=$CODE/scripts/echog_g1/PROSODY_STATS_echog_g1_hos_train.json
T=$O/TASKS.txt; : > $T
for s in 0 1 2; do for arm in sf0 full enc; do echo "train $arm $s" >> $T; done; done
for s in 0 1 2; do for arm in sf0 full enc; do for p in offline psync; do for k in 0 1; do echo "bench $arm $s $p $k" >> $T; done; done; done; done
run_task() {  # gpu kind arm seed [proto iseed]
  local g=$1 kind=$2 arm=$3 s=$4; local lab=$arm-s$s
  if [ $kind = train ]; then
    ( GPU=$g TRAIN_SEED=$s EXTRA_SC="${XSC[$arm]}" bash scripts/echog_g1/causal_g1.sh lane $O $lab 2000 ) && log "TRAINED $lab" || log "TRAIN_FAILED $lab"
  else
    local p=$5 k=$6; local R=$O/dev/$lab/bench/$p
    INFER_EXTRA="--seed-mode per_clip" GPUS="$g $g" CFGW=2.5 PY=$BENCH_PY bash scripts/echog_g1/bench.sh scmg $O/dev/$lab/final.pt $DEC $R/seed_00$k $p $k $Q \
      >> $O/dev/$lab/bench_full.log 2>&1 && log "BENCH_DONE $lab $p s$k" || log "BENCH_FAILED $lab $p s$k"
  fi
}
worker() {
  local g=$1
  while :; do
    local left=0 did=0
    while read -r kind arm s p k; do
      local key=$(echo "$kind-$arm-$s-${p:-}-${k:-}"); [ -e $O/locks/$key.done ] && continue; left=1
      [ -d $O/locks/$key ] && continue
      if [ $kind = bench ] && [ ! -f $O/dev/$arm-s$s/final.pt ]; then
        grep -q "TRAIN_FAILED $arm-s$s" $O/full.log 2>/dev/null && { touch $O/locks/$key.done; } ; continue; fi
      mkdir $O/locks/$key 2>/dev/null || continue
      run_task $g $kind $arm $s $p $k; touch $O/locks/$key.done; did=1; break
    done < $T
    [ $left = 0 ] && break
    [ $did = 0 ] && sleep 30
  done
}
for g in $(seq 0 $((NG - 1))); do worker $g & sleep 2; done
wait $(jobs -p); log "QUEUE_DONE"
# pair_win: mismatched GT = 400-stem derangement across recordings (seed 20261109)
M=$O/sem/mis_gt; mkdir -p $M
$PY - "$Q" "$A/motion_39d_30fps" "$M" <<'EOF'
import sys, os, numpy as np
q = [l.strip() for l in open(sys.argv[1]) if l.strip()]; rng = np.random.default_rng(20261109); rec = lambda s: s.rsplit('_utt_', 1)[0]
p = rng.permutation(len(q))
for _ in range(100):
    bad = [k for k in range(len(q)) if rec(q[p[k]]) == rec(q[k])]
    if not bad: break
    for k in bad: j = int(rng.integers(len(q))); p[k], p[j] = p[j], p[k]
for k, s in enumerate(q):
    d = os.path.join(sys.argv[3], s + '.pt')
    if not os.path.exists(d): os.symlink(os.path.join(sys.argv[2], q[p[k]] + '.pt'), d)
EOF
sp=(gt-s0=- mis-s0=$M)
for s in 0 1 2; do for arm in sf0 full enc; do for k in 0 1; do
  d=$O/dev/$arm-s$s/bench/offline/seed_00$k; [ -d $d ] && sp+=($arm-t$s-s$k=$d); done; done; done
PAIRS= bash scripts/semantic/sem_score_dirs.sh 0 $SB/data-sem-hos "$SB/rec-sem-hos-seed*" $O/sem dev "${sp[@]}" > $O/sem.log 2>&1 && log "SEM_SCORED" || log "SEM_FAILED"
lv=(GT=$A/motion_39d_30fps)
for s in 0 1 2; do for arm in sf0 full enc; do for p in offline psync; do for k in 0 1; do
  d=$O/dev/$arm-s$s/bench/$p/seed_00$k; [ -d $d ] && lv+=($arm-t$s-$p-s$k=$d); done; done; done; done
PYTHONPATH=$OFF/src $EVAL_PY scripts/echog_g1/g1_liveliness.py --norm-gt $A/motion_39d_30fps --norm-stems $Q --stems $Q --out $O/LIVELY.json "${lv[@]}" > $O/lively.log 2>&1 \
  && log "LIVELY_DONE" || log "LIVELY_FAILED"
PYTHONPATH=$OFF/src $EVAL_PY scripts/echog_g1/beat_phase.py --mmae-file $MMAE --ref-dir $A/motion_39d_30fps --stems $Q "${lv[@]:1}" > $O/STUTTER.txt 2>&1 \
  && log "STUTTER_DONE" || log "STUTTER_FAILED"
chmod -R a+rwX $O 2>/dev/null; log "FULL_DONE"
