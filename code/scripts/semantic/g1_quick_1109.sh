#!/bin/bash
# 11-09 G1 dev quick lanes in ONE pod (pre-registrations on exp/text-path: docs/criteria/SEMREPA_1109.md amendments 1 / 3 and
# docs/criteria/G1_SFSCOPE_1109.md). Four arms, czm2kws recipe (hos-base warm start, 2000 steps, training seed 0, w_safe 300,
# causal_front / spk / text, pre-train-slim):
#   GPU0 ctrl (= sf0)   GPU1 semrepa (SemREPA-C, teacher $TEACHER, lambda 0.5, block 2)   GPU2 sffull (scope all)   GPU3 sfenc (scope encoders)
# Then QUICK100, per_clip seed 0: ctrl offline / offline no-text / psync; semrepa offline / no-text; sffull, sfenc offline / psync;
# czm2kws isodev-1107 seeds 0-2 re-evaluated on QUICK100 (offline, psync); pair_win (rec-sem-hos x 3) with GT / mismatched-GT rows;
# liveliness (g1_liveliness.py, 400-clip dev GT normaliser); stutter (beat_phase.py).
#   g1_quick_1109.sh <out root>     env TEACHER (required), QUICK, PY / BENCH_PY / EVAL_PY, ONLY_EVAL=1 (skip training)
set -uo pipefail
O=$1; CODE=$(cd "$(dirname "$0")/../.." && pwd); cd $CODE
B=/mnt/cpfs/zbl-cpfs-new/CKPT/zehan/gesture-scmg; C=/mnt/cpfs/zbl-cpfs-new/dataset/zehan/gesture
CZ=$B/causal-train-1105; A=$B/echog-g1-hos-assets; HV=$B/views/echog-g1-v1-hos; SB=$B/semantic-baseline-1105/g1
DEC=$B/echog-g1-decft-1105/decft-w3d0m05/final.pt; REF=$CZ/isodev-1107/dev/czm2kws/bench
Q=${QUICK:-$B/g1-quick-1109/QUICK100.txt}; TEACHER=${TEACHER:?}
D=$B/baseline-repro-1104/echog/data_root; W=/mnt/cpfs/zbl-cpfs-new/USERS/zehan/baselines/echog-official
export PY=${PY:-/usr/local/bin/python}; export BENCH_PY=${BENCH_PY:-$PY} EVAL_PY=${EVAL_PY:-/mnt/cpfs/zbl-cpfs-new/USERS/zehan/envs/echog/bin/python}
export EG=$D OFF=$W/official MMAE=$D/eval_assets/mmae/g1_mmae_30body_30fps.npy FGD_CKPT=$W/weights/evaluation/g1_fgd_encoder.pt
export WORKERS=${WORKERS:-6} OMP_NUM_THREADS=2 USE_TF=0 SPK_REFRESH=grades CAUSAL_FEAT=inc SIDE=dev G1_BENCH=0
unset STEPS SCMG_LOCAL_MIRROR
mkdir -p $O; log() { echo "$(date -Is) $*" >> $O/quick.log; }
log "START commit $(cat GIT_COMMIT 2>/dev/null || git rev-parse HEAD) teacher $TEACHER quick $(sha256sum $Q | cut -c1-16)"
SF='"self_force_p": 0.5, "self_force_steps": 4, "self_force_rounds": 1, "self_force_cfg": 0.0, "self_force_mix": 0.0, "self_force_noise": 0.0'
if [ "${ONLY_EVAL:-0}" != 1 ]; then
  ( GPU=0 EXTRA_SC='"w_safe": 300' bash scripts/echog_g1/causal_g1.sh lane $O ctrl 2000 ) &
  ( GPU=1 LANE_CFG='{"adsc": true, "semrepa": true}' EXTRA_SC="\"w_safe\": 300, \"w_semrepa\": 0.5, \"semrepa_teacher\": \"$TEACHER\"" \
      bash scripts/echog_g1/causal_g1.sh lane $O semrepa 2000 ) &
  ( GPU=2 EXTRA_SC="\"w_safe\": 300, $SF, \"self_force_scope\": \"all\"" bash scripts/echog_g1/causal_g1.sh lane $O sffull 2000 ) &
  ( GPU=3 EXTRA_SC="\"w_safe\": 300, $SF, \"self_force_scope\": \"encoders\"" bash scripts/echog_g1/causal_g1.sh lane $O sfenc 2000 ) &
  wait $(jobs -p)
  log "TRAIN_DONE $(ls $O/dev/*/final.pt 2>/dev/null | wc -l) final.pt"
fi
# dev benchmark environment (= causal_g1.sh dev_env)
export G1_VIEW=$HV REF_DIR=$A/motion_39d_30fps WAV_DIR=$A/audio SPLIT=$A/dev_stems.txt VIEW_TEST=$HV VSPLIT=val \
  TEXT_TEST=$C/scmg/text-e5-echog-g1-v1 TEXT_CAUSAL_TEST=$B/models/text-e5-causal-echog-g1-v1 WORDS_DIR=$CZ/words-dev CAUSAL_WAV=$A/audio \
  CAMPPLUS=$CZ/campplus PSTATS=$CODE/scripts/echog_g1/PROSODY_STATS_echog_g1_hos_train.json
bn() {   # <arm> <gpu> <protocol> <tag> [extra infer args]
  local arm=$1 gpu=$2 proto=$3 tag=$4 extra=${5:-}; local R=$O/dev/$arm/bench/$tag
  [ -f $R/benchmark_seed_000.json ] && return 0
  [ -f $O/dev/$arm/final.pt ] || { log "NO_FINAL $arm"; return 1; }
  INFER_EXTRA="--seed-mode per_clip $extra" GPUS="$gpu $gpu" CFGW=2.5 PY=$BENCH_PY bash scripts/echog_g1/bench.sh scmg $O/dev/$arm/final.pt $DEC $R/seed_000 $proto 0 $Q \
    >> $O/dev/$arm/bench_quick.log 2>&1 && log "BENCH_DONE $arm $tag" || log "BENCH_FAILED $arm $tag"
}
refev() {   # czm2kws isodev seeds 0-2 on QUICK100 (no new inference): QUICK100-only link dirs, then the official evaluator
  for p in offline psync; do for s in 0 1 2; do
    local L=$O/ref-czm2kws/$p/seed_00$s; local J=$O/ref-czm2kws/$p/benchmark_seed_00$s.json; [ -f $J ] && continue
    mkdir -p $L; while read st; do ln -sf $REF/$p/seed_00$s/$st.pt $L/$st.pt; done < $Q
    PY=$BENCH_PY bash scripts/echog_g1/bench.sh eval $L $Q czm2kws_${p}_seed$s $J >> $O/ref-czm2kws/eval.log 2>&1 && log "REF_DONE $p s$s" || log "REF_FAILED $p s$s"
  done; done
}
( bn ctrl 0 offline offline; bn ctrl 0 offline offline-notext "--modality audio"; bn ctrl 0 psync psync ) &
( bn semrepa 1 offline offline; bn semrepa 1 offline offline-notext "--modality audio"; refev ) &
( bn sffull 2 offline offline; bn sffull 2 psync psync ) &
( bn sfenc 3 offline offline; bn sfenc 3 psync psync ) &
wait $(jobs -p); log "BENCH_ALL_DONE"
# pair_win: mismatched GT = derangement inside QUICK100 (seed 20261109)
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
print('mismatched GT: same recording', sum(rec(q[p[k]]) == rec(q[k]) for k in range(len(q))))
EOF
sp=(gt-s0=- mis-s0=$M)
for arm in ctrl semrepa sffull sfenc; do
  [ -d $O/dev/$arm/bench/offline/seed_000 ] && sp+=($arm-full-s0=$O/dev/$arm/bench/offline/seed_000)
  [ -d $O/dev/$arm/bench/offline-notext/seed_000 ] && sp+=($arm-notext-s0=$O/dev/$arm/bench/offline-notext/seed_000)
done
SCORE_EXTRA="--allow-missing --stems $Q" PAIRS= bash scripts/semantic/sem_score_dirs.sh 0 $SB/data-sem-hos "$SB/rec-sem-hos-seed*" $O/sem dev "${sp[@]}" > $O/sem.log 2>&1
srcs=$(for s in "${sp[@]}"; do x=${s%%=*}; echo -n "${x%-s0} "; done)
$PY scripts/semantic/sem_multiseed.py --scores $O/sem --rec-prefix rec-sem-hos-seed --seeds 0 --sources $srcs \
  --pairs semrepa-full:ctrl-full,ctrl-full:ctrl-notext,semrepa-full:semrepa-notext,sfenc-full:sffull-full --out $O/sem/MULTI.json >> $O/sem.log 2>&1 \
  && log "SEM_DONE" || log "SEM_FAILED"
lv=(GT=$A/motion_39d_30fps)
for arm in ctrl semrepa sffull sfenc; do for t in offline psync; do [ -d $O/dev/$arm/bench/$t/seed_000 ] && lv+=($arm-$t=$O/dev/$arm/bench/$t/seed_000); done; done
PYTHONPATH=$OFF/src $EVAL_PY scripts/echog_g1/g1_liveliness.py --norm-gt $A/motion_39d_30fps --norm-stems $A/dev_stems.txt --stems $Q \
  --out $O/LIVELY.json "${lv[@]}" > $O/lively.log 2>&1 && log "LIVELY_DONE" || log "LIVELY_FAILED"
bp=(); for x in "${lv[@]:1}"; do bp+=($x); done
PYTHONPATH=$OFF/src $EVAL_PY scripts/echog_g1/beat_phase.py --mmae-file $MMAE --ref-dir $A/motion_39d_30fps --stems $Q "${bp[@]}" > $O/STUTTER.txt 2>&1 \
  && log "STUTTER_DONE" || log "STUTTER_FAILED"
chmod -R a+rwX $O 2>/dev/null; log "QUICK_DONE"
