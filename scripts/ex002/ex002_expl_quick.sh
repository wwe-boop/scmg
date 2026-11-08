#!/bin/bash
# 11-09 explicit semantics, canonical path quick lane (exp/text-path docs/criteria/EXPLICIT_SEM_1109.md amendment 1): the 10 P2 quick-lane
# recordings, seed 0; plans tts (t = onset - 0.2, psync + look-ahead 0.5 = the TTS-sync row, primary) and psync (t = onset); metric
# validation on synthetic positives / negatives; generation with the tok10ab model (m10-enc b4k) + 5c blend + module-10 override
# library; BIG / SMALL keyframes; per-category success vs none; FGD v2 (i) / (ii) and the liveliness guard.
#   ex002_expl_quick.sh <out dir>    env PY, G
set -uo pipefail
O=$1; CODE=$(cd "$(dirname "$0")/../.." && pwd); cd $CODE; mkdir -p $O
B=/mnt/cpfs/zbl-cpfs-new/CKPT/zehan/gesture-scmg; X=$B/control-token-1105/ex002; T=$X/tokens; P2=$X/p2-quick-1109
PY=${PY:-python}; G=${G:-0}; export SCMG_J=34 SCMG_NO_GLITCH=1 OMP_NUM_THREADS=${OMP_NUM_THREADS:-2}
CIS=0,4,8,12,16,20,24,28,32,36; CK=$X/m10-enc/b4k/final.pt; NONE=$P2/psync/none_s0
BL="scripts/ex002/ex002_local_blend.py --pre 0.3 --post 0.9 --compose smooth"; FO="--finger-override $T/HAND_SHAPES.json --fo-ramp 0.3 --fo-profile minjerk --fo-library $T/HAND_LIBRARY_TRAIN.npz"
log() { echo "$(date -Is) $*" >> $O/lane.log; }
log "START commit $(cat GIT_COMMIT 2>/dev/null || git rev-parse HEAD 2>/dev/null)"
for p in tts psync; do $PY scripts/ex002/expl_canon.py plan --cis $CIS --proto $p --none $NONE --out $O/plans >> $O/lane.log 2>&1; done; log "PLANS"
$PY scripts/ex002/expl_canon.py validate --plan $O/plans/PLAN_tts.json --gt $P2/psync --out $O/METRIC_VALIDATION.json >> $O/validate.log 2>&1 && log "VALIDATED"
for p in tts psync; do
  LA=""; [ $p = tts ] && LA="--ctl-lookahead 0.5"
  [ -f $O/gen/$p/.done ] || { CUDA_VISIBLE_DEVICES=$G $PY scripts/ex002/bench_ex002.py infer --ckpt $CK --protocol psync --seed 0 --cis $CIS --out $O/gen/$p \
      --plan $O/plans/TOKPLAN_$p.json --mode given $LA >> $O/infer.log 2>&1 && touch $O/gen/$p/.done && log "GEN $p"; }
  for g in $NONE/gt_*.npy; do ln -sf $g $O/gen/; done
  $PY $BL --none $NONE --given $O/gen/$p --plan $O/plans/TOKPLAN_$p.json --out $O/blend/$p --mode given $FO > /dev/null 2>> $O/blend.log && log "BLEND $p"
  mkdir -p $O/final; $PY scripts/ex002/expl_canon.py keyframe --plan $O/plans/PLAN_$p.json --run $O/blend/$p --out $O/final/$p >> $O/lane.log 2>&1 && log "KEYFRAMES $p"
  $PY scripts/ex002/expl_canon.py keyframe --self-chest --plan $O/plans/PLAN_$p.json --run $O/blend/$p --out $O/final/${p}_chest >> $O/lane.log 2>&1 && log "KEYFRAMES chest $p"
  for g in $NONE/gt_*.npy; do ln -sf $g $O/final/; done
  $PY scripts/ex002/expl_canon.py eval --plan $O/plans/PLAN_$p.json --none $NONE --out $O/EVAL_$p.json given=$O/final/$p chest=$O/final/${p}_chest >> $O/eval.log 2>&1 && log "EVAL $p"
done
CUDA_VISIBLE_DEVICES=$G $PY scripts/ex002/bench_ex002.py eval --out $O/BENCH_EVAL.json none=$NONE tts=$O/final/tts psync=$O/final/psync >> $O/eval.log 2>&1 && log "BENCH_EVAL"
CUDA_VISIBLE_DEVICES=$G $PY scripts/ex002/ex002_fgd_outside.py --plan $O/plans/PLAN_tts.json --radius 0,1.0 --out $O/FGDOUT_tts.json none=$NONE tts=$O/final/tts >> $O/eval.log 2>&1 && log "FGDOUT"
chmod -R a+rwX $O 2>/dev/null; log "LANE_DONE"
