#!/bin/bash
# 11-09 EX002 single test-set scoring run (EVAL_PROTOCOL v1, tag eval-protocol-v1; TEST_USAGE entry 2026-11-09 09:27).
# Final method = enc-s0 (none stream, psync) + explicit-semantics canonical path (the ex002_expl_full.sh chain + the amendment-6
# NUMBER insert of ex002_expl_a6.sh) + safety layer 3c-4b (scripts/ex002_cbf_dev.py proj_l2 --recursive --max-iter 8) on the none
# stream; the same commands as on dev, with the test-only view and its caches.   ex002_test_1109.sh <run dir>   env PY, NG
set -uo pipefail
O=$1; CODE=$(cd "$(dirname "$0")/../.." && pwd); cd $CODE; mkdir -p $O; NG=${NG:-2}
B=/mnt/cpfs/zbl-cpfs-new/CKPT/zehan/gesture-scmg; X=$B/control-token-1105/ex002; T=$X/tokens; TD=$B/ex002-test-1109
PY=${PY:-python}; export SCMG_J=34 SCMG_NO_GLITCH=1 OMP_NUM_THREADS=${OMP_NUM_THREADS:-2} SCMG_EXPL_VIEW=$TD/view
VIEW=$TD/view; DATA="--split test --view $VIEW --text-cache $TD/text-e5 --tok-cache $TD/tok-qwen25 --text-causal $TD/text-e5-causal --words-dir $TD/causal/words"
ENCK=$B/ex002-sfscope-1108/enc-s0/train/final.pt; CK=$X/m10-enc/b4k/final.pt; ENC=$O/enc/psync
CIS=$(seq -s, 0 16)
BL="scripts/ex002/ex002_local_blend.py --pre 0.3 --post 0.9 --compose smooth"; FO="--finger-override $T/HAND_SHAPES.json --fo-ramp 0.3 --fo-profile minjerk --fo-library $T/HAND_LIBRARY_TRAIN.npz"
KF="scripts/ex002/expl_canon.py"
log() { echo "$(date -Is) $*" >> $O/lane.log; }
log "START commit $(git rev-parse HEAD 2>/dev/null)"
# 1. none stream: enc-s0, psync, seeds 0-2
for s in 0 1 2; do G=$ENC/s$s; [ -f $G/.done ] && continue
  for k in $(seq 0 $((NG - 1))); do CUDA_VISIBLE_DEVICES=$k $PY scripts/ex002/bench_ex002.py infer --ckpt $ENCK --protocol psync --seed $s --out $G $DATA --shard $k/$NG >> $O/infer_enc_$k.log 2>&1 & done; wait $(jobs -p)
  touch $G/.done; log "ENC s$s"; done
# 2. plans, 3. given generation (control modules)
for s in 0 1 2; do for p in tts psync; do $PY $KF plan --cis $CIS --proto $p --none $ENC/s$s --out $O/plans/s$s >> $O/lane.log 2>&1; done; done; log "PLANS"
for s in 0 1 2; do for p in tts psync; do
  LA=""; [ $p = tts ] && LA="--ctl-lookahead 0.5"; G=$O/gen/$p/s$s; [ -f $G/.done ] && continue
  for k in $(seq 0 $((NG - 1))); do CUDA_VISIBLE_DEVICES=$k $PY scripts/ex002/bench_ex002.py infer --ckpt $CK --protocol psync --seed $s --out $G $DATA \
      --plan $O/plans/s$s/TOKPLAN_$p.json --mode given $LA --shard $k/$NG >> $O/infer_$k.log 2>&1 & done; wait $(jobs -p)
  touch $G/.done; log "GEN $p s$s"; done; done
# 4. blend + SELF chest + point keyframes
post() { local p=$1 s=$2; local F=$O/final/$p/s$s; mkdir -p $O/final/$p
  $PY $BL --none $ENC/s$s --given $O/gen/$p/s$s --plan $O/plans/s$s/TOKPLAN_$p.json --out $O/blend/$p/s$s --mode given $FO > /dev/null 2>> $O/blend.log
  $PY $KF keyframe --self-chest --plan $O/plans/s$s/PLAN_$p.json --run $O/blend/$p/s$s --out $O/chest/$p/s$s >> $O/kf.log 2>&1
  $PY $KF keyframe --point-only --plan $O/plans/s$s/PLAN_$p.json --run $O/chest/$p/s$s --out $F >> $O/kf.log 2>&1
  for g in $ENC/../gt_*.npy $ENC/gt_*.npy; do [ -f $g ] && ln -sf $g $O/final/$p/; done; log "POST $p s$s"; }
for s in 0 1 2; do for p in tts psync; do post $p $s & done; done; wait $(jobs -p)
# 5. amendment 6 NUMBER insert on tts s0 (as ex002_expl_a6.sh)
A6=$O/a6; mkdir -p $A6
$PY - $O/plans/s0/PLAN_tts.json $A6/NUMPLAN_tts_s0.json <<'PYEOF'
import json, sys
P = json.load(open(sys.argv[1])); ev = [e for e in P['events'] if e['cat'] == 'NUMBER']
json.dump(dict(P, events=ev, n_events=len(ev)), open(sys.argv[2], 'w')); print('NUMBER events', len(ev))
PYEOF
for k in 0 1 2 3 4 5 6 7; do $PY scripts/ex002/ex002_keyframe.py insert --version 9c --hold-pool $T/HOLD_POOL_DEICTIC_TRAIN.npz --pool point --release-search --palm-fix --finger-hold 0.2,1.0 \
  --fo-library $T/HAND_LIBRARY_TRAIN.npz --given $O/final/tts/s0 --plan $A6/NUMPLAN_tts_s0.json --out $A6/tts_s0 --mode given --shard $k/8 >> $O/lane_a6.log 2>&1 & done; wait $(jobs -p)
for g in $ENC/../gt_*.npy $ENC/gt_*.npy; do [ -f $g ] && ln -sf $g $A6/; done; log "A6 INSERTED"
# 6. safety layer 3c-4b on the none stream
for s in 0 1 2; do $PY scripts/ex002_cbf_dev.py run --src $O/enc --out $O/proj4b --proto psync --seed $s --layer proj_l2 --recursive --max-iter 8 > $O/proj_psync_$s.log 2>&1 & done; wait $(jobs -p)
$PY scripts/ex002_cbf_dev.py check --src $O/enc --out $O/proj4b > $O/check.log 2>&1; log "PROJ"
# 7. scoring
for p in tts psync; do
  $PY $KF eval --plan $O/plans/s0/PLAN_$p.json,$O/plans/s1/PLAN_$p.json,$O/plans/s2/PLAN_$p.json --none $ENC/s0,$ENC/s1,$ENC/s2 \
    --out $O/EVAL_$p.json given=$O/final/$p/s0,$O/final/$p/s1,$O/final/$p/s2 >> $O/eval.log 2>&1 && log "EVAL $p" & done; wait $(jobs -p)
$PY $KF eval --plan $O/plans/s0/PLAN_tts.json --none $ENC/s0 --out $A6/EVAL_tts_s0.json before=$O/final/tts/s0 a6=$A6/tts_s0 >> $O/eval.log 2>&1 && log "EVAL a6"
$PY scripts/ex002/ex002_present.py eval --plan $A6/NUMPLAN_tts_s0.json --out $A6/PRESENT.json none=$ENC/s0 before=$O/final/tts/s0 a6=$A6/tts_s0 >> $O/eval.log 2>&1 && log "PRESENT"
CUDA_VISIBLE_DEVICES=0 $PY scripts/ex002/bench_ex002.py eval --out $O/BENCH_EVAL.json none_s0=$ENC/s0 none_s1=$ENC/s1 none_s2=$ENC/s2 \
  proj4b_s0=$O/proj4b/psync/s0 proj4b_s1=$O/proj4b/psync/s1 proj4b_s2=$O/proj4b/psync/s2 a6_tts_s0=$A6/tts_s0 \
  $(for p in tts psync; do for s in 0 1 2; do echo ${p}_s$s=$O/final/$p/s$s; done; done) >> $O/eval.log 2>&1 && log "BENCH_EVAL"
CUDA_VISIBLE_DEVICES=1 $PY scripts/ex002/ex002_fgd_outside.py --plan $O/plans/s0/PLAN_tts.json --radius 0,1.0 --out $O/FGDOUT_tts_s0.json none=$ENC/s0 tts=$O/final/tts/s0 >> $O/eval.log 2>&1 && log "FGDOUT"
$PY scripts/ex002/ex002_exec.py --view $VIEW --plan $O/plans/s0/PLAN_tts.json --out $O/EXEC.json $(for p in tts psync; do for s in 0 1 2; do echo ${p}_s$s=$O/final/$p/s$s; done; done) a6_tts_s0=$A6/tts_s0 >> $O/eval.log 2>&1 && log "EXEC"
log "LANE_DONE"
