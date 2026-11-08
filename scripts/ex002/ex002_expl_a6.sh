#!/bin/bash
# 11-09 explicit semantics amendment 6 (EXPLICIT_SEM_1109.md): NUMBER present keyframes with the 1.2 s hold (--finger-hold 0.2,1.0) on the full run final/tts/s0;
# gates: count hit, signed palm, raised share; side gates.   ex002_expl_a5.sh <full run dir>   env PY, G
set -uo pipefail
O=$1; CODE=$(cd "$(dirname "$0")/../.." && pwd); cd $CODE
B=/mnt/cpfs/zbl-cpfs-new/CKPT/zehan/gesture-scmg; C=/mnt/cpfs/zbl-cpfs-new/dataset/zehan/gesture; X=$B/control-token-1105/ex002; T=$X/tokens
PY=${PY:-python}; G=${G:-0}; export SCMG_J=34 SCMG_NO_GLITCH=1 OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
ENC=$B/ex002-sfscope-1108/enc-s0/psync; VIEW=$C/train-v1-20261101/views/data-v36_1; A5=$O/a6; A5PREV=$O/a5
log() { echo "$(date -Is) $*" >> $O/lane_a6.log; }
log "START commit $(cat GIT_COMMIT 2>/dev/null || git rev-parse HEAD 2>/dev/null)"
mkdir -p $A5
$PY - $O/plans/s0/PLAN_tts.json $A5/NUMPLAN_tts_s0.json <<'PYEOF'
import json, sys
P = json.load(open(sys.argv[1])); ev = [e for e in P['events'] if e['cat'] == 'NUMBER']
json.dump(dict(P, events=ev, n_events=len(ev)), open(sys.argv[2], 'w')); print('NUMBER events', len(ev))
PYEOF
for k in 0 1 2 3 4 5 6 7; do $PY scripts/ex002/ex002_keyframe.py insert --version 9c --hold-pool $T/HOLD_POOL_DEICTIC_TRAIN.npz --pool point --release-search --palm-fix --finger-hold 0.2,1.0 \
  --fo-library $T/HAND_LIBRARY_TRAIN.npz --given $O/final/tts/s0 --plan $A5/NUMPLAN_tts_s0.json --out $A5/tts_s0 --mode given --shard $k/8 >> $O/lane_a6.log 2>&1 & done; wait $(jobs -p)
for g in $ENC/gt_*.npy; do ln -sf $g $A5/; done; log "INSERTED"
$PY scripts/ex002/expl_canon.py eval --plan $O/plans/s0/PLAN_tts.json --none $ENC/s0 --out $A5/EVAL_tts_s0.json before=$O/final/tts/s0 a5=$A5PREV/tts_s0 a6=$A5/tts_s0 >> $O/lane_a6.log 2>&1 && log "EVAL"
$PY scripts/ex002/ex002_present.py eval --plan $A5/NUMPLAN_tts_s0.json --out $A5/PRESENT.json none=$ENC/s0 before=$O/final/tts/s0 a5=$A5PREV/tts_s0 a6=$A5/tts_s0 >> $O/lane_a6.log 2>&1 && log "PRESENT"
$PY scripts/ex002/expl_canon.py readable --plan $A5/NUMPLAN_tts_s0.json --out $A5/READABLE.json before=$O/final/tts/s0 a5=$A5PREV/tts_s0 a6=$A5/tts_s0 >> $O/lane_a6.log 2>&1 && log "READABLE"
CUDA_VISIBLE_DEVICES=$G $PY scripts/ex002/bench_ex002.py eval --out $A5/BENCH_EVAL.json none=$ENC/s0 a5=$A5PREV/tts_s0 a6=$A5/tts_s0 >> $O/lane_a6.log 2>&1 && log "BENCH_EVAL"
$PY scripts/ex002/ex002_exec.py --view $VIEW --plan $A5/NUMPLAN_tts_s0.json --out $A5/EXEC.json a5=$A5PREV/tts_s0 a6=$A5/tts_s0 >> $O/lane_a6.log 2>&1 && log "EXEC"
chmod -R a+rwX $O 2>/dev/null; log "A6_DONE"
