#!/bin/bash
# 11-09 explicit semantics amendment 4 (EXPLICIT_SEM_1109.md): YOU / POINT IK point keyframes on top of final/<proto>_chest of the
# amendment-3 run; eval + side gates as before.   ex002_expl_a4.sh <amendment-3 run dir>   env PY, G
set -uo pipefail
O=$1; CODE=$(cd "$(dirname "$0")/../.." && pwd); cd $CODE
X=/mnt/cpfs/zbl-cpfs-new/CKPT/zehan/gesture-scmg/control-token-1105/ex002; NONE=$X/p2-quick-1109/psync/none_s0
PY=${PY:-python}; G=${G:-0}; export SCMG_J=34 SCMG_NO_GLITCH=1 OMP_NUM_THREADS=${OMP_NUM_THREADS:-2}
log() { echo "$(date -Is) $*" >> $O/lane_a4.log; }
log "START commit $(cat GIT_COMMIT 2>/dev/null || git rev-parse HEAD 2>/dev/null)"
for p in tts psync; do
  $PY scripts/ex002/expl_canon.py keyframe --point-only --plan $O/plans/PLAN_$p.json --run $O/final/${p}_chest --out $O/final/${p}_point >> $O/lane_a4.log 2>&1 && log "POINT_KEYS $p"
  $PY scripts/ex002/expl_canon.py eval --plan $O/plans/PLAN_$p.json --none $NONE --out $O/EVAL_a4_$p.json given=$O/final/$p chest=$O/final/${p}_chest point=$O/final/${p}_point >> $O/eval_a4.log 2>&1 && log "EVAL $p"
done
CUDA_VISIBLE_DEVICES=$G $PY scripts/ex002/bench_ex002.py eval --out $O/BENCH_EVAL_a4.json tts_point=$O/final/tts_point psync_point=$O/final/psync_point >> $O/eval_a4.log 2>&1 && log "BENCH_EVAL"
CUDA_VISIBLE_DEVICES=$G $PY scripts/ex002/ex002_fgd_outside.py --plan $O/plans/PLAN_tts.json --radius 0,1.0 --out $O/FGDOUT_a4_tts.json none=$NONE tts_point=$O/final/tts_point >> $O/eval_a4.log 2>&1 && log "FGDOUT"
chmod -R a+rwX $O 2>/dev/null; log "A4_DONE"
