#!/bin/bash
# 11-09 explicit semantics canonical FULL dev run (exp/text-path EXPLICIT_SEM_1109.md "Full dev run"): 43 recordings x seeds 0-2, TTS-sync + psync,
# 4 GPUs (generation sharded by recording), CPU keyframes / eval, pooled per-category gates, FGD v2 / liveliness / executability.
#   ex002_expl_full.sh <out dir>     env PY, NG (default 4)
set -uo pipefail
O=$1; CODE=$(cd "$(dirname "$0")/../.." && pwd); cd $CODE; mkdir -p $O; NG=${NG:-4}
B=/mnt/cpfs/zbl-cpfs-new/CKPT/zehan/gesture-scmg; C=/mnt/cpfs/zbl-cpfs-new/dataset/zehan/gesture; X=$B/control-token-1105/ex002; T=$X/tokens
PY=${PY:-python}; export SCMG_J=34 SCMG_NO_GLITCH=1 OMP_NUM_THREADS=${OMP_NUM_THREADS:-2}
ENC=$B/ex002-sfscope-1108/enc-s0/psync; CK=$X/m10-enc/b4k/final.pt; VIEW=$C/train-v1-20261101/views/data-v36_1
CIS=$(seq -s, 0 42)
BL="scripts/ex002/ex002_local_blend.py --pre 0.3 --post 0.9 --compose smooth"; FO="--finger-override $T/HAND_SHAPES.json --fo-ramp 0.3 --fo-profile minjerk --fo-library $T/HAND_LIBRARY_TRAIN.npz"
KF="scripts/ex002/expl_canon.py"
log() { echo "$(date -Is) $*" >> $O/lane.log; }
log "START commit $(cat GIT_COMMIT 2>/dev/null || git rev-parse HEAD 2>/dev/null)"
for s in 0 1 2; do for p in tts psync; do $PY $KF plan --cis $CIS --proto $p --none $ENC/s$s --out $O/plans/s$s >> $O/lane.log 2>&1; done; done; log "PLANS"
for s in 0 1 2; do for p in tts psync; do
  LA=""; [ $p = tts ] && LA="--ctl-lookahead 0.5"; G=$O/gen/$p/s$s
  [ -f $G/.done ] && continue
  for k in $(seq 0 $((NG - 1))); do CUDA_VISIBLE_DEVICES=$k $PY scripts/ex002/bench_ex002.py infer --ckpt $CK --protocol psync --seed $s --out $G \
      --plan $O/plans/s$s/TOKPLAN_$p.json --mode given $LA --shard $k/$NG >> $O/infer_$k.log 2>&1 & done; wait $(jobs -p)
  touch $G/.done; log "GEN $p s$s"
done; done
post() {  # proto seed
  local p=$1 s=$2; local F=$O/final/$p/s$s; mkdir -p $O/final/$p
  $PY $BL --none $ENC/s$s --given $O/gen/$p/s$s --plan $O/plans/s$s/TOKPLAN_$p.json --out $O/blend/$p/s$s --mode given $FO > /dev/null 2>> $O/blend.log
  $PY $KF keyframe --self-chest --plan $O/plans/s$s/PLAN_$p.json --run $O/blend/$p/s$s --out $O/chest/$p/s$s >> $O/kf.log 2>&1
  $PY $KF keyframe --point-only --plan $O/plans/s$s/PLAN_$p.json --run $O/chest/$p/s$s --out $F >> $O/kf.log 2>&1
  for g in $ENC/../gt_*.npy $ENC/gt_*.npy; do [ -f $g ] && ln -sf $g $O/final/$p/; done
  log "POST $p s$s"
}
for s in 0 1 2; do for p in tts psync; do post $p $s & done; done; wait $(jobs -p)
for p in tts psync; do
  $PY $KF eval --plan $O/plans/s0/PLAN_$p.json,$O/plans/s1/PLAN_$p.json,$O/plans/s2/PLAN_$p.json --none $ENC/s0,$ENC/s1,$ENC/s2 \
    --out $O/EVAL_$p.json given=$O/final/$p/s0,$O/final/$p/s1,$O/final/$p/s2 >> $O/eval.log 2>&1 && log "EVAL $p" &
done; wait $(jobs -p)
CUDA_VISIBLE_DEVICES=0 $PY scripts/ex002/bench_ex002.py eval --out $O/BENCH_EVAL.json none_s0=$ENC/s0 none_s1=$ENC/s1 none_s2=$ENC/s2 \
  $(for p in tts psync; do for s in 0 1 2; do echo ${p}_s$s=$O/final/$p/s$s; done; done) >> $O/eval.log 2>&1 && log "BENCH_EVAL" &
CUDA_VISIBLE_DEVICES=1 $PY scripts/ex002/ex002_fgd_outside.py --plan $O/plans/s0/PLAN_tts.json --radius 0,1.0 --out $O/FGDOUT_tts_s0.json none=$ENC/s0 tts=$O/final/tts/s0 >> $O/eval.log 2>&1 && log "FGDOUT" &
$PY scripts/ex002/ex002_exec.py --view $VIEW --plan $O/plans/s0/PLAN_tts.json --out $O/EXEC.json $(for p in tts psync; do for s in 0 1 2; do echo ${p}_s$s=$O/final/$p/s$s; done; done) >> $O/eval.log 2>&1 && log "EXEC" &
wait $(jobs -p)
chmod -R a+rwX $O 2>/dev/null; log "LANE_DONE"
