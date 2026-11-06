#!/bin/bash
# Local CPU post-processing of the Div/BA dev runs (no GPU): for every finished prediction dir (benchmark_seed_XXX.json
# present) under the given roots: copy predictions locally, diag_beats.py (BA lift / shape / Div with bootstrap SE), CBF-QP
# safety layer (exp/safe-proj safety_run.py, layers none + cbf_qp, K = 10) and the independent executability checker on raw
# and filtered outputs. Idempotent; loops until STOP file exists or all roots report JOB_DONE (--once: single pass).
#   postproc.sh <local out root> <tag>=<cpfs run root> [...]
# Run-root layouts: <root>/<cfg>/<proto>/seed_00X (dlc_infer.sh) and <root>/dev/<lane>/bench/<proto>/seed_00X (lanes).
set -uo pipefail
OUT=$1; shift; mkdir -p $OUT
PY=/home/xr-25b7/envs/part2-exec-20261029/bin/python
OFF=/home/xr-25b7/bulk/echog-official-ro/official; SAFE=/home/xr-25b7/projects/gesture/scmg-wt-safeproj
EG=/mnt/bulk/zehan/echog-dataset; DIV=$(cd "$(dirname "$0")" && pwd)
S=/tmp/claude-1000/-home-xr-25b7-projects-gesture-scmg/a38031be-89bb-49ab-9285-885953dd878a/scratchpad/divba
G1=/home/xr-25b7/bulk/g1-assets/unitree_g1
export OMP_NUM_THREADS=2
one() {  # <tag> <pred dir on cpfs> <label>
  local tag=$1 P=$2 lab=$3 D=$OUT/$1/$3
  [ -f $D/DONE ] && return; mkdir -p $D
  rsync -a --include='*.pt' --exclude='*' $P/ $D/pred/ || return
  [ $(ls $D/pred/*.pt 2>/dev/null | wc -l) -ge 400 ] || { echo "$(date -Is) SHORT $tag $lab"; return; }
  PYTHONPATH=$OFF/src nice $PY $DIV/diag_beats.py --mmae-file $EG/official-meta/eval_assets/mmae/g1_mmae_30body_30fps.npy \
    --ref-dir $EG/ds/motion_39d_30fps --words-dir $S/words-dev --wav-dir $EG/ds/audio --stems $S/dev_stems.txt \
    --out $D/diag.json $lab=$D/pred > $D/diag.log 2>&1 || { echo "$(date -Is) DIAG_FAILED $tag $lab"; return; }
  (cd $SAFE && PYTHONPATH=$OFF/src:/home/xr-25b7/bulk/pydeps/safeproj nice $PY scripts/echog_g1/safety_run.py --pred-dir $D/pred \
    --stems $S/dev_stems.txt --out-root $D/safe --layers none,cbf_qp --workers 6 > $D/safe.log 2>&1 \
   && PYTHONPATH=$OFF/src:/home/xr-25b7/bulk/pydeps/safeproj nice $PY scripts/echog_g1/executability.py --ref-dir $EG/ds/motion_39d_30fps \
    --stems $S/dev_stems.txt --urdf $G1/g1_custom_collision_29dof.urdf --mjcf $G1/g1_mocap_29dof.xml --collision-stride 1 \
    --exclusion-file scripts/echog_g1/g1_exclusion_train.json --out $D/EXEC.json raw=$D/safe/none cbf_qp=$D/safe/cbf_qp >> $D/safe.log 2>&1) \
    || { echo "$(date -Is) SAFE_FAILED $tag $lab"; return; }
  rm -rf $D/safe/none $D/safe/cbf_qp/*.pt; touch $D/DONE; echo "$(date -Is) POST_DONE $tag $lab"
}
while :; do
  for spec in "$@"; do
    tag=${spec%%=*}; R=${spec#*=}
    for bj in $R/*/*/benchmark_seed_*.json $R/dev/*/bench/*/benchmark_seed_*.json; do
      [ -f $bj ] || continue
      pd=$(dirname $bj); sd=$(basename $bj .json); sd=${sd#benchmark_}; proto=$(basename $pd)
      cfg=$(basename $(dirname $pd)); [ $cfg = bench ] && cfg=$(basename $(dirname $(dirname $pd)))
      one $tag $pd/$sd ${cfg}_${proto}_${sd}
    done
  done
  [ "${ONCE:-0}" = 1 ] && break; [ -f $OUT/STOP ] && break
  sleep 120
done
