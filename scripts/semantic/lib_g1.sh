#!/bin/bash
# 11-07 item 5: fixed-action library insertion on one generator, dev 400, per_clip seeds 0 / 1 / 2 (criteria: docs/criteria/
# SEMLINE_7_55.md, item 5 section, written before the run). Runs per seed: base (no insertion) and point-gt / count-gt / class-gt /
# class-lex / class-llm insertions (infer_test.py --lib, official benchmark each); then g1_ctl_eval.py (STG and count hit against the
# given tokens), lib_boundary_jerk.py (vs base, same seed), R-half recognition (the library is K-half), CBF-QP + executability (seed 0).
#   lib_g1.sh <out dir>
#   env: CK DEC ASSETS LIB CTLC PRED VIEW REC_DATA REC_GLOB SAFE PY GPU PAR GPU_PER_SEED (1: seed k on GPU k) (+ bench.sh paths REF_DIR WAV_DIR VIEW_TEST TEXT_TEST STATS)
set -uo pipefail
O=$1; CODE=$(cd "$(dirname "$0")/../.." && pwd); cd $CODE; mkdir -p $O
PY=${PY:-python}; ST=$ASSETS/dev_stems.txt
export SCMG_J=39 SCMG_NO_GLITCH=1 CFGW=2.5 BENCH_STEPS=10 VSPLIT=val PY
log() { echo "$(date -Is) $*" >> $O/lane.log; }
log "START commit $(git rev-parse HEAD 2>/dev/null) ck $CK lib $LIB"
extra() {
  case $1 in
    base) echo "";;
    point-gt) echo "--lib $LIB --lib-ctl $CTLC --lib-what point --lib-mode gt";;
    count-gt) echo "--lib $LIB --lib-ctl $CTLC --lib-what count --lib-mode gt";;
    class-gt) echo "--lib $LIB --lib-ctl $CTLC --lib-what class --lib-mode gt";;
    class-lex) echo "--lib $LIB --lib-ctl $CTLC --lib-what class --lib-mode lex";;
    class-llm) echo "--lib $LIB --lib-ctl $CTLC --lib-what class --lib-mode llm --ctl-pred $PRED";;
  esac
}
one() {   # run seed
  local r=$1 sd=$2 P=$O/$1/seed_00$2
  [ -f $O/$r/benchmark_seed_00$sd.json ] && return 0
  local g=${GPU:-0}; [ "${GPU_PER_SEED:-0}" = 1 ] && g=$sd
  GPUS=$g INFER_EXTRA="--save50 --seed-mode per_clip $(extra $r)" bash scripts/echog_g1/bench.sh scmg $CK $DEC $P offline $sd $ST \
    >> $O/bench_$r.log 2>&1 || log "BENCH_FAILED $r s$sd"
}
RUNS="base point-gt count-gt class-gt class-lex class-llm"
for sd in 0 1 2; do for r in $RUNS; do
  while [ $(jobs -rp | wc -l) -ge ${PAR:-3} ]; do sleep 5; done
  one $r $sd &
done; done; wait
log "RUNS_DONE"
for sd in 0 1 2; do
  $PY scripts/semantic/g1_ctl_eval.py --ctl $CTLC --view $VIEW --out $O/CTL_EVAL_s$sd.json point=$O/point-gt/seed_00$sd \
    count=$O/count-gt/seed_00$sd base=$O/base/seed_00$sd >> $O/ctl_eval.log 2>&1 || log "CTL_EVAL_FAILED s$sd"
  $PY scripts/semantic/lib_boundary_jerk.py --base $O/base/seed_00$sd --out $O/JERK_s$sd.json $(for r in point-gt count-gt class-gt class-lex class-llm; do
    echo $r=$O/$r/seed_00$sd; done) >> $O/jerk.log 2>&1 || log "JERK_FAILED s$sd"
done
sp=(); for r in $RUNS; do for sd in 0 1 2; do sp+=(lib-$r-pc-s$sd=$O/$r/seed_00$sd); done; done
PY=$PY PAIRS= bash scripts/semantic/sem_score_dirs.sh ${GPU:-0} $REC_DATA "$REC_GLOB" $O/semR dev gt=- "${sp[@]}" > $O/semR.log 2>&1
$PY scripts/semantic/sem_multiseed.py --scores $O/semR --rec-prefix rec-sem-hosR-seed --seeds 0 1 2 --split dev --out $O/MULTISEED_R.json \
  --sources lib-base-pc lib-class-gt-pc lib-class-lex-pc lib-class-llm-pc lib-point-gt-pc lib-count-gt-pc \
  --pairs lib-class-gt-pc:lib-base-pc,lib-class-lex-pc:lib-base-pc,lib-class-llm-pc:lib-base-pc,lib-point-gt-pc:lib-base-pc,lib-count-gt-pc:lib-base-pc \
  >> $O/semR.log 2>&1 && log "MULTISEED_DONE"
if [ -n "${SAFE:-}" ]; then
  for r in point-gt count-gt class-gt class-lex class-llm; do
    D=$O/safe/$r; mkdir -p $D
    (cd $SAFE && PYTHONPATH=${OFF_SRC}:${SAFE_DEPS:-} $PY scripts/echog_g1/safety_run.py --pred-dir $O/$r/seed_000 --stems $ST --out-root $D \
      --layers none,cbf_qp --workers ${SAFE_WORKERS:-6} > $D/safe.log 2>&1 && PYTHONPATH=${OFF_SRC}:${SAFE_DEPS:-} $PY scripts/echog_g1/executability.py \
      --ref-dir $REF_DIR --stems $ST --urdf $G1/g1_custom_collision_29dof.urdf --mjcf $G1/g1_mocap_29dof.xml --collision-stride 1 \
      --exclusion-file scripts/echog_g1/g1_exclusion_train.json --out $D/EXEC.json raw=$D/none cbf_qp=$D/cbf_qp >> $D/safe.log 2>&1) \
      || log "SAFE_FAILED $r"
    rm -rf $D/none $D/cbf_qp/*.pt
  done
fi
log "LANE_DONE"
