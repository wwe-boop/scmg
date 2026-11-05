#!/bin/bash
# 11-04 user: improve the r4 main line (chunk planner + e5 text), train while validating.
#   r4_lane.sh <gpu> <out root> <lane> <cfg-json (addable switches)> <sc-json> [steps=6000]
# Fine-tunes r4 base (base stage, lr 1e-4) with EMA snapshots every SNAP steps; every snapshot (and final.pt) gets the
# QUICK baseline protocol (rollout 20 clips + semantic 60 clips, seed 0, offline AND engine-streaming); final.pt also gets
# the full tier (FGD gen + rollout/semantic seeds 0/1).  Inference fixes are default-on in the code.
# PROFILE=g1 (11-04): the same lane on the ECHO-G G1 view (scripts/lane_profile.sh; INIT + VAE required). Evaluation there:
# rollout/stillness probe (dev split, arms only) offline + engine for every snapshot and final.pt; with G1_BENCH=1 final.pt
# also runs the official ECHO-G benchmark on val_common (scripts/echog_g1/bench.sh; G1_BENCH_STEMS = stem list, default
# all 3,242; seeds G1_BENCH_SEEDS, default 0 1) offline AND engine.
set -uo pipefail
gpu=$1; R=$2; o=$3; cf=$4; scj=$5; ST=${6:-6000}; SNAP=${SNAP:-2000}
CODE=$(cd "$(dirname "$0")/.." && pwd); cd $CODE
export PYTHONPATH=$CODE${PYTHONPATH_EXTRA:+:$PYTHONPATH_EXTRA} USE_TF=0 OMP_NUM_THREADS=${OMP_NUM_THREADS:-4} CUDA_VISIBLE_DEVICES=$gpu
PY=${PY:-python}
source $CODE/scripts/lane_profile.sh
mkdir -p $R/$o; log(){ echo "$(date -Is) $*" >> $R/$o/lane.log; }
if [ ! -f $R/$o/DONE.json ]; then
  SCMG_EXTRA_MASK=${EXTRA_MASK_TRAIN:-} $PY -m scmg.train --out $R/$o --init $INIT "${DATA_ARGS[@]}" --vae $VAE --workers ${WORKERS:-6} \
    --semsynth-share ${SEMSHARE:-0.4} --batch ${BATCH:-64} --stage base --steps $ST --lr ${LR:-1e-4} \
    --warmup 200 --eval-every 1000 --snap-every $SNAP --seed ${TRAIN_SEED:-0} --cfg "$cf" --sc "$scj" ${ANCHOR:+--anchor-share "$ANCHOR"} ${BAGGER_DIR:+--bagger-dir $BAGGER_DIR --bagger-share ${BAGGER_SHARE:-0.33}} ${CAUSAL_DIR:+--causal-dir $CAUSAL_DIR} >> $R/$o/log.txt 2>&1 &
  TP=$!; log "TRAIN_START pid $TP"
else TP=; fi
evalq() {  # <ckpt> <label>
  if [ "${PROFILE:-ex002}" = g1 ]; then          # G1: stillness / speed probe on the dev split, offline + engine
    mkdir -p $R/$o/probe
    local sd; for sd in ${PROBE_SEEDS:-0}; do
      ROLLOUT_SEED=$sd ROLLOUT_VIEW=$VIEW ROLLOUT_STATS=$STATS ROLLOUT_TEXT=$TEXT $PY scripts/rollout_probe.py $VAE $2=$1@${CFGW:-2.5} >> $R/$o/probe/offline.jsonl 2>> $R/$o/probe/probe.log || log "PROBE_FAILED $2 offline"
      ROLLOUT_SEED=$sd ROLLOUT_VIEW=$VIEW ROLLOUT_STATS=$STATS ROLLOUT_TEXT=$TEXT ROLLOUT_TEXT_MODE=x2 ROLLOUT_X2_LAG=0.15 ROLLOUT_H=2 \
        $PY scripts/rollout_probe.py $VAE $2=$1@${CFGW:-2.5} >> $R/$o/probe/engine.jsonl 2>> $R/$o/probe/probe.log || log "PROBE_FAILED $2 engine"; done
    log "PROBE_DONE $2"; return
  fi
  if [ "${ROLLOUT_ONLY:-0}" = 1 ]; then         # stillness curve only: rollout 20 clips, seeds 0/1, offline + engine
    mkdir -p $R/$o/curve
    local sd; for sd in 0 1; do ROLLOUT_SEED=$sd ROLLOUT_WINDOW_LM_TEXT=1 $PY scripts/rollout_probe.py $VAE $2=$1@${CFGW:-2.5} >> $R/$o/curve/offline.jsonl 2>> $R/$o/curve/probe.log
      ROLLOUT_SEED=$sd ROLLOUT_TEXT_MODE=x2 ROLLOUT_X2_LAG=0.15 ROLLOUT_H=2 $PY scripts/rollout_probe.py $VAE $2=$1@${CFGW:-2.5} >> $R/$o/curve/engine.jsonl 2>> $R/$o/curve/probe.log; done
    log "CURVE_DONE $2"; return
  fi
  for P in offline engine; do QUICK=1 NOVOX=1 bash $CODE/scripts/baseline_eval.sh $gpu $R/$o/eval $2 $1 $P probes ${CFGW:-2.5} || log "QUICK_FAILED $2 $P"; done
  log "QUICK_DONE $2"
}
done_snaps=""
while :; do
  for f in $(ls $R/$o/snap_*.pt 2>/dev/null); do
    s=$(basename $f .pt); case " $done_snaps " in *" $s "*) continue;; esac
    sleep 30; evalq $f $s; done_snaps="$done_snaps $s"
  done
  if [ -n "$TP" ] && kill -0 $TP 2>/dev/null; then sleep 60; continue; fi
  left=0; for f in $(ls $R/$o/snap_*.pt 2>/dev/null); do case " $done_snaps " in *" $(basename $f .pt) "*) ;; *) left=1;; esac; done
  [ $left = 1 ] && continue                    # snapshots written while an evaluation was running (11-04 fix)
  break
done
[ -n "$TP" ] && { wait $TP; rc=$?; log "TRAIN_EXIT $rc"; }
[ -f $R/$o/final.pt ] || { log "NO_FINAL"; exit 1; }
rm -f $R/$o/resume.pt
[ "${ROLLOUT_ONLY:-0}" = 1 ] && { evalq $R/$o/final.pt final; log "LANE_DONE"; exit 0; }
if [ "${PROFILE:-ex002}" = g1 ]; then
  evalq $R/$o/final.pt final
  if [ "${G1_BENCH:-0}" = 1 ]; then
    for P in ${BENCH_PROTOS:-offline engine}; do for sd in ${G1_BENCH_SEEDS:-0 1}; do
      PY=${BENCH_PY:-$PY} GPUS="$(for r in $(seq ${SPG:-1}); do printf "%s " $gpu; done)" CFGW=${CFGW:-2.5} bash $CODE/scripts/echog_g1/bench.sh scmg $R/$o/final.pt $VAE $R/$o/bench/$P/seed_$(printf %03d $sd) $P $sd ${G1_BENCH_STEMS:-} \
        >> $R/$o/bench.log 2>&1 && log "BENCH_DONE $P s$sd" || log "BENCH_FAILED $P s$sd"
    done; done
  fi
  log "LANE_DONE"; exit 0
fi
for P in offline engine; do for part in fgd probes; do NOVOX=1 bash $CODE/scripts/baseline_eval.sh $gpu $R/$o/eval final $R/$o/final.pt $P $part ${CFGW:-2.5} || log "FULL_FAILED $P $part"; done; done
log "LANE_DONE"
