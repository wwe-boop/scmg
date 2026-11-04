#!/bin/bash
# 11-04 user: improve the r4 main line (chunk planner + e5 text), train while validating.
#   r4_lane.sh <gpu> <out root> <lane> <cfg-json (addable switches)> <sc-json> [steps=6000]
# Fine-tunes r4 base (base stage, lr 1e-4) with EMA snapshots every SNAP steps; every snapshot (and final.pt) gets the
# QUICK baseline protocol (rollout 20 clips + semantic 60 clips, seed 0, offline AND engine-streaming); final.pt also gets
# the full tier (FGD gen + rollout/semantic seeds 0/1).  Inference fixes are default-on in the code.
set -uo pipefail
gpu=$1; R=$2; o=$3; cf=$4; scj=$5; ST=${6:-6000}; SNAP=${SNAP:-2000}
CODE=$(cd "$(dirname "$0")/.." && pwd); cd $CODE
export PYTHONPATH=$CODE${PYTHONPATH_EXTRA:+:$PYTHONPATH_EXTRA} USE_TF=0 OMP_NUM_THREADS=${OMP_NUM_THREADS:-4} CUDA_VISIBLE_DEVICES=$gpu
export SCMG_TEXT_CAUSAL=/mnt/cpfs/zbl-cpfs-new/CKPT/zehan/gesture-scmg/models/text-e5-causal-v36_1
PY=${PY:-python}; B=/mnt/cpfs/zbl-cpfs-new/CKPT/zehan/gesture-scmg; D=/mnt/cpfs/zbl-cpfs-new/dataset/zehan/gesture/scmg
INIT=${INIT:-$B/dlc-scmg-train-r4-1102/base/best.pt}; VAE=$B/trained/vae-r3/final.pt; SEMMAN=$B/semantic-recognizer-v1/manifest-n200
mkdir -p $R/$o; log(){ echo "$(date -Is) $*" >> $R/$o/lane.log; }
if [ ! -f $R/$o/DONE.json ]; then
  $PY -m scmg.train --out $R/$o --init $INIT --stats $D/STATS_v36_1.json --text-cache $D/text-e5-v36_1 --vae $VAE --workers ${WORKERS:-6} \
    --tok-cache $D/tok-qwen25-v36_1 --tok-intent $B/models/intent-qwen25-v36_1 --intent-frames $B/models/intent-frames-v36_1 \
    --holdout $SEMMAN/MANIFEST.json --semsynth-share ${SEMSHARE:-0.4} --batch ${BATCH:-64} --stage base --steps $ST --lr ${LR:-1e-4} \
    --warmup 200 --eval-every 1000 --snap-every $SNAP --seed ${TRAIN_SEED:-0} --cfg "$cf" --sc "$scj" ${BAGGER_DIR:+--bagger-dir $BAGGER_DIR --bagger-share ${BAGGER_SHARE:-0.33}} >> $R/$o/log.txt 2>&1 &
  TP=$!; log "TRAIN_START pid $TP"
else TP=; fi
evalq() {  # <ckpt> <label>
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
for P in offline engine; do for part in fgd probes; do NOVOX=1 bash $CODE/scripts/baseline_eval.sh $gpu $R/$o/eval final $R/$o/final.pt $P $part ${CFGW:-2.5} || log "FULL_FAILED $P $part"; done; done
log "LANE_DONE"
