#!/bin/bash
# 11-04 main line, formal comparison: {stream+BAgger, stream+intent+BAgger} x training seeds, from the r4 base, 2000-step
# schedule (fully trained = final.pt, no snapshot picking), BAgger data = the r4 round (bagger-pai/bagger-data, collected
# from the r4 base).  One lane per GPU; every lane ends with the full tier (offline + engine, FGD + probes, seeds 0/1).
#   mainline_seeds.sh <out root>     env SEEDS="0 1 2" LANES="sbag sibag" STEPS=2000 GPUS="0 1 ..." SKIP="sbag-s1" (lanes already trained elsewhere)
set -uo pipefail
R=$1; CODE=$(cd "$(dirname "$0")/.." && pwd); cd $CODE; mkdir -p $R
ARM_MASK=${ARM_MASK:-/mnt/cpfs/zbl-cpfs-new/USERS/zehan/gesture/pose-audit-1104/full/ARM_MASK.npy}   # sibag-arm: pose-audit arm frames off the losses
export BAGGER_DIR=${BAGGER_DIR:-/mnt/cpfs/zbl-cpfs-new/CKPT/zehan/gesture-scmg/r4-1104/bagger-pai/bagger-data} BAGGER_SHARE=${BAGGER_SHARE:-0.33}
[ -f $BAGGER_DIR/MANIFEST.json ] || { echo "no BAgger data at $BAGGER_DIR"; exit 1; }
export SNAP=${SNAP:-1000000}                 # no intermediate snapshots: the formal runs are judged on final.pt only
S='"p_stream": 0.67, "p_async": 1.0, "p_async_seg": 0.0'
GL=(${GPUS:-$(seq 0 $(($(nvidia-smi -L | wc -l) - 1)))}); i=0
for k in ${SEEDS:-0 1 2}; do for l in ${LANES:-sbag sibag}; do
  case " ${SKIP:-} " in *" $l-s$k "*) echo "skip $l-s$k (reuse the existing run)"; continue;; esac
  g=${GL[$((i % ${#GL[@]}))]}; i=$((i + 1))
  case $l in
    sbag)  TRAIN_SEED=$k bash scripts/r4_lane.sh $g $R sbag-s$k '{"adsc": true}' "{$S}" ${STEPS:-2000} > $R/lane-sbag-s$k.log 2>&1 & ;;
    sibag) TRAIN_SEED=$k SEMSHARE=0.6 bash scripts/r4_lane.sh $g $R sibag-s$k '{"adsc": true, "intent_aux": true}' "{$S, \"w_int\": 0.3, \"sem_w\": 2.0}" ${STEPS:-2000} > $R/lane-sibag-s$k.log 2>&1 & ;;
    sibag-tf)   TRAIN_SEED=$k SEMSHARE=0.6 bash scripts/r4_lane.sh $g $R sibag-tf-s$k '{"adsc": true, "intent_aux": true}' "{$S, \"w_int\": 0.3, \"sem_w\": 2.0, \"text_focus_p\": 0.2}" ${STEPS:-2000} > $R/lane-sibag-tf-s$k.log 2>&1 & ;;
    sibag-int6) TRAIN_SEED=$k SEMSHARE=0.6 bash scripts/r4_lane.sh $g $R sibag-int6-s$k '{"adsc": true, "intent_aux": true}' "{$S, \"w_int\": 0.6, \"sem_w\": 2.0}" ${STEPS:-2000} > $R/lane-sibag-int6-s$k.log 2>&1 & ;;
    sbag-phys03) TRAIN_SEED=$k bash scripts/r4_lane.sh $g $R sbag-phys03-s$k '{"adsc": true}' "{$S, \"w_phys\": 0.03}" ${STEPS:-2000} > $R/lane-sbag-phys03-s$k.log 2>&1 & ;;
    sbag-ftx)  TRAIN_SEED=$k bash scripts/r4_lane.sh $g $R sbag-ftx-s$k '{"adsc": true, "flow_text_xattn": true}' "{$S}" ${STEPS:-2000} > $R/lane-sbag-ftx-s$k.log 2>&1 & ;;
    sibag-ftx) TRAIN_SEED=$k SEMSHARE=0.6 bash scripts/r4_lane.sh $g $R sibag-ftx-s$k '{"adsc": true, "intent_aux": true, "flow_text_xattn": true}' "{$S, \"w_int\": 0.3, \"sem_w\": 2.0}" ${STEPS:-2000} > $R/lane-sibag-ftx-s$k.log 2>&1 & ;;
    sibag-arm) EXTRA_MASK_TRAIN=$ARM_MASK TRAIN_SEED=$k SEMSHARE=0.6 bash scripts/r4_lane.sh $g $R sibag-arm-s$k '{"adsc": true, "intent_aux": true}' "{$S, \"w_int\": 0.3, \"sem_w\": 2.0}" ${STEPS:-2000} > $R/lane-sibag-arm-s$k.log 2>&1 & ;;
    s6bag) TRAIN_SEED=$k SEMSHARE=0.6 bash scripts/r4_lane.sh $g $R s6bag-s$k '{"adsc": true}' "{$S}" ${STEPS:-2000} > $R/lane-s6bag-s$k.log 2>&1 & ;;   # intent control: sibag data mix without intent
    *) echo "unknown lane $l"; exit 2 ;;
  esac
done; done
wait
if [ -n "${ARM_PY:-}" ]; then   # landing metrics for the arms (scripts/landing/arm_audit.py) on every lane's final exports, CPU
  A=""; for d in $R/*/eval/final/{offline,engine}/eval-s0; do [ -d $d ] && A="$A $(basename $(dirname $(dirname $(dirname $(dirname $d)))))-$(basename $(dirname $d))=$d"; done
  OPENBLAS_NUM_THREADS=1 $ARM_PY scripts/landing/arm_audit.py $R/arms.jsonl $A > $R/arms.log 2>&1 && echo "$(date -Is) ARM_AUDIT_DONE" || echo "$(date -Is) ARM_AUDIT_FAILED"
fi
echo "$(date -Is) MAINLINE_SEEDS_DONE"
