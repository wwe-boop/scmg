#!/bin/bash
# 11-04 DLC pool for stress_eval.py.  Queue lines: <label> <ckpt> <cfg_w> <tm> <H> <text> <seed> <conds(comma)>
set -u
REL=${REL:?code release}; R=${R:?out root}; Q=${Q:?queue}; K=${K:-3}; PY=${PY:-/usr/local/bin/python}
export SCMG_TEXT_CAUSAL=/mnt/cpfs/zbl-cpfs-new/CKPT/zehan/gesture-scmg/models/text-e5-causal-v36_1 USE_TF=0
mkdir -p $R/locks $R/rows
worker() {
  local gpu=$1 slot=$2 lab ck w tm h tx sd cs j rc t0
  while read -r lab ck w tm h tx sd cs; do
    [ -n "$lab" ] || continue; j=${lab}__${tm}-${h}-${tx}-s${sd}-${cs//,/+}
    [ -f $R/rows/$j.DONE ] && continue; mkdir $R/locks/$j 2>/dev/null || continue
    t0=$(date +%s); echo "$(date +%T) gpu$gpu.$slot START $j"
    ( cd $REL && PYTHONPATH=$REL CUDA_VISIBLE_DEVICES=$gpu OMP_NUM_THREADS=2 STRESS_TM=$tm STRESS_H=$h STRESS_TEXT=$tx STRESS_SEED=$sd STRESS_CONDS=$cs \
      $PY scripts/stress_eval.py $R/rows/$j.jsonl $R/../feats $lab=$ck@$w > $R/rows/$j.log 2>&1 ); rc=$?
    [ $rc = 0 ] && touch $R/rows/$j.DONE
    echo "$(date +%T) gpu$gpu.$slot END $j rc=$rc $(( $(date +%s) - t0 ))s"
  done < $Q
}
NG=$(nvidia-smi -L | wc -l); echo "$(date) gpus=$NG K=$K rel=$REL q=$Q"
for g in $(seq 0 $((NG-1))); do for k in $(seq 1 $K); do worker $g $k & sleep 3; done; done
wait; echo "$(date) ALL_QUEUES_DRAINED"
