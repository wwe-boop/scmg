#!/bin/bash
# 11-04 DLC pool for the baseline protocol.  Queue lines: <label> <ckpt> <cfg_w> <vox|novox>; every line expands to
# 4 jobs: {offline,engine} x {fgd,probes}, each run by scripts/baseline_eval.sh on one GPU slot (mkdir locks, K slots/GPU).
set -u
REL=${REL:?code release}; R=${R:?out root}; Q=${Q:?queue file}; K=${K:-3}; export PY=${PY:-/usr/local/bin/python}
mkdir -p $R/locks
worker() {
  local gpu=$1 slot=$2 lab ck w vx P part j rc t0
  for P in engine offline; do for part in probes fgd; do
    while read -r lab ck w vx; do
      [ -n "$lab" ] || continue; [ -f "$ck" ] || continue; j=$lab-$P-$part
      grep -q "PART_DONE $part" $R/$lab/$P/runner.log 2>/dev/null && continue
      mkdir $R/locks/$j 2>/dev/null || continue
      t0=$(date +%s); echo "$(date +%T) gpu$gpu.$slot START $j"
      NOVOX=$([ "$vx" = novox ] && echo 1 || echo 0) OMP_NUM_THREADS=2 bash $REL/scripts/baseline_eval.sh $gpu $R $lab $ck $P $part $w; rc=$?
      echo "$(date +%T) gpu$gpu.$slot END $j rc=$rc $(( $(date +%s) - t0 ))s"
    done < $Q
  done; done
}
NG=$(nvidia-smi -L | wc -l); echo "$(date) gpus=$NG K=$K rel=$REL q=$Q"
for g in $(seq 0 $((NG-1))); do for k in $(seq 1 $K); do worker $g $k & sleep 3; done; done
wait; echo "$(date) ALL_QUEUES_DRAINED"
