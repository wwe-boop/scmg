#!/bin/bash
# 11-05 multi-seed confirmation of the selected G1 configuration (decoder + inference chosen on the held-out-speaker dev):
# official benchmark, offline, for each <generator ckpt> x inference seed, plus MM20 on the 300-stem subset.
#   confirm_g1.sh <vae/decoder> <out root> <label=ckpt> ...     env SEEDS (inference seeds, default "0 1"), GPUS, MM_CK (label for MM20)
set -uo pipefail
CODE=$(cd "$(dirname "$0")/../.." && pwd); cd $CODE
vae=$1; O=$2; shift 2; mkdir -p $O; GL=(${GPUS:-0}); n=${#GL[@]}; i=0
log() { echo "$(date -Is) $*" >> $O/confirm.log; }
for spec in "$@"; do
  lab=${spec%%=*}; ck=${spec#*=}
  for p in ${PROTOS:-offline}; do for sd in ${SEEDS:-0 1}; do        # PROTOS: offline and/or engine (EX002 streaming protocol)
    g=${GL[$((i % n))]}; i=$((i + 1))
    ( [ -f $O/$lab/benchmark_${p}_seed_00$sd.json ] || { mkdir -p $O/$lab; GPUS="$g $g $g" bash scripts/echog_g1/bench.sh scmg $ck $vae $O/$lab/${p}_seed_00$sd $p $sd >> $O/$lab/bench.log 2>&1 && log "BENCH_DONE $lab $p s$sd" || log "BENCH_FAILED $lab $p s$sd"; } ) &
    [ $((i % n)) -eq 0 ] && wait                                         # at most one bench per GPU at a time
  done; done
done
wait
if [ -n "${MM_CK:-}" ]; then
  for spec in "$@"; do [ "${spec%%=*}" = "$MM_CK" ] && mck=${spec#*=}; done
  GPUS="${GL[*]}" EGPU=${GL[0]} bash scripts/echog_g1/mm20_subset.sh $mck $vae $O/mm20-$MM_CK /mnt/cpfs/zbl-cpfs-new/CKPT/zehan/gesture-scmg/baseline-repro-1104/robogesture/subset_stems.txt \
    && log "MM20_DONE $MM_CK" || log "MM20_FAILED $MM_CK"
fi
log CONFIRM_DONE
