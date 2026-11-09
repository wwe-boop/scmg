#!/bin/bash
# 11-04 ECHO-G same-data round, ONE DLC job (user: evaluate inside the job, do not stop after training):
#   per training seed s (in parallel, GPS GPUs each):
#     1) official benchmark of the original recipe base-s<s>/best.pt on val_common: offline + engine x inference seeds 0/1
#     2) BAgger collection from that base + the improved lane (stream + BAgger, 2000 steps, fully trained -> final.pt)
#     3) official benchmark of the improved final.pt, same four settings
#   g1_round.sh <G1 training root (vae/, base-s*/)> <out root>     env SEEDS="0 1 2" GPS=2 SPG=3 GPUS="0 1 ..." + bench.sh paths
set -uo pipefail
PY=${PY:-python}; R=$1; O=$2; CODE=$(cd "$(dirname "$0")/../.." && pwd); cd $CODE; mkdir -p $O
GL=(${GPUS:-$(seq 0 $(($(nvidia-smi -L | wc -l) - 1)))}); GPS=${GPS:-2}; i=0
log() { echo "$(date -Is) $*" >> $O/round.log; }
bench4() {  # <ckpt> <out dir> <gpus>
  local p sd; for p in offline engine; do for sd in 0 1; do
    [ -f $2/benchmark_${p}_seed_00$sd.json ] && continue
    PY=${BENCH_PY:-$PY} GPUS="$3" bash scripts/echog_g1/bench.sh scmg $1 $R/vae/final.pt $2/${p}_seed_00$sd $p $sd >> $2/bench.log 2>&1 || log "BENCH_FAILED $2 $p $sd"
  done; done
}
for s in ${SEEDS:-0 1 2}; do
  G="${GL[@]:$((i * GPS)):$GPS}"; i=$((i + 1))
  GB=$(for g in $G; do for r in $(seq ${SPG:-3}); do printf "%s " $g; done; done)   # SPG inference shards per GPU (small model: fill the GPU)
  (
    mkdir -p $O/base-s$s; bench4 $R/base-s$s/best.pt $O/base-s$s "$GB"; log "BASE_BENCH_DONE s$s"
    PROFILE=g1 INIT=$R/base-s$s/best.pt VAE=$R/vae/final.pt LANES=b STEPS=2000 TRAIN_SEED=$s GPUS="$G" \
      bash scripts/bagger_r4.sh $O/improve-s$s >> $O/improve-s$s.log 2>&1 || log "IMPROVE_FAILED s$s"
    F=$O/improve-s$s/r4-stream-bag/final.pt
    if [ -f $F ]; then mkdir -p $O/improve-s$s/bench; bench4 $F $O/improve-s$s/bench "$GB"; log "IMPROVE_BENCH_DONE s$s"; else log "NO_IMPROVED_FINAL s$s"; fi
  ) &
done
wait; log "G1_ROUND_DONE"
