#!/bin/bash
# 11-04 module screen on the ECHO-G same-data benchmark (user: ablations on the ECHO-G dataset first): the G1 base recipe
# (= train_full.sh base: r4 recipe, global batch 1024 = 128 x 8 GPUs, kept on fewer GPUs by a larger per-GPU batch, 8000 steps, training seed 0, shared G1 VAE) with ONE
# architecture change, then the official benchmark of best.pt in the SAME job (val_common 3242, offline + engine x inference
# seeds 0/1, CFG 2.5).  Control = <train root>/base-s0 (benchmarked by g1_round.sh).
#   arch_g1.sh <G1 train root (vae/)> <out root> <label> <cfg-json>     env SPG=3 + bench.sh paths, BENCH_PY, VAE_CK (another VAE), SCMG_STRIDE
set -uo pipefail
R=$1; O=$2/$3; cf=$4; CODE=$(cd "$(dirname "$0")/../.." && pwd); cd $CODE; PY=${PY:-python}
C=/mnt/cpfs/zbl-cpfs-new/dataset/zehan/gesture
# GPUs = the ones this run was GIVEN (CUDA_VISIBLE_DEVICES, physical ids), so two runs can share one pod
DEVS=(${CUDA_VISIBLE_DEVICES:-$(seq -s, 0 $(($(nvidia-smi -L | wc -l) - 1)))}); DEVS=(${DEVS//,/ }); NGPU=${#DEVS[@]}; mkdir -p $O
export SCMG_J=39 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4 USE_TF=0; unset SCMG_TEXT_CAUSAL
log() { echo "$(date -Is) $*" >> $O/arch.log; }
if [ ! -f $O/DONE.json ]; then
  log "TRAIN_START $cf on $NGPU GPUs"
  $PY -m torch.distributed.run --nnodes=1 --rdzv-backend=c10d --rdzv-endpoint=localhost:${RDZV_PORT:-29611} --nproc_per_node=$NGPU -m scmg.train --stage base --view $C/echog-g1-v1 \
    --stats $C/scmg/STATS_echog_g1_v1.json --text-cache $C/scmg/text-e5-echog-g1-v1 --vae ${VAE_CK:-$R/vae/final.pt} --out $O --seed ${TRAIN_SEED:-0} \
    --cfg "$cf" ${ARCH_SC:+--sc "$ARCH_SC"} --batch $((1024 / NGPU)) --steps 8000 --lr 3e-4 --warmup 1000 --frames 256 --workers ${WORKERS:-6} --eval-every 500 --snap-every 2000 >> $O/log.txt 2>&1
fi
[ -f $O/DONE.json ] || { log "TRAIN_FAILED"; exit 1; }
log "TRAIN_DONE"
GB=$(for g in ${DEVS[@]}; do for r in $(seq ${SPG:-3}); do printf "%s " $g; done; done)   # bench.sh sets CUDA_VISIBLE_DEVICES per shard = physical ids
for p in offline engine; do for sd in 0 1; do
  [ -f $O/benchmark_${p}_seed_00$sd.json ] && continue
  PY=${BENCH_PY:-$PY} GPUS="$GB" bash scripts/echog_g1/bench.sh scmg $O/best.pt ${VAE_CK:-$R/vae/final.pt} $O/${p}_seed_00$sd $p $sd >> $O/bench.log 2>&1 || log "BENCH_FAILED $p $sd"
done; done
log "ARCH_DONE"
