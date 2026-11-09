#!/bin/bash
# 11-04: G1 motion-VAE variant -> round-trip diagnosis on val_common with the official evaluator (the VAE alone already
# costs Jerk ratio 0.75 and doubles contact sliding).  vae_variant.sh <gpu> <out root> <label> <vae-cfg-json> [steps=30000]
#   env PY, SCMG_STRIDE (2 = 25 Hz latents), bench.sh paths (EG, OFF, ...; defaults = this 5090 host)
set -uo pipefail
gpu=$1; O=$2/$3; cf=$4; ST=${5:-30000}; CODE=$(cd "$(dirname "$0")/../.." && pwd); cd $CODE; PY=${PY:-python}
C=/mnt/cpfs/zbl-cpfs-new/dataset/zehan/gesture; mkdir -p $O
export SCMG_J=39 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=$gpu; unset SCMG_TEXT_CAUSAL
log() { echo "$(date -Is) $*" >> $O/variant.log; }
if [ ! -f $O/vae/DONE.json ]; then
  log "VAE_START $cf"
  $PY -m scmg.train --stage vae --view $C/echog-g1-v1 --stats $C/scmg/STATS_echog_g1_v1.json --out $O/vae --steps $ST --batch 128 \
    --lr 1e-3 --warmup 500 --frames 256 --workers ${WORKERS:-6} --deriv-ramp 3000 --snap-every 0 --eval-every 2000 --vae-cfg "$cf" >> $O/vae.log 2>&1
fi
[ -f $O/vae/DONE.json ] || { log "VAE_FAILED"; exit 1; }
$PY scripts/echog_g1/vae_roundtrip.py --vae $O/vae/final.pt --stats $C/scmg/STATS_echog_g1_v1.json --view $C/echog-g1-v1/test \
  --text-cache $C/scmg/text-e5-echog-g1-v1-test --out $O/rt --mode vae >> $O/rt.log 2>&1 || { log "RT_FAILED"; exit 1; }
bash scripts/echog_g1/bench.sh eval $O/rt ${SPLIT:-/mnt/bulk/zehan/echog-dataset/official-meta/splits/val_common.txt} vaert_$3 $O/benchmark_rt.json >> $O/rt.log 2>&1 || log "EVAL_FAILED"
log "VARIANT_DONE"
