#!/bin/bash
# 11-08 EX002 decoder-only fine-tune r3d (criteria docs/criteria/EX002_CTL.md "Decoder-only fine-tune r3d", pre-registered 600946f):
# arms A / B, quick lane 2000 steps -> gates 1-2 -> 10k (only arms that pass) -> gates 1-2 on the 10k checkpoint -> regenerate enc-s0 none
# (offline + psync, seeds 0-2) with the candidate decoder -> gate 3 (window p95, FGD v2 / FKD vs r3 = A's enc-s0 runs, liveliness). 4 GPUs.
#   ex002_r3d.sh <out dir>      env PY
set -uo pipefail
O=$1; CODE=$(cd "$(dirname "$0")/../.." && pwd); cd $CODE; mkdir -p $O
B=/mnt/cpfs/zbl-cpfs-new/CKPT/zehan/gesture-scmg; C=/mnt/cpfs/zbl-cpfs-new/dataset/zehan/gesture; X=$B/control-token-1105/ex002; T=$X/tokens
PY=${PY:-python}; export SCMG_J=34 SCMG_NO_GLITCH=1 OMP_NUM_THREADS=${OMP_NUM_THREADS:-1} SCMG_TEXT_CAUSAL=$B/models/text-e5-causal-v36_1
R3=$B/trained/vae-r3/final.pt; E=$B/ex002-sfscope-1108/enc-s0; CK=$E/train/final.pt
FT=scripts/ex002/vae_decoder_ft_ex002.py; G12=scripts/ex002/diag/r3d_gates.py; AC=scripts/ex002/diag/accel_stats.py; LA=scripts/ex002/lively_floor_ex002.py
LV="scripts/video/static_share.py score --norm $T/NORM_DEV.json"
log() { echo "$(date -Is) $*" >> $O/lane.log; }
log "START commit $(cat GIT_COMMIT 2>/dev/null || git rev-parse HEAD 2>/dev/null)"
# 1. quick lane
for arm in A B; do g=$([ $arm = A ] && echo 0 || echo 1)
  ( [ -f $O/$arm/q/DONE.json ] || CUDA_VISIBLE_DEVICES=$g OMP_NUM_THREADS=4 $PY $FT --vae $R3 --out $O/$arm/q --arm $arm --steps 2000 --workers 8 > $O/$arm.q.log 2>&1
    CUDA_VISIBLE_DEVICES=$g $PY $G12 --cand $O/$arm/q/final.pt --out $O/$arm/G12_quick.json >> $O/$arm.q.log 2>&1 ) &
done; wait $(jobs -p)
for arm in A B; do log "QUICK $arm $(grep R3D_G12 $O/$arm.q.log | cut -c1-400)"; done
# 2. 10k for the arms whose quick lane passes gates 1-2
for arm in A B; do g=$([ $arm = A ] && echo 0 || echo 1)
  if $PY -c "import json,sys; sys.exit(0 if json.load(open('$O/$arm/G12_quick.json'))['gate12'] else 1)"; then
    ( [ -f $O/$arm/f/DONE.json ] || CUDA_VISIBLE_DEVICES=$g OMP_NUM_THREADS=4 $PY $FT --vae $R3 --init $O/$arm/q/final.pt --out $O/$arm/f --arm $arm --steps 8000 --workers 8 > $O/$arm.f.log 2>&1
      CUDA_VISIBLE_DEVICES=$g $PY $G12 --cand $O/$arm/f/final.pt --out $O/$arm/G12_final.json >> $O/$arm.f.log 2>&1 ) &
  else log "ARM_STOPPED $arm (quick lane fails gates 1-2)"; fi
done; wait $(jobs -p)
ARMS=""; for arm in A B; do [ -f $O/$arm/f/final.pt ] && ARMS="$ARMS $arm" && log "FINAL $arm $(grep R3D_G12 $O/$arm.f.log | cut -c1-400)"; done
[ -z "$ARMS" ] && { log "NO_ARM_REACHED_10K"; log "LANE_DONE"; exit 0; }
# 3. regenerate enc-s0 none with each 10k decoder (offline + psync, seeds 0-2)
i=0; for arm in $ARMS; do for pr in offline psync; do for sd in 0 1 2; do for k in 0 1 2 3; do echo "$((i % 16))|$arm|$pr|$sd|$k/4"; i=$((i + 1)); done; done; done; done > $O/jobs.txt
gen_shard() {  # gpu arm protocol seed shard
  local od=$O/$2/$3/s$4; mkdir -p $od
  CUDA_VISIBLE_DEVICES=$1 $PY scripts/ex002/bench_ex002.py infer --ckpt $CK --vae $O/$2/f/final.pt --protocol $3 --seed $4 --out $od --shard $5 >> $O/infer_$1.log 2>&1 || log "INFER_FAILED $od $5"
}
for w in $(seq 0 15); do ( grep "^$w|" $O/jobs.txt | while IFS='|' read -r ww arm pr sd sh; do gen_shard $((ww % 4)) $arm $pr $sd $sh; done ) & done; wait $(jobs -p)
log "GENERATED"
# 4. gate 3 evaluations
for pr in offline psync; do
  CUDA_VISIBLE_DEVICES=0 $PY scripts/ex002/bench_ex002.py eval --out $O/EVAL_$pr.json $(for s in 0 1 2; do echo r3_s$s=$E/$pr/s$s; for arm in $ARMS; do echo ${arm}_s$s=$O/$arm/$pr/s$s; done; done) > $O/eval_$pr.log 2>&1 &
  $PY $AC --out $O/ACCEL_$pr.json GT=$E/offline r3=$E/$pr/s0,$E/$pr/s1,$E/$pr/s2 $(for arm in $ARMS; do echo $arm=$O/$arm/$pr/s0,$O/$arm/$pr/s1,$O/$arm/$pr/s2; done) >> $O/eval_$pr.log 2>&1 &
  $PY $LA --out $O/LIVE_protocol_$pr.json GT=$E/offline r3=$E/$pr/s0,$E/$pr/s1,$E/$pr/s2 $(for arm in $ARMS; do echo $arm=$O/$arm/$pr/s0,$O/$arm/$pr/s1,$O/$arm/$pr/s2; done) >> $O/eval_$pr.log 2>&1 &
  $PY $LV --out $O/LIVE_mine_$pr.json $(for s in 0 1 2; do echo r3_s$s=$E/$pr/s$s; for arm in $ARMS; do echo ${arm}_s$s=$O/$arm/$pr/s$s; done; done) >> $O/eval_$pr.log 2>&1 &
done; wait $(jobs -p)
log "LANE_DONE"; exit 0
