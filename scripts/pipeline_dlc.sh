#!/bin/bash
# SCMG full training + ablations on ONE DLC pod with NGPU (default 8) A800s.
#   pipeline_dlc.sh <code dir> <out root> <vae.pt>
# env: NGPU (8), SCALE (1.0: multiply every step count; selftest uses a tiny value), NEVAL (380 dev clips),
#      BATCH (128 per GPU), EXTRA (extra train args, e.g. --mmap for selftests)
# Every train/eval is skipped when its DONE.json / RESULTS.json exists (resubmission continues where it stopped).
#
#   base (all GPUs, DDP) --+--> g0 main      peft -> causal -> state
#                          +--> g1 nomtc     peft(mtc=false) -> causal          [Multi-Token Conditioning]
#                          +--> g2 nostyle   peft(style=false) -> causal        [Style Distribution Head]
#                          +--> g3 seed1     peft -> causal -> state (seed 1)   [seed variance]
#                          +--> g4 eval base, then nocat-causal (p_stream=0)    [Chunk-Aware Causal Training]
#                          +--> g5 noop-causal (no out-painting / continuity)   [Masked Out-painting]
#                          +--> g6 fullft-causal (--unfreeze-flow)              [PEFT vs limited unfreezing]
#                          +--> g7 measured-state (history = measured), then unfreeze-state   [Robot State Adapter]
set -uo pipefail
CODE=${1:?code dir}; R=${2:?out root}; VAE=${3:?vae}
NGPU=${NGPU:-8}; SCALE=${SCALE:-1.0}; NEVAL=${NEVAL:-120}; NFULL=${NFULL:-380}; NMID=${NMID:-40}; BATCH=${BATCH:-128}; EXTRA=${EXTRA:-}
PY=${PY:-python}
D=/mnt/cpfs/zbl-cpfs-new/dataset/zehan/gesture/scmg
TOKC=${TOKC:-}; BASECFG=${BASECFG:-{\}}          # VoxCPM-style planner: TOKC=<token cache>, BASECFG={"planner":"voxcpm"}
C="--stats $D/STATS_v36_1.json --text-cache $D/text-e5-v36_1 --vae $VAE --workers ${WORKERS:-6} $EXTRA ${TOKC:+--tok-cache $TOKC}"
cd $CODE && export PYTHONPATH=$CODE
mkdir -p $R
rm -f $R/*/FAILED                          # a resubmission retries lanes that failed in an earlier job
steps() { python3 -c "print(max(4, int($1 * $SCALE)))"; }
g() { echo $(( $1 % NGPU )); }            # lane -> GPU (selftest: everything on GPU 0)

train() {  # gpu out args...
  local gpu=$1 o=$2; shift 2
  [ -f $R/$o/DONE.json ] && { echo "skip $o"; return 0; }
  [ -f $R/$o/FAILED ] && return 1
  mkdir -p $R/$o; echo "$(date -Is) start $o on GPU $gpu"
  if CUDA_VISIBLE_DEVICES=$gpu $PY -m scmg.train --out $R/$o $C --batch $BATCH "$@" >> $R/$o/log.txt 2>&1; then
    echo "$(date -Is) done $o"
  else
    echo "$(date -Is) FAILED $o"; touch $R/$o/FAILED; return 1
  fi
}
evaluate() {  # gpu dir n ckpt-name extra...
  local gpu=$1 o=$2 n=$3 ck=$4; shift 4
  local out=$R/$o/eval-${ck%.pt}
  [ -f $out/RESULTS.json ] && return 0
  CUDA_VISIBLE_DEVICES=$gpu $PY -m scmg.evaluate --ckpt $R/$o/$ck --vae $VAE --stats $D/STATS_v36_1.json \
    --text-cache $D/text-e5-v36_1 ${TOKC:+--tok-cache $TOKC} --out $out --n $n "$@" $( [[ "$EXTRA" == *--mmap* ]] && echo --mmap ) >> $R/$o/eval.log 2>&1 \
    || echo "$(date -Is) eval FAILED $o/$ck"
}
wait_for() {  # dir : returns 1 if upstream failed
  until [ -f $R/$1/DONE.json ]; do [ -f $R/$1/FAILED ] && return 1; sleep 30; done
}
mid_eval() {  # gpu dir until-file : evaluate every new snapshot (mid-training evaluation, ~every 5000 steps)
  local gpu=$1 o=$2
  while true; do
    for s in $R/$o/snap_*.pt; do [ -f "$s" ] && evaluate $gpu $o $NMID $(basename $s) --horizons inf,0; done
    [ -f $R/$o/DONE.json ] || [ -f $R/$o/FAILED ] && break
    sleep 120
  done
}

# ---------------- runtime check (CUDA really computes on every GPU)
$PY - <<'EOF' || exit 2
import torch, sys
print('torch', torch.__version__, 'cuda', torch.version.cuda, 'devices', torch.cuda.device_count(), flush=True)
for i in range(torch.cuda.device_count()):
    x = torch.randn(1024, 1024, device=f'cuda:{i}'); y = (x @ x).sum().item()
    assert y == y, 'nan'
print('CUDA_COMPUTE_OK', flush=True)
EOF

# ---------------- stage 0: motion VAE (only when the given VAE does not exist yet, e.g. the 25 Hz ablation)
if [ ! -f $VAE ]; then
  vd=$(basename $(dirname $VAE))
  train 0 $vd --stage vae --steps $(steps 30000) --batch 128 --lr 1e-3 --warmup $(steps 500) \
    --deriv-ramp $(steps 3000) --snap-every 0 || exit 1
  CUDA_VISIBLE_DEVICES=0 $PY scripts/check_vae.py $VAE > $R/$vd/CHECK_VAE.json 2>> $R/$vd/log.txt || exit 1
  cat $R/$vd/CHECK_VAE.json
fi

# ---------------- base: all GPUs, DDP
if [ ! -f $R/base/DONE.json ]; then
  mkdir -p $R/base; echo "$(date -Is) start base on $NGPU GPUs"
  ( mid_eval $(g 7) base ) & BASEWATCH=$!          # keeps evaluating snapshots in the background; never blocks lanes
  if [ $NGPU -gt 1 ]; then
    $PY -m torch.distributed.run --standalone --nproc_per_node=$NGPU -m scmg.train --stage base --out $R/base $C --cfg "$BASECFG" \
      --batch $BATCH --steps $(steps 40000) --lr 3e-4 --warmup $(steps 2000) --snap-every $(steps ${SNAP:-5000}) >> $R/base/log.txt 2>&1
  else
    CUDA_VISIBLE_DEVICES=0 $PY -m scmg.train --stage base --out $R/base $C --cfg "$BASECFG" \
      --batch $BATCH --steps $(steps 40000) --lr 3e-4 --warmup $(steps 2000) --snap-every $(steps ${SNAP:-5000}) >> $R/base/log.txt 2>&1
  fi
  [ -f $R/base/DONE.json ] || { echo "base FAILED"; touch $R/base/FAILED; kill $BASEWATCH 2>/dev/null; exit 1; }
fi

PE="--steps $(steps 20000) --lr 1e-4 --warmup $(steps 500) --snap-every $(steps ${SNAP:-5000})"
CA="--steps $(steps 30000) --lr 1e-4 --warmup $(steps 500) --snap-every $(steps ${SNAP:-5000})"
ST="--steps $(steps 20000) --lr 1e-4 --warmup $(steps 500) --snap-every $(steps ${SNAP:-5000})"
B=$R/base/final.pt
LANES=()
( train $(g 0) main-peft --stage peft --init $B $PE && evaluate $(g 0) main-peft $NEVAL final.pt --horizons inf,4,0 \
  && train $(g 0) main-causal --stage causal --init $R/main-peft/final.pt $CA && evaluate $(g 0) main-causal $NFULL final.pt --horizons inf,8,4,2,0 \
  && train $(g 0) main-state --stage state --init $R/main-causal/final.pt $ST && evaluate $(g 0) main-state $NFULL final.pt --horizons 4,0 --closed-loop ) & LANES+=($!)
( train $(g 1) nomtc-peft --stage peft --init $B --cfg '{"mtc": false}' $PE \
  && train $(g 1) nomtc-causal --stage causal --init $R/nomtc-peft/final.pt --cfg '{"mtc": false}' $CA && evaluate $(g 1) nomtc-causal $NEVAL final.pt --horizons inf,4,0 ) & LANES+=($!)
( train $(g 2) nostyle-peft --stage peft --init $B --cfg '{"style": false}' $PE \
  && train $(g 2) nostyle-causal --stage causal --init $R/nostyle-peft/final.pt --cfg '{"style": false}' $CA && evaluate $(g 2) nostyle-causal $NEVAL final.pt --horizons inf,4,0 ) & LANES+=($!)
( train $(g 3) seed1-peft --stage peft --init $B --seed 1 $PE \
  && train $(g 3) seed1-causal --stage causal --init $R/seed1-peft/final.pt --seed 1 $CA && evaluate $(g 3) seed1-causal $NEVAL final.pt --horizons inf,4,0 \
  && train $(g 3) seed1-state --stage state --init $R/seed1-causal/final.pt --seed 1 $ST && evaluate $(g 3) seed1-state $NEVAL final.pt --horizons 4 --closed-loop ) & LANES+=($!)
( evaluate $(g 4) base $NEVAL final.pt --horizons inf,8,4,0; wait_for main-peft \
  && train $(g 4) nocat-causal --stage causal --init $R/main-peft/final.pt --sc '{"p_stream": 0.0}' $CA && evaluate $(g 4) nocat-causal $NEVAL final.pt --horizons inf,8,4,2,0 ) & LANES+=($!)
( wait_for main-peft \
  && train $(g 5) noop-causal --stage causal --init $R/main-peft/final.pt --sc '{"outpaint": false, "w_cont": 0.0}' $CA && evaluate $(g 5) noop-causal $NEVAL final.pt --horizons inf,4,0 ) & LANES+=($!)
( wait_for main-peft \
  && train $(g 6) fullft-causal --stage causal --init $R/main-peft/final.pt --unfreeze-flow $CA && evaluate $(g 6) fullft-causal $NEVAL final.pt --horizons inf,4,0 ) & LANES+=($!)
( wait_for main-causal \
  && train $(g 7) measured-state --stage state --init $R/main-causal/final.pt --sc '{"state_hist_measured": true}' $ST && evaluate $(g 7) measured-state $NEVAL final.pt --horizons 4 --closed-loop
  wait_for main-causal && train $(g 7) unfreeze-state --stage state --init $R/main-causal/final.pt --unfreeze-flow $ST && evaluate $(g 7) unfreeze-state $NEVAL final.pt --horizons 4,0 --closed-loop ) & LANES+=($!)
WATCH=()
for o in main-peft main-causal main-state; do ( mid_eval $(g 0) $o ) & WATCH+=($!); done
wait "${LANES[@]}"
kill "${WATCH[@]}" ${BASEWATCH:-} 2>/dev/null
$PY -m scmg.summarize $R > $R/SUMMARY.md 2>&1 || true
echo "$(date -Is) PIPELINE_DONE"; touch $R/PIPELINE_DONE
