#!/bin/bash
# Full SCMG training + ablations on zehan-server (4 x A800, shared: only the free memory is used).
#   bash pipeline_zehan.sh <vae.pt>        (run inside tmux; every step is skipped if its DONE.json exists)
# Lanes after the base model (one GPU each):
#   GPU0 main      peft -> causal -> state
#   GPU1 no-MTC    peft(mtc=false: style + LoRA only) -> causal          [Multi-Token Conditioning ablation]
#   GPU2 no-CAT    causal from main peft with p_stream=0                  [Chunk-Aware Causal Training ablation]
#   GPU3 no-OP     causal from main peft without out-painting masks / continuity loss   [Out-painting ablation]
#   then GPU2 state-measured (history = measured state, no adapter), GPU3 state + unfreeze-flow
set -uo pipefail
VAE=${1:?vae checkpoint}
PY=/workspace/zehan/env/tts_drift/bin/python
R=/mnt/cpfs/zbl-cpfs-new/CKPT/zehan/gesture-scmg/trained
D=/mnt/cpfs/zbl-cpfs-new/dataset/zehan/gesture/scmg
C="--stats $D/STATS_v36_1.json --text-cache $D/text-e5-v36_1 --vae $VAE --workers 6"
# run from a frozen snapshot of the code: later syncs must not touch running jobs (CPFS stale file handles)
SNAP=${CODE:?set CODE=<release dir>}
cd $SNAP || exit 1
export PYTHONPATH=$SNAP
echo "code snapshot $SNAP"

train() {  # gpu out args...
  local g=$1 o=$2; shift 2
  [ -f $R/$o/DONE.json ] && { echo "skip $o"; return 0; }
  mkdir -p $R/$o
  CUDA_VISIBLE_DEVICES=$g $PY -m scmg.train --out $R/$o $C "$@" >> $R/$o/log.txt 2>&1
}
evaluate() {  # gpu ckpt-dir n extra...
  local g=$1 o=$2 n=$3; shift 3
  [ -f $R/$o/eval/RESULTS.json ] && return 0
  CUDA_VISIBLE_DEVICES=$g $PY -m scmg.evaluate --ckpt $R/$o/final.pt --vae $VAE --stats $D/STATS_v36_1.json \
    --text-cache $D/text-e5-v36_1 --out $R/$o/eval --n $n "$@" >> $R/$o/eval.log 2>&1
}

# ---------------- base model: 4-GPU data parallel, global batch 256
if [ ! -f $R/base/DONE.json ]; then
  mkdir -p $R/base
  CUDA_VISIBLE_DEVICES=0,1,2,3 $PY -m torch.distributed.run --standalone --nproc_per_node=4 -m scmg.train \
    --stage base --out $R/base $C --steps 60000 --batch 64 --lr 2e-4 --warmup 2000 >> $R/base/log.txt 2>&1 || exit 1
fi
evaluate 0 base 120 --horizons inf,4,0 &
# ---------------- PEFT stages
P="--steps 20000 --batch 64 --lr 1e-4 --warmup 500"
( train 0 main-peft --stage peft --init $R/base/final.pt $P && evaluate 0 main-peft 120 --horizons inf,4,0 ) &
( train 1 nomtc-peft --stage peft --init $R/base/final.pt --cfg '{"mtc": false}' $P ) &
wait
P="--steps 30000 --batch 64 --lr 1e-4 --warmup 500"
( train 0 main-causal --stage causal --init $R/main-peft/final.pt $P && evaluate 0 main-causal 120 --horizons inf,8,4,0 ) &
( train 1 nomtc-causal --stage causal --init $R/nomtc-peft/final.pt --cfg '{"mtc": false}' $P && evaluate 1 nomtc-causal 120 --horizons inf,4,0 ) &
( train 2 nocat-causal --stage causal --init $R/main-peft/final.pt --sc '{"p_stream": 0.0}' $P && evaluate 2 nocat-causal 120 --horizons inf,8,4,0 ) &
( train 3 noop-causal --stage causal --init $R/main-peft/final.pt --sc '{"outpaint": false, "w_cont": 0.0}' $P && evaluate 3 noop-causal 120 --horizons inf,4,0 ) &
wait
P="--steps 20000 --batch 64 --lr 1e-4 --warmup 500"
( train 0 main-state --stage state --init $R/main-causal/final.pt $P && evaluate 0 main-state 120 --horizons 4,0 --closed-loop ) &
( train 2 measured-state --stage state --init $R/main-causal/final.pt --sc '{"state_hist_measured": true}' $P && evaluate 2 measured-state 120 --horizons 4 --closed-loop ) &
( train 3 unfreeze-state --stage state --init $R/main-causal/final.pt --unfreeze-flow $P && evaluate 3 unfreeze-state 120 --horizons 4,0 --closed-loop ) &
wait
echo PIPELINE_DONE > $R/PIPELINE_DONE
