#!/bin/bash
# 11-05 G1, dev side (hos-base on the held-out-speaker view): two ECHO-G borrowings re-tested on top of the selected
# setting (stream training, self-forcing OFF), each evaluated offline + engine (r4_lane G1_BENCH) and under the TTS
# protocol (infer_sweep PROTOS=engine: ref / h8tf / hinftf):
#   sf0ftx : + word-time text cross-attention after every Local Flow block (flow_text_xattn)
#   sf0llm : text features e5 (768) -> Qwen3-4B-Instruct-2507 penultimate layer (2560, as ECHO-G); fine-tune re-initialises
#            only text_in (train.py RESIZE_TEXT); the LLM cache is natively prefix-causal, so it is also the x2 cache.
# GPUs: 0 = main-view LLM cache, then sf0llm; 1 = sf0ftx; 2 = test-view LLM cache (for a later test-side run).
#   borrow_g1.sh <out root>
set -uo pipefail
O=$1; CODE=$(cd "$(dirname "$0")/../.." && pwd); cd $CODE; mkdir -p $O
B=/mnt/cpfs/zbl-cpfs-new/CKPT/zehan/gesture-scmg; C=/mnt/cpfs/zbl-cpfs-new/dataset/zehan/gesture
DEC=${DEC:-$B/echog-g1-decft-1105/decft-w3d0m05/final.pt}; INIT=$B/echog-g1-arch-1104/hos-base/best.pt
Q4=/mnt/cpfs/zbl-cpfs-new/CKPT/zehan/tts/qwen3_text/offical/Qwen3-4B-Instruct-2507
LLM=$B/models/text-qwen3-4b-echog-g1-v1; LLM_T=$B/models/text-qwen3-4b-echog-g1-v1-test
S='"p_stream": 0.67, "p_async": 1.0, "p_async_seg": 0.0, "self_force_p": 0'
A=$B/echog-g1-hos-assets; HV=$B/views/echog-g1-v1-hos
export PROFILE=g1 G1_BENCH=1 G1_BENCH_SEEDS=0 SNAP=1000000 TRAIN_SEED=0
log() { echo "$(date -Is) $*" >> $O/borrow.log; }
dev_env() { echo "G1_VIEW=$HV G1_BENCH_STEMS=$A/dev_stems.txt REF_DIR=$A/motion_39d_30fps WAV_DIR=$A/audio SPLIT=$A/dev_stems.txt VIEW_TEST=$HV VSPLIT=val"; }
cache() {   # gpu view out
  [ -f $3/PREP.json ] && { log "cache exists $3"; return 0; }
  CUDA_VISIBLE_DEVICES=$1 $PY -m scmg.prep_text_llm --view $2 --out $3 --model $Q4 --layer -2 > $O/cache-$(basename $3).log 2>&1 \
    && log "CACHE_DONE $3" || { log "CACHE_FAILED $3"; return 1; }
}
( cache 2 $C/echog-g1-v1/test $LLM_T ) &
( env $(dev_env) INIT=$INIT VAE=$DEC TEXT_TEST=$C/scmg/text-e5-echog-g1-v1 TEXT_CAUSAL_TEST=$B/models/text-e5-causal-echog-g1-v1 \
    bash scripts/r4_lane.sh 1 $O/dev sf0ftx '{"adsc": true, "flow_text_xattn": true}' "{$S}" 2000 > $O/lane-sf0ftx.log 2>&1; log "LANE_END sf0ftx" ) &
( cache 0 $C/echog-g1-v1 $LLM && env $(dev_env) G1_TEXT=$LLM INIT=$INIT VAE=$DEC TEXT_TEST=$LLM TEXT_CAUSAL_TEST=$LLM \
    bash scripts/r4_lane.sh 0 $O/dev sf0llm '{"adsc": true, "text_dim": 2560}' "{$S}" 2000 > $O/lane-sf0llm.log 2>&1; log "LANE_END sf0llm" ) &
wait
# TTS protocol on the dev set for both lanes (ref = 160 ms look-ahead + streaming text)
export PROTOS=engine SETTINGS="ref h8tf hinftf" DEV_ASSETS=$A DEV_VIEW=$HV
[ -f $O/dev/sf0ftx/final.pt ] && GPUS="1 1" bash scripts/echog_g1/infer_sweep.sh $O/dev/sf0ftx/final.pt $DEC $O/dev/sf0ftx/tts-sweep > $O/tts-sf0ftx.log 2>&1 &
[ -f $O/dev/sf0llm/final.pt ] && SWEEP_TEXT=$LLM SWEEP_TEXT_CAUSAL=$LLM GPUS="0 0" bash scripts/echog_g1/infer_sweep.sh $O/dev/sf0llm/final.pt $DEC $O/dev/sf0llm/tts-sweep > $O/tts-sf0llm.log 2>&1 &
wait; log "BORROW_DONE"
