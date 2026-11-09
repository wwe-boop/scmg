#!/bin/bash
# 11-05 G1 text-input study (user: "文本模型输入多做一些 做深"), dev side (hos-base on the held-out-speaker view), all on
# the selected setting (stream training, self-forcing OFF), 6000 steps each (the 2000-step LLM lane re-initialised text_in
# and was likely under-trained):
#   e5      : multilingual-e5 (768), the current text encoder                         = control
#   q4bL-2  : Qwen3-4B-Instruct-2507 penultimate layer (2560)                         (as ECHO-G)
#   q4bL18  : Qwen3-4B-Instruct-2507 middle layer 18 of 36 (2560)                     (middle layers carry more semantics)
#   q4bftx  : q4bL-2 + word-time text cross-attention after every Local Flow block    (= the ECHO-G combination)
# Every lane: official benchmark offline + engine (r4_lane G1_BENCH), then TTS protocol (h4tf = 320 ms look-ahead +
# sentence-final text) and separate text guidance under TTS (ct1tts / ct4tts; cfg_t = cfg_w = 2.5 is the joint CFG).
#   text_study_g1.sh <out root>     GPUs 0..3 (one lane each); LLM caches are built first on GPUs 2 / 3 when missing
set -uo pipefail
O=$1; CODE=$(cd "$(dirname "$0")/../.." && pwd); cd $CODE; mkdir -p $O
B=/mnt/cpfs/zbl-cpfs-new/CKPT/zehan/gesture-scmg; C=/mnt/cpfs/zbl-cpfs-new/dataset/zehan/gesture
DEC=${DEC:-$B/echog-g1-decft-1105/decft-w3d0m05/final.pt}; INIT=$B/echog-g1-arch-1104/hos-base/best.pt
Q4=/mnt/cpfs/zbl-cpfs-new/CKPT/zehan/tts/qwen3_text/offical/Qwen3-4B-Instruct-2507
E5=$C/scmg/text-e5-echog-g1-v1; E5C=$B/models/text-e5-causal-echog-g1-v1
L2=$B/models/text-qwen3-4b-echog-g1-v1; L18=$B/models/text-qwen3-4b-L18-echog-g1-v1
S='"p_stream": 0.67, "p_async": 1.0, "p_async_seg": 0.0, "self_force_p": 0'
A=$B/echog-g1-hos-assets; HV=$B/views/echog-g1-v1-hos; STEPS=${STEPS:-6000}
export PROFILE=g1 G1_BENCH=1 G1_BENCH_SEEDS=0 SNAP=1000000 TRAIN_SEED=0
log() { echo "$(date -Is) $*" >> $O/study.log; }
cache() {   # gpu layer out
  [ -f $3/PREP.json ] && { log "cache exists $3"; return 0; }
  CUDA_VISIBLE_DEVICES=$1 $PY -m scmg.prep_text_llm --view $C/echog-g1-v1 --out $3 --model $Q4 --layer $2 > $O/cache-$(basename $3).log 2>&1 \
    && log "CACHE_DONE $3" || { log "CACHE_FAILED $3"; return 1; }
}
lane() {    # gpu label text causal cfg
  local g=$1 lab=$2 tx=$3 tc=$4 cf=$5
  env G1_VIEW=$HV G1_BENCH_STEMS=$A/dev_stems.txt REF_DIR=$A/motion_39d_30fps WAV_DIR=$A/audio SPLIT=$A/dev_stems.txt VIEW_TEST=$HV VSPLIT=val \
      G1_TEXT=$tx TEXT_TEST=$tx TEXT_CAUSAL_TEST=$tc INIT=$INIT VAE=$DEC \
      bash scripts/r4_lane.sh $g $O/dev $lab "$cf" "{$S}" $STEPS > $O/lane-$lab.log 2>&1
  log "LANE_END $lab"
  [ -f $O/dev/$lab/final.pt ] || return 1
  PROTOS=engine SETTINGS="h4tf ct1tts ct4tts" DEV_ASSETS=$A DEV_VIEW=$HV SWEEP_TEXT=$tx SWEEP_TEXT_CAUSAL=$tc GPUS="$g $g" \
      bash scripts/echog_g1/infer_sweep.sh $O/dev/$lab/final.pt $DEC $O/dev/$lab/tts-sweep > $O/tts-$lab.log 2>&1
  log "TTS_END $lab"
}
( lane 0 e5 $E5 $E5C '{"adsc": true}' ) &
( lane 1 q4bL-2 $L2 $L2 '{"adsc": true, "text_dim": 2560}' ) &
( cache 2 18 $L18 && lane 2 q4bL18 $L18 $L18 '{"adsc": true, "text_dim": 2560}' ) &
( lane 3 q4bftx $L2 $L2 '{"adsc": true, "text_dim": 2560, "flow_text_xattn": true}' ) &
wait; log "STUDY_DONE"
