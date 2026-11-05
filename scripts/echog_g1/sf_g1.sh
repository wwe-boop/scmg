#!/bin/bash
# 11-05 G1: inference-matched self-forcing. Since the self-forcing fix (dcd50fb) the self-forced batches (default
# self_force_p 0.5, history generated with 4 Euler steps and NO CFG) really train the planner / Local Flow, while inference
# builds the history with 10 steps and CFG 2.5. Post-fix models show a 320 ms chunk-periodic speed oscillation (fast chunk
# start, slow tail; beats pile up at chunk phase 0.6-0.9: max/min phase bin 1.38 -> 1.60 offline, 1.45 -> 2.01 engine)
# and streaming speed +56 %. Lanes (warm start, stream training as the improve lanes, 2000 steps, official benchmark
# offline + engine, inference seed 0), each = one StageCfg fragment:
#   ctrl  : current defaults                         sf0  : self_force_p 0 (what pre-fix training effectively did)
#   sfs10 : self-forced history with 10 steps         sfm  : 10 steps + CFG 2.5 (= the inference sampler)
# dev side = hos-base on the held-out-speaker view (selection); test side = fix-base on val_common (run only after selection).
#   sf_g1.sh <out root>     env SETTINGS ("label:fragment ..."; fragment = comma-joined JSON members, no spaces), SIDES, GPUS
set -uo pipefail
O=$1; CODE=$(cd "$(dirname "$0")/../.." && pwd); cd $CODE; mkdir -p $O
B=/mnt/cpfs/zbl-cpfs-new/CKPT/zehan/gesture-scmg; C=/mnt/cpfs/zbl-cpfs-new/dataset/zehan/gesture
DEC=${DEC:-$B/echog-g1-decft-1105/decft-w3d0m05/final.pt}
S='"p_stream": 0.67, "p_async": 1.0, "p_async_seg": 0.0'
GL=(${GPUS:-0 1 2 3}); i=0
export PROFILE=g1 G1_BENCH=1 G1_BENCH_SEEDS=0 SNAP=1000000 TRAIN_SEED=0
for spec in ${SETTINGS:-ctrl: sf0:\"self_force_p\":0 sfs10:\"self_force_steps\":10 sfm:\"self_force_steps\":10,\"self_force_cfg\":2.5}; do
  lab=${spec%%:*}; frag=${spec#*:}
  sc="{$S}"; [ -n "$frag" ] && sc="{$S, ${frag//,/, }}"
  python3 -c "import json,sys; json.loads(sys.argv[1])" "$sc" || { echo "bad StageCfg JSON for $lab: $sc"; exit 2; }
  for side in ${SIDES:-dev}; do
    g=${GL[$((i % ${#GL[@]}))]}; i=$((i + 1))
    echo "$(date -Is) lane $side/$lab gpu $g sc $sc" >> $O/sf.log
    if [ $side = test ]; then
      ( INIT=$B/echog-g1-arch-1104/fix-base/best.pt VAE=$DEC bash scripts/r4_lane.sh $g $O/test $lab '{"adsc": true}' "$sc" 2000 > $O/lane-test-$lab.log 2>&1 ) &
    else
      A=$B/echog-g1-hos-assets
      ( INIT=$B/echog-g1-arch-1104/hos-base/best.pt VAE=$DEC G1_VIEW=$B/views/echog-g1-v1-hos G1_BENCH_STEMS=$A/dev_stems.txt \
        REF_DIR=$A/motion_39d_30fps WAV_DIR=$A/audio SPLIT=$A/dev_stems.txt VIEW_TEST=$B/views/echog-g1-v1-hos VSPLIT=val \
        TEXT_TEST=$C/scmg/text-e5-echog-g1-v1 TEXT_CAUSAL_TEST=$B/models/text-e5-causal-echog-g1-v1 \
        bash scripts/r4_lane.sh $g $O/dev $lab '{"adsc": true}' "$sc" 2000 > $O/lane-dev-$lab.log 2>&1 ) &
    fi
  done
done
wait; echo "$(date -Is) SF_DONE ${SIDES:-dev}" >> $O/sf.log
