#!/bin/bash
# 11-05 G1 stage 2: decoder-aware generator fine-tuning. The contact-aware decoder (frozen encoder -> same latent space)
# replaces the VAE decoder INSIDE the training loss; the generator continues 2000 steps (stream training, no BAgger) with
# the G1 foot-contact term on the decoded [ctx | target] frames, so it learns latents that decode to planted feet.
# Every setting runs twice: from fix-base (val_common = test, report) and from hos-base on the held-out-speaker view
# (400-clip unseen-speaker dev = selection). One lane per GPU; official benchmark offline + engine, inference seed 0.
#   stage2_g1.sh <out root>        env SETTINGS ("label=w_foot ..."), GPUS
set -uo pipefail
O=$1; CODE=$(cd "$(dirname "$0")/../.." && pwd); cd $CODE; mkdir -p $O
B=/mnt/cpfs/zbl-cpfs-new/CKPT/zehan/gesture-scmg; C=/mnt/cpfs/zbl-cpfs-new/dataset/zehan/gesture
DEC=${DEC:-$B/echog-g1-decft-1105/decft-w3d0m05/final.pt}
S='"p_stream": 0.67, "p_async": 1.0, "p_async_seg": 0.0'
GL=(${GPUS:-0 1 2 3 4 5}); i=0
export PROFILE=g1 G1_BENCH=1 G1_BENCH_SEEDS=0 SNAP=1000000 TRAIN_SEED=0
for spec in ${SETTINGS:-ctrl=0 foot03=0.3 foot1=1.0}; do
  lab=${spec%%=*}; wf=${spec#*=}
  sc="{$S}"; [ "$wf" != 0 ] && sc="{$S, \"w_foot\": $wf}"
  for side in test dev; do
    g=${GL[$((i % ${#GL[@]}))]}; i=$((i + 1))
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
wait; echo "$(date -Is) STAGE2_DONE" >> $O/stage2.log
