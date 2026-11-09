#!/bin/bash
# 11-06 floor-state training lanes for the G1 interruption benchmark (DLC, 1 pod x 2 GPUs).
# Warm start from the sf0 lane (echog-g1-sf-1105 {dev,test}/sf0/final.pt), the sf0 StageCfg, 2000 steps (= sf_g1.sh), lanes:
#   ctrl : original view, no tails, cfg {"adsc": true}                         (the effect of 2000 more steps alone)
#   yld  : tail view "yield" (only retract / hands-low tails, floor 2), cfg + {"floor": true}, tail share 0.3
#   lab  : tail view "labelled" (all tails: floor 2 retract / low, floor 1 hold), cfg + {"floor": true}, tail share 0.3
# Tails come from training speakers only; the dev held-out speakers (2_scott, 6_carla) and the val_common speakers never
# get tails (build_tail_view.py). Dev side: hos views, probes + the official benchmark on the 400 dev stems (G1_BENCH=1).
# Test side: trained only (G1_BENCH=0: val_common is touched later, once, for the dev-selected lane).
#   floor_g1.sh <out root>        env LANES (default "ctrl yld lab"), SIDES (default "dev test")
set -uo pipefail
O=$1; CODE=$(cd "$(dirname "$0")/../.." && pwd); cd $CODE; mkdir -p $O
B=/mnt/cpfs/zbl-cpfs-new/CKPT/zehan/gesture-scmg; C=/mnt/cpfs/zbl-cpfs-new/dataset/zehan/gesture; V=$B/interrupt-baseline-1105/views
DEC=$B/echog-g1-decft-1105/decft-w3d0m05/final.pt
SC='{"p_stream": 0.67, "p_async": 1.0, "p_async_seg": 0.0, "self_force_p": 0}'
export PROFILE=g1 SNAP=1000000 TRAIN_SEED=0 G1_BENCH_SEEDS=0 TAIL_SHARE=${TAIL_SHARE:-0.3}
lane() {  # <side> <gpu> <lane>
  local side=$1 g=$2 lab=$3 view= fl= cf='{"adsc": true}'
  case $lab in yld) view=echog-g1-v1-tail-yld;; lab) view=echog-g1-v1-tail-lab;; esac
  [ -n "$view" ] && cf='{"adsc": true, "floor": true}'
  echo "$(date -Is) lane $side/$lab gpu $g view ${view:-orig} cf $cf" >> $O/floor.log
  if [ $side = test ]; then
    ( [ -n "$view" ] && export G1_VIEW=$V/$view G1_FLOOR=$V/$view
      INIT=$B/echog-g1-sf-1105/test/sf0/final.pt VAE=$DEC G1_BENCH=0 bash scripts/r4_lane.sh $g $O/test $lab "$cf" "$SC" 2000 ) > $O/lane-test-$lab.log 2>&1
  else
    A=$B/echog-g1-hos-assets
    ( export G1_VIEW=$B/views/echog-g1-v1-hos; [ -n "$view" ] && export G1_VIEW=$V/$view-hos G1_FLOOR=$V/$view-hos
      INIT=$B/echog-g1-sf-1105/dev/sf0/final.pt VAE=$DEC G1_BENCH=1 G1_BENCH_STEMS=$A/dev_stems.txt \
      REF_DIR=$A/motion_39d_30fps WAV_DIR=$A/audio SPLIT=$A/dev_stems.txt VIEW_TEST=$B/views/echog-g1-v1-hos VSPLIT=val \
      TEXT_TEST=$C/scmg/text-e5-echog-g1-v1 TEXT_CAUSAL_TEST=$B/models/text-e5-causal-echog-g1-v1 \
      bash scripts/r4_lane.sh $g $O/dev $lab "$cf" "$SC" 2000 ) > $O/lane-dev-$lab.log 2>&1
  fi
  echo "$(date -Is) done $side/$lab" >> $O/floor.log
}
i=0
for side in ${SIDES:-dev test}; do
  g=$i; i=$((i + 1))
  ( for lab in ${LANES:-ctrl yld lab}; do lane $side $g $lab; done ) &
done
wait; echo "$(date -Is) FLOOR_DONE" >> $O/floor.log
