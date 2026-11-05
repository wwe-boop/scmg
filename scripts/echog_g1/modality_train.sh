#!/bin/bash
# 11-05 TRAINING-level input-modality ablation: the selected setting (sf0 = stream warm start, self_force_p 0, contact-aware
# decoder, 2000 steps, adsc) trained with ONE modality only, same starts as sf0 (dev: hos-base on the held-out-speaker
# view; test: fix-base on the full view); control = the existing sf0 lanes (echog-g1-sf-1105/{dev,test}/sf0).
#   aonly : StageCfg force_drop_text  (text never visible in training)   -> benchmark with --modality audio
#   tonly : StageCfg force_drop_audio (audio always the null audio)      -> benchmark with --modality text
# The other modality keeps its 10 % dropout (CFG null branch still trained); CFG 2.5 at inference as everywhere.
# Each lane = sf_g1.sh lane (r4_lane.sh: train, rollout probe, official benchmark offline + engine, inference seed 0), then
# the semantic recognition of its predictions (dev: hos recognisers; test: v1 recognisers).
#   modality_train.sh <out root>      env GPUS (4 ids: aonly dev, aonly test, tonly dev, tonly test), SIDES (dev test), PY
set -uo pipefail
O=$1; CODE=$(cd "$(dirname "$0")/../.." && pwd); cd $CODE; mkdir -p $O
B=/mnt/cpfs/zbl-cpfs-new/CKPT/zehan/gesture-scmg; SB=$B/semantic-baseline-1105/g1
GL=(${GPUS:-0 1 2 3}); SIDES=${SIDES:-dev test}
log() { echo "$(date -Is) $*" >> $O/modality_train.log; }
log "START sides '$SIDES' gpus '${GL[*]}' commit $(cat GIT_COMMIT 2>/dev/null || git rev-parse HEAD 2>/dev/null)"
# sf_g1.sh hands its GPU list out lane by lane in SIDES order: two invocations = two GPU pairs
( GPUS="${GL[0]} ${GL[1]}" SIDES="$SIDES" SPG=${SPG:-2} INFER_EXTRA="--modality audio" SETTINGS='aonly:"force_drop_text":true' \
    bash scripts/echog_g1/sf_g1.sh $O > $O/sf-aonly.log 2>&1 ) &
( GPUS="${GL[2]} ${GL[3]}" SIDES="$SIDES" SPG=${SPG:-2} INFER_EXTRA="--modality text" SETTINGS='tonly:"force_drop_audio":true' \
    bash scripts/echog_g1/sf_g1.sh $O > $O/sf-tonly.log 2>&1 ) &
wait; log "LANES_DONE"
for side in $SIDES; do
  if [ $side = dev ]; then DATA=$SB/data-sem-hos; RECS="$SB/rec-sem-hos-seed*"; SPL=dev; R=$O/dev
  else DATA=$SB/data-sem; RECS="$SB/rec-sem-v1*"; SPL=test; R=$O/${TEST_SUB:-test}; fi
  specs=(gt=-)
  for lab in aonly tonly; do for p in offline engine; do
    [ -d $R/$lab/bench/$p/seed_000 ] && specs+=($lab-$p=$R/$lab/bench/$p/seed_000)
  done; done
  PY=${PY:-python} bash scripts/semantic/sem_score_dirs.sh ${GL[0]} $DATA "$RECS" $O/sem-$side $SPL "${specs[@]}" > $O/sem-$side.log 2>&1 \
    && log "SEM_DONE $side" || log "SEM_FAILED $side"
done
log "MODALITY_TRAIN_DONE"
