#!/bin/bash
# B3 runs of the G1 interruption benchmark on xr-xx (2 x 5090), 11-05: per GPU shard, sequentially
#   1. --no-interrupt full-clip engine outputs of the protocol stems (uninterrupted reference / B1 / B2 base, same GPU type)
#   2. interruption, audio = end   3. interruption, audio = silence
#   run_b3.sh <work dir> <interrupts json> <stems txt> <utterances json> [view split=test]
set -uo pipefail
W=$1; INT=$2; ST=$3; UT=$4; SPL=${5:-test}
source $W/run_env.sh; cd $CODE
VIEW=${VIEW:-$VIEW_TEST}; TXT=${TXT:-$TEXT_TEST}; TXC=${TXC:-$TEXT_CAUSAL_TEST}; TAG=${TAG:-test}
COMMON=(--ckpt $CK --vae $VAE --stats $STATS --view $VIEW --text-cache $TXT --split $SPL --protocol engine --text-causal $TXC)
for g in 0 1; do
  ( CUDA_VISIBLE_DEVICES=$g $PY scripts/interrupt/infer_interrupt.py "${COMMON[@]}" --out $W/b3_$TAG/no_int --no-interrupt --stems $ST --shard $g/2
    CUDA_VISIBLE_DEVICES=$g $PY scripts/interrupt/infer_interrupt.py "${COMMON[@]}" --out $W/b3_$TAG/end --interrupts $INT --words $UT --audio end --shard $g/2
    CUDA_VISIBLE_DEVICES=$g $PY scripts/interrupt/infer_interrupt.py "${COMMON[@]}" --out $W/b3_$TAG/silence --interrupts $INT --words $UT --audio silence --shard $g/2
  ) > $W/logs/b3_${TAG}_gpu$g.log 2>&1 &
done
wait; echo "$(date -Is) B3_DONE $TAG" >> $W/logs/b3.log
