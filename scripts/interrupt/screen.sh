#!/bin/bash
# Dev screening of the zero-training retraction settings (11-05): every config = "label|infer_retract args", run on both
# GPUs (2 shards) one config after the other.   screen.sh <work dir> <interrupts json> <out root> <config file> [split=val]
set -uo pipefail
W=$1; INT=$2; O=$3; CF=$4; export SPL=${5:-val}
R=$(dirname "$0"); U=$HOME/bulk/interrupt-1105/utterances.json; HJ=$HOME/bulk/interrupt-1105/human_ref_v2_train_drop.json
while IFS='|' read -r lab args; do
  [ -z "$lab" ] && continue; case $lab in \#*) continue;; esac
  for g in 0 1; do
    bash $HOME/bulk/interrupt-1105/rr.sh $W $g $O/$lab --interrupts $INT --words $U --human $HJ --shard $g/2 $args > $O/$lab.gpu$g.log 2>&1 < /dev/null &
  done; wait; echo "$(date -Is) DONE $lab" >> $O/screen.log
done < $CF
echo "$(date -Is) SCREEN_DONE" >> $O/screen.log
