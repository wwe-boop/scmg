#!/bin/bash
# Engine-protocol (deployment streaming) full-tier eval of one checkpoint under SCMG_COND conditions (Claude 11-05, landing).
#   cond_eval.sh <gpu> <out root> <label> <ckpt> <SCMG_COND | -> [cfg-w=2.5]
# = baseline_eval.sh engine fgd (43 dev clips, seeds 0/1) + engine probes (rollout 20 clips + semantic 200 clips, seeds 0/1),
# with SCMG_COND exported to scmg.evaluate / rollout_probe.py / gen_semantic.py ('-' = unchanged conditions).
# REFRESH (static-history refresh) and PY pass through to baseline_eval.sh.
set -uo pipefail
gpu=$1; R=$2; lab=$3; ck=$4; cond=$5; w=${6:-2.5}
CODE=$(cd "$(dirname "$0")/../.." && pwd)
[ "$cond" = - ] && unset SCMG_COND || export SCMG_COND="$cond"
mkdir -p $R/$lab/engine
echo "$(date -Is) START label=$lab ckpt=$ck SCMG_COND=${SCMG_COND:-<unset>} cfg_w=$w REFRESH=${REFRESH:-} code=$(git -C $CODE rev-parse --short HEAD)$(git -C $CODE diff --quiet || echo -dirty)" >> $R/$lab/engine/cond_eval.log
for part in fgd probes; do
  NOVOX=1 bash $CODE/scripts/baseline_eval.sh $gpu $R $lab $ck engine $part $w || { echo "$(date -Is) FAILED $part" >> $R/$lab/engine/cond_eval.log; exit 1; }
done
echo "$(date -Is) DONE" >> $R/$lab/engine/cond_eval.log
