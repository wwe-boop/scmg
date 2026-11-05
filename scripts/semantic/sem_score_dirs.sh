#!/bin/bash
# Semantic recognition (pair_win / det_auc / bal_top1) of G1 prediction dirs with every recogniser seed, then the
# seed-averaged table (sem_report.py).  11-05 semantic-head lanes.
#   sem_score_dirs.sh <gpu> <data dir> <rec dirs glob> <scores dir> <split: dev|test> label=<pred dir> ...
#   e.g. sem_score_dirs.sh 0 $G/data-sem-hos "$G/rec-sem-hos-seed*" $O/sem-dev dev gt=- sem1-offline=$O/dev/sem1/bench/offline/seed_000
# label=- scores the ground truth.  Existing score files are kept (delete to recompute).
set -uo pipefail
gpu=$1; data=$2; recs=$3; S=$4; split=$5; shift 5
CODE=$(cd "$(dirname "$0")/../.." && pwd); PY=${PY:-python}
mkdir -p $S
for rec in $recs; do
  r=$(basename $rec)
  for spec in "$@"; do
    lab=${spec%%=*}; d=${spec#*=}; o=$S/$r--$lab--$split.json; [ -s $o ] && continue
    pt=(); [ "$d" != - ] && pt=(--pt-dir $d)
    CUDA_VISIBLE_DEVICES=$gpu $PY $CODE/scripts/semantic/g1_recognizer.py score --data $data --ckpt $rec/model.pt --name $lab "${pt[@]}" \
      --split $split --out $o --save-probs ${SCORE_EXTRA:-} 2>&1 | grep -v -i warn || echo "SCORE_FAILED $lab $r"
  done
done
r0=$(basename $(ls -d $recs | head -1)); pre=${r0%seed*}
$PY $CODE/scripts/semantic/sem_report.py --scores $S --rec-prefix "$pre" --split $split --pairs "${PAIRS:-}" --out $S/REPORT.json
