#!/bin/bash
# Score every G1 val_common source (GT + generator outputs) with every recogniser checkpoint (semantic-baseline-1105).
#   g1_score_all.sh <gpu> <G root (data, data-sem, rec-*, outputs/)>
set -uo pipefail
gpu=$1; G=$2
CODE=$(cd "$(dirname "$0")/../.." && pwd); PY=${PY:-/home/xr-25b7/envs/part2-exec-20261029/bin/python}
mkdir -p $G/scores
for rec in $G/rec-sem-v1*/model.pt $G/rec-v1/model.pt; do
  r=$(basename $(dirname $rec)); case $r in rec-sem*) data=$G/data-sem;; *) data=$G/data;; esac
  for src in gt $(ls $G/outputs); do
    o=$G/scores/$r--$src--test.json; [ -s $o ] && continue
    pt=(); [ $src != gt ] && pt=(--pt-dir $G/outputs/$src)
    CUDA_VISIBLE_DEVICES=$gpu $PY $CODE/scripts/semantic/g1_recognizer.py score --data $data --ckpt $rec --name $src "${pt[@]}" \
      --split test --out $o --save-probs 2>&1 | grep -v -i warn
  done
done
