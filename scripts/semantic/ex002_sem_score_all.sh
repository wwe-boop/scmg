#!/bin/bash
# EX002 dev-43 BEAT2-annotation semantic recognition: score GT + lane exports with every recogniser seed (semantic-baseline-1105).
#   ex002_sem_score_all.sh <gpu> <E root (data, rec-sem-seed*)> <exports root (<lane>/<offline|engine>/eval-s<k>/gen_H*_<ci>.npy)>
set -uo pipefail
gpu=$1; E=$2; X=$3
CODE=$(cd "$(dirname "$0")/../.." && pwd); PY=${PY:-/home/xr-25b7/envs/part2-exec-20261029/bin/python}
mkdir -p $E/scores
ref=$(ls -d $X/*/offline/eval-s0 | head -1)
for rec in $E/rec-sem-seed*/model.pt; do
  r=$(basename $(dirname $rec))
  run(){ o=$E/scores/$r--$1--test.json; [ -s $o ] && return; shift
    CUDA_VISIBLE_DEVICES=$gpu $PY $CODE/scripts/semantic/g1_recognizer.py score --data $E/data --ckpt $rec --split test --out $o --save-probs "$@" 2>&1 | grep -v -i warn; }
  run gt --name gt --npy-pattern 'gen_Hinf_{ci}.npy' --gen-len-from $ref
  for d in $X/*/*/eval-s*; do
    p=$(basename $(dirname $d)); lane=$(basename $(dirname $(dirname $d))); s=$(basename $d)
    pat='gen_Hinf_{ci}.npy'; [ $p = engine ] && pat='gen_H2_{ci}.npy'
    run $lane-$p-${s#eval-} --name $lane-$p-${s#eval-} --pt-dir $d --npy-pattern "$pat"
  done
done
