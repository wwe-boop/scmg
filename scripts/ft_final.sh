#!/bin/bash
# 11-03: probes on a fine-tune lane's FINAL checkpoint (best.pt is chosen by a dev loss on the human dev split, which has no
# semsynth/word labels and rises after ~1500 fine-tune steps, so it picks a barely-trained text route).
#   ft_final.sh <gpu> <out root> <lane> [cfg-w=2.0]   -> <lane>/final-eval/{planner_probe.json, semantic/, rollout_probe.json}
set -uo pipefail
gpu=$1; R=$2; o=$3; w=${4:-2.0}
CODE=$(cd "$(dirname "$0")/.." && pwd); cd $CODE; export PYTHONPATH=$CODE USE_TF=0 OMP_NUM_THREADS=${OMP_NUM_THREADS:-4} CUDA_VISIBLE_DEVICES=$gpu
PY=${PY:-python}; B=/mnt/cpfs/zbl-cpfs-new/CKPT/zehan/gesture-scmg; VAE=$B/trained/vae-r3/final.pt; SEMMAN=$B/semantic-recognizer-v1/manifest-n200
F=$R/$o/final-eval; mkdir -p $F; ck=$R/$o/final.pt
source "$CODE/scripts/artifacts.sh"
$PY -m scmg.token_cache "$ck" --entry ft_final.sh || exit 1
json_run "$F/planner_probe.json" planner stdout $PY scripts/planner_probe.py "$VAE" "$o-final=$ck" 2>> "$F/probe.log" || exit 1
semantic_run "$F/semantic" "$o" "$ck" "$w" own,notext,swap 200 >> "$F/semantic.log" 2>&1 || exit 1
json_run "$F/rollout_probe.json" rollout stdout $PY scripts/rollout_probe.py "$VAE" "$o-final=$ck@$w" 2>> "$F/probe.log" || exit 1
echo "$(date -Is) FINAL_DONE $o"
