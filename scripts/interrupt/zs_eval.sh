#!/bin/bash
# zehan-server: interruption runs of one floor lane on one GPU (native CPFS paths, <= 4 CPU threads)
#   zs_eval.sh <gpu> <ckpt> <out dir> <side dev|test> <floor values...>
set -uo pipefail
G=$1; CK=$2; O=$3; SIDE=$4; shift 4
I=/mnt/cpfs/zbl-cpfs-new/CKPT/zehan/gesture-scmg/interrupt-baseline-1105; B=/mnt/cpfs/zbl-cpfs-new/CKPT/zehan/gesture-scmg; C=/mnt/cpfs/zbl-cpfs-new/dataset/zehan/gesture
export SCMG_J=39 SCMG_NO_GLITCH=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=$G PYTHONPATH=$I/code-floor
PY=/mnt/cpfs/zbl-cpfs-new/USERS/zehan/envs/echog/bin/python; VAE=$B/echog-g1-decft-1105/decft-w3d0m05/final.pt; STATS=$C/scmg/STATS_echog_g1_v1.json
if [ $SIDE = dev ]; then V=$B/views/echog-g1-v1-hos; T=$C/scmg/text-e5-echog-g1-v1; TC=$B/models/text-e5-causal-echog-g1-v1; SPL=val
else V=$C/echog-g1-v1/test; T=$C/scmg/text-e5-echog-g1-v1-test; TC=$B/models/text-e5-causal-echog-g1-v1-test; SPL=test; fi
A=(--ckpt $CK --vae $VAE --stats $STATS --view $V --text-cache $T --split $SPL --protocol engine --text-causal $TC)
cd $I/code-floor
EV=${EVERY:-1}; ST=$I/stems_$SIDE.txt; [ $EV -gt 1 ] && { LC_ALL=C sort $ST | awk "NR % $EV == 1" > $O.stems; ST=$O.stems; }
$PY scripts/interrupt/infer_retract.py "${A[@]}" --out $O/no_int --no-interrupt --stems $ST > $O.noint.log 2>&1
for v in "$@"; do
  $PY scripts/interrupt/infer_retract.py "${A[@]}" --out $O/f$v --interrupts $I/interrupts_$SIDE.json --words $I/utterances.json \
    --audio end --restart chunk --prefix-feats $I/prefix_$SIDE --floor $v --every $EV > $O.f$v.log 2>&1
done
echo ZS_DONE >> $O.noint.log
