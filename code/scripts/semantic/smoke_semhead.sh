#!/bin/bash
# Smoke of the G1 semantic-head lane (11-05): a few dozen training steps with the BEAT2-annotation g_k track
# (prints the gesture CE / accuracy and the chunk label distribution), then the zero-init identity check
# (scripts/semantic/check_semhead_identity.py: hos-base vs hos-base + sem_head/sem_cond, bitwise) and one inference of the
# smoke checkpoint.  On the 5090 host with the local CPFS mirror:
#   smoke_semhead.sh <gpu> <out dir> [steps=40]
set -uo pipefail
gpu=$1; O=$2; ST=${3:-40}
CODE=$(cd "$(dirname "$0")/../.." && pwd); cd $CODE
PY=${PY:-/home/xr-25b7/envs/part2-exec-20261029/bin/python}
B=/mnt/cpfs/zbl-cpfs-new/CKPT/zehan/gesture-scmg; D=/mnt/cpfs/zbl-cpfs-new/dataset/zehan/gesture/scmg; M=${SCMG_LOCAL_MIRROR:-$HOME/bulk/cpfs-mirror}
export SCMG_LOCAL_MIRROR=$M SCMG_J=39 SCMG_NO_GLITCH=1 USE_TF=0 OMP_NUM_THREADS=4 PYTHONPATH=$CODE CUDA_VISIBLE_DEVICES=$gpu
export SCMG_TEXT_CAUSAL=$B/models/text-e5-causal-echog-g1-v1
loc(){ local q=$M${1#/mnt/cpfs/zbl-cpfs-new}; [ -e $q ] && echo $q || echo $1; }
INIT=$(loc $B/echog-g1-arch-1104/hos-base/best.pt); VAE=$(loc $B/echog-g1-decft-1105/decft-w3d0m05/final.pt)
GEST=$B/models/gest-beat2sem-echog-g1-v1-hos; HV=$B/views/echog-g1-v1-hos
SC='{"p_stream": 0.67, "p_async": 1.0, "p_async_seg": 0.0, "self_force_p": 0, "w_gest": 1.0, "gest_cw": [1.0, 3.8, 3.0, 3.0, 0.7, 1.0, 1.0]}'
mkdir -p $O
$PY -m scmg.train --out $O/train --init $INIT --view $HV --stats $D/STATS_echog_g1_v1.json --text-cache $D/text-e5-echog-g1-v1 \
  --gest-frames $GEST --vae $VAE --workers 2 --batch 16 --stage base --steps $ST --lr 1e-4 --warmup 10 --eval-every 1000000 \
  --snap-every 1000000 --log-every 10 --mmap --seed 0 --cfg '{"adsc": true, "sem_head": true, "sem_cond": true}' --sc "$SC" \
  > $O/train.log 2>&1 || { echo "TRAIN_FAILED"; tail -30 $O/train.log; exit 1; }
grep -E "added modules|trainable" $O/train.log
$PY - $O/train/train.jsonl <<'EOF'
import json, sys
for l in open(sys.argv[1]):
    r = json.loads(l)
    print('step', r.get('step'), 'loss %.4f' % r['loss'], 'gest %.4f' % r.get('gest', float('nan')), 'gest_acc %.3f' % r.get('gest_acc', float('nan')),
          'chunk labels', {k: round(v, 4) for k, v in r.items() if k.startswith('glab_')})
EOF
$PY scripts/semantic/check_semhead_identity.py --init $INIT --vae $VAE --view $HV --stats $D/STATS_echog_g1_v1.json \
  --text-cache $D/text-e5-echog-g1-v1 --smoke $O/train/final.pt
