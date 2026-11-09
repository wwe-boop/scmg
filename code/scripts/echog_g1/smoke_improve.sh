#!/bin/bash
# Smoke of the G1 improved recipe + ECHO-G streaming control on this 5090 host (11-04), N val_common clips:
#   BAgger collection (COUNT) -> stream+bagger lane (STEPS) from INIT -> offline AND engine inference -> official evaluator
#   -> stillness probe (dev split) [all inside bagger_r4.sh / r4_lane.sh, PROFILE=g1] ; ECHO-G sliding window (strict:
#   prefix-recomputed audio, voiced text; literal = plain regeneration, strong = inpaint + 1.28 s empty tail) -> evaluator;
#   frame-count check of every prediction against the official GT.
#   GPUS="0 1" INIT=.../base/final.pt VAE=.../vae/final.pt OUT=~/bulk/scmg-local/echog-g1-improve-smoke/run N=10 bash scripts/echog_g1/smoke_improve.sh
set -euo pipefail
CODE=$(cd "$(dirname "$0")/../.." && pwd); cd $CODE
OUT=${OUT:?}; N=${N:-10}; GPUS=${GPUS:-0}; G0=${GPUS%% *}
export PY=${PY:-/home/xr-25b7/envs/part2-exec-20261029/bin/python} SCMG_LOCAL_MIRROR=${SCMG_LOCAL_MIRROR:-$HOME/bulk/cpfs-mirror}
EG=${EG:-/mnt/bulk/zehan/echog-dataset}; OFF=${OFF:-$HOME/bulk/echog-official-ro/official}
EW=${EW:-$HOME/bulk/echog-official-ro}                       # local copies: weights/best.pt, models/wav2vec2-large-xlsr-53-english
COND=${COND:?ECHO-G condition_30fps dir holding the N stems}
mkdir -p $OUT; head -n $N $EG/official-meta/splits/val_common.txt > $OUT/stems.txt
echo "code $(git rev-parse HEAD) $(date -Is)" > $OUT/SMOKE_CODE.txt
PROFILE=g1 INIT=${INIT:?} VAE=${VAE:?} COUNT=${COUNT:-8} NGEN=15 GPUS="$GPUS" LANES=b STEPS=${STEPS:-50} G1_BENCH=1 \
  G1_BENCH_STEMS=$OUT/stems.txt G1_BENCH_SEEDS=0 WORKERS=${WORKERS:-4} bash scripts/bagger_r4.sh $OUT/bag > $OUT/bagger_r4.log 2>&1
for v in literal strong; do
  X=(); [ $v = strong ] && X=(--prefix inpaint --tail-ms 1280)
  CUDA_VISIBLE_DEVICES=$G0 PYTHONPATH=$OFF/src $PY scripts/echog_g1/echog_slide.py --checkpoint $EW/weights/best.pt \
    --config $OFF/configs/sgdit_audio_text.yaml --cond-dir $COND --meta $EG/official-meta --stems $OUT/stems.txt \
    --out $OUT/echog-slide-$v/seed_000 --seed 0 --audio prefix --wav-dir $EG/ds/audio \
    --audio-model $EW/models/wav2vec2-large-xlsr-53-english "${X[@]}" > $OUT/echog-slide-$v.log 2>&1
  bash scripts/echog_g1/bench.sh eval $OUT/echog-slide-$v/seed_000 $OUT/stems.txt echog_slide_$v $OUT/echog-slide-$v/benchmark_seed_000.json >> $OUT/echog-slide-$v.log 2>&1
done
$PY - $OUT $EG/ds/motion_39d_30fps <<'EOF'
import sys, glob, torch
out, gt = sys.argv[1], sys.argv[2]; stems = open(out + '/stems.txt').read().split()
for d in sorted(glob.glob(out + '/bag/r4-stream-bag/bench/*/seed_000') + glob.glob(out + '/echog-slide-*/seed_000')):
    bad = []
    for s in stems:
        p = torch.load(f'{d}/{s}.pt', weights_only=False); g = torch.load(f'{gt}/{s}.pt', weights_only=False)
        n = int(g.get('real_num_frames', len(g['robot_repr'])))
        if tuple(p['robot_repr'].shape) != (min(n, len(g['robot_repr'])), 39) or not torch.isfinite(p['robot_repr']).all(): bad.append(s)
    print('FRAMES', d.replace(out + '/', ''), len(stems), 'clips', 'OK' if not bad else f'MISMATCH {bad}')
EOF
grep -h "G1-emageFGD" $OUT/bag/r4-stream-bag/bench.log $OUT/echog-slide-*.log
echo SMOKE_IMPROVE_DONE $(date -Is)
