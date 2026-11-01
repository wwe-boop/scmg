#!/bin/bash
# Smoke test of the VoxCPM-style planner through every stage (small flow, real Qwen2.5-0.5B TSLM).
set -euo pipefail
O=${1:-/tmp/zehan-scmg-smokevox}; rm -rf $O; mkdir -p $O
D=/mnt/cpfs/zbl-cpfs-new/dataset/zehan/gesture/scmg
VAE=/mnt/cpfs/zbl-cpfs-new/CKPT/zehan/gesture-scmg/trained/vae-r3/final.pt
C="--stats $D/STATS_v36_1.json --text-cache $D/text-e5-v36_1 --tok-cache $D/tok-qwen25-v36_1 --vae $VAE --mmap --workers 2 --batch 8 --log-every 10 --eval-every 1000000 --save-every 1000000 --snap-every 0"
CFG='{"planner":"voxcpm","d":256,"heads":4,"audio_layers":1,"flow_layers":2}'
cd "$(dirname "$0")/.."; export PYTHONPATH=$PWD
export CUDA_VISIBLE_DEVICES=${GPU:-1}
python -m scmg.train --stage base --out $O/base --steps 30 --warmup 5 --cfg "$CFG" $C
python -m scmg.train --stage peft --out $O/peft --init $O/base/final.pt --steps 20 --warmup 5 $C
python -m scmg.train --stage causal --out $O/causal --init $O/peft/final.pt --steps 20 --warmup 5 $C
python -m scmg.train --stage state --out $O/state --init $O/causal/final.pt --steps 20 --warmup 5 $C
python -m scmg.evaluate --ckpt $O/state/final.pt --vae $VAE --stats $D/STATS_v36_1.json --text-cache $D/text-e5-v36_1 \
  --tok-cache $D/tok-qwen25-v36_1 --n 2 --horizons inf,0 --out $O/eval --closed-loop --mmap
echo SMOKE_VOX_OK
