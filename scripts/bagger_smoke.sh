#!/bin/bash
set -euo pipefail
source "$(dirname "$0")/bagger_env.sh"
export CUDA_VISIBLE_DEVICES=${1:-1}
B=/mnt/cpfs/zbl-cpfs-new/CKPT/zehan/gesture-scmg
R=$B/bagger-1103/smoke
D=/mnt/cpfs/zbl-cpfs-new/dataset/zehan/gesture/scmg
I=$B/dlc-scmg-train-r7c-1102-voxabl/vox-randlm/best.pt
V=$B/trained/vae-r3/final.pt
mkdir -p "$R"
"$PY" scripts/bagger_collect.py --ckpt "$I" --out "$R/data" --cond null --count 4 --dry-run > "$R/dry-run.log" 2>&1
"$PY" scripts/bagger_collect.py --ckpt "$I" --out "$R/data" --cond null --count 4 > "$R/collect.log" 2>&1
"$PY" -m scmg.train --stage base --out "$R/train" --init "$I" --vae "$V" --stats "$D/STATS_v36_1.json" \
  --text-cache "$D/text-e5-v36_1" --tok-cache "$D/tok-qwen25-v36_1" --mmap --workers 0 \
  --holdout "$B/semantic-recognizer-v1/manifest-n200/MANIFEST.json" \
  --batch 32 --steps 20 --warmup 200 --lr 1e-4 --log-every 5 --eval-every 20 --save-every 20 --snap-every 0 \
  --cfg '{"vox_prefsq_norm": true}' --sc '{}' --bagger-dir "$R/data" --bagger-share .33 > "$R/train.log" 2>&1
# Own resume only; task explicitly requires deletion, rather than relocation.
rm -- "$R/train/resume.pt"
"$PY" scripts/bagger_collect.py --ckpt "$R/train/final.pt" --out "$R/recollect" --cond reverse --count 2 > "$R/recollect.log" 2>&1
printf 'SMOKE_PASS\n' > "$R/PASS"
