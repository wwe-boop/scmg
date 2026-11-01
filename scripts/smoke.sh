#!/bin/bash
# End-to-end smoke test of every stage with a tiny model (speech_server GPU1). usage: bash smoke.sh <outdir>
set -euo pipefail
O=${1:-/tmp/zehan-scmg-smoke}; rm -rf $O; mkdir -p $O
D=/mnt/cpfs/zbl-cpfs-new/dataset/zehan/gesture/scmg
C="--stats $D/STATS_v36_1.json --text-cache $D/text-e5-v36_1 --mmap --workers 3 --batch 16 --log-every 10 --eval-every 1000000 --save-every 1000000"
TINY='{"d":128,"heads":4,"audio_layers":1,"planner_layers":2,"flow_layers":2}'
cd "$(dirname "$0")/.."; export PYTHONPATH=$PWD
export CUDA_VISIBLE_DEVICES=${GPU:-1}
python -m scmg.train --stage vae --out $O/vae --steps 60 --warmup 10 $C
python -m scmg.train --stage base --out $O/base --vae $O/vae/final.pt --steps 40 --warmup 5 --cfg "$TINY" $C
python -m scmg.train --stage peft --out $O/peft --vae $O/vae/final.pt --init $O/base/final.pt --steps 30 --warmup 5 $C
python -m scmg.train --stage causal --out $O/causal --vae $O/vae/final.pt --init $O/peft/final.pt --steps 30 --warmup 5 $C
python -m scmg.train --stage state --out $O/state --vae $O/vae/final.pt --init $O/causal/final.pt --steps 30 --warmup 5 $C
python -m scmg.train --stage state --out $O/measured --vae $O/vae/final.pt --init $O/causal/final.pt --steps 20 --warmup 5 --sc '{"state_hist_measured": true}' $C
python -m scmg.evaluate --ckpt $O/measured/final.pt --vae $O/vae/final.pt --stats $D/STATS_v36_1.json --text-cache $D/text-e5-v36_1 --n 3 --horizons 4 --out $O/eval-measured --closed-loop --mmap
python -m scmg.evaluate --ckpt $O/state/final.pt --vae $O/vae/final.pt --stats $D/STATS_v36_1.json --text-cache $D/text-e5-v36_1 --n 3 --horizons inf,0 --out $O/eval --closed-loop --mmap
echo SMOKE_OK
