#!/bin/bash
# 11-08 reference-training regression check (user: "确保不要出现bug"; coordinator: a silent training drift must not go unnoticed).
# A fixed short training of the EX002 base recipe (sibag: r4 base init, adsc + intent_aux, ADSC sc, self-forcing p 0.5 x 4 steps,
# BAgger 0.33, semsynth 0.6, seed 0, batch 64, 300 steps, lr 1e-4, warmup 200 -> fixed data, fixed RNG), then an offline bench
# (per_clip seed 0, 10 flow steps, cfg 2.0) of a fixed quarter of the dev clips with the CURRENT bench code, then
# ref_train_ex002.py summarises loss / grad-norm curves + the 13-point liveliness stats and (CHECK=1) compares them with
# scripts/tests/ref/REF_TRAIN_EX002.json. Run on an A800 (DLC) for every release before any lane is submitted:
#   CUDA_VISIBLE_DEVICES=g TREE=<code tree> BENCH=<release> bash scripts/tests/ref_train_ex002.sh <out dir>
# TREE = code whose TRAINING is checked (default: the tree holding this script); BENCH = code for inference (default TREE).
set -uo pipefail
O=${1:?out dir}; mkdir -p $O
HERE=$(cd "$(dirname "$0")/../.." && pwd); TREE=${TREE:-$HERE}; BENCH=${BENCH:-$TREE}
B=/mnt/cpfs/zbl-cpfs-new/CKPT/zehan/gesture-scmg; D=/mnt/cpfs/zbl-cpfs-new/dataset/zehan/gesture/scmg
PY=${PY:-/usr/local/bin/python}; export SCMG_TEXT_CAUSAL=$B/models/text-e5-causal-v36_1 OMP_NUM_THREADS=1 USE_TF=0
unset SCMG_J SCMG_LOCAL_MIRROR
SIB='{"p_stream": 0.67, "p_async": 1.0, "p_async_seg": 0.0, "w_int": 0.3, "sem_w": 2.0, "self_force_cfg": 0.0, "self_force_mix": 0.0, "self_force_noise": 0.0}'
STEPS=${REF_STEPS:-300}
[ -f $O/train/DONE.json ] || ( cd $TREE && PYTHONPATH=$TREE $PY -m scmg.train --out $O/train --init $B/dlc-scmg-train-r4-1102/base/best.pt \
  --stats $D/STATS_v36_1.json --text-cache $D/text-e5-v36_1 --tok-cache $D/tok-qwen25-v36_1 --tok-intent $B/models/intent-qwen25-v36_1 \
  --intent-frames $B/models/intent-frames-v36_1 --holdout $B/semantic-recognizer-v1/manifest-n200/MANIFEST.json --vae $B/trained/vae-r3/final.pt \
  --workers 5 --semsynth-share 0.6 --batch 64 --stage base --steps $STEPS --lr 1e-4 --warmup 200 --eval-every 150 --log-every 25 --snap-every 1000000 \
  --seed 0 --cfg '{"adsc": true, "intent_aux": true}' --sc "$SIB" --self-force-p 0.5 --self-force-steps 4 --self-force-rounds 1 \
  --bagger-dir $B/r4-1104/bagger-pai/bagger-data --bagger-share 0.33 > $O/train.log 2>&1 ) || { echo "REF_TRAIN_FAILED $O"; exit 1; }
[ -f $O/offline/s0/PROTOCOL.json ] || ( cd $BENCH && PYTHONPATH=$BENCH $PY scripts/ex002/bench_ex002.py infer --ckpt $O/train/final.pt \
  --protocol offline --seed 0 --out $O/offline/s0 --shard 0/4 > $O/bench.log 2>&1 ) || { echo "REF_BENCH_FAILED $O"; exit 1; }
( cd $HERE && PYTHONPATH=$HERE $PY scripts/tests/ref_train_ex002.py summarise $O ${CHECK:+--check scripts/tests/ref/REF_TRAIN_EX002.json} )
