#!/bin/bash
# ECHO-G G1 smoke (local 5090): VAE ~200 steps (vae-r3 recipe) -> base ~200 steps (r4 base recipe) -> infer N test clips
# (50 Hz) -> 30 FPS -> official evaluator on those clips (all 39 channels) + the report-only legs-masked variant.
#   GPU=0 OUT=~/bulk/scmg-local/echog-g1-smoke N=20 bash scripts/echog_g1/smoke.sh
set -euo pipefail
GPU=${GPU:?set GPU}; OUT=${OUT:-$HOME/bulk/scmg-local/echog-g1-smoke}; N=${N:-20}; STEPS=${STEPS:-200}
CODE=$(cd "$(dirname "$0")/../.." && pwd)
PY=${PY:-/home/xr-25b7/envs/part2-exec-20261029/bin/python}
C=/mnt/cpfs/zbl-cpfs-new/dataset/zehan/gesture
VIEW=${VIEW:-$C/echog-g1-v1}; STATS=${STATS:-$C/scmg/STATS_echog_g1_v1.json}; TEXT=${TEXT:-$C/scmg/text-e5-echog-g1-v1}; TEXT_TEST=${TEXT_TEST:-$C/scmg/text-e5-echog-g1-v1-test}
EG=${EG:-/mnt/bulk/zehan/echog-dataset}; OFF=${OFF:-$HOME/bulk/echog-official-ro/official}
export SCMG_J=39 SCMG_LOCAL_MIRROR=${SCMG_LOCAL_MIRROR:-$HOME/bulk/cpfs-mirror} CUDA_VISIBLE_DEVICES=$GPU PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
L() { local m=$SCMG_LOCAL_MIRROR${1#/mnt/cpfs/zbl-cpfs-new}; [ -e "$m" ] && echo "$m" || echo "$1"; }
mkdir -p $OUT; cd $CODE
echo "code $(git -C $CODE rev-parse HEAD) $(date -Is)" | tee $OUT/SMOKE_CODE.txt
if [ ! -f $OUT/vae/final.pt ]; then
  $PY -m scmg.train --stage vae --view $VIEW --stats $(L $STATS) --out $OUT/vae --steps $STEPS --batch 128 --lr 1e-3 \
      --warmup ${VAE_WARMUP:-500} --frames 256 --workers 6 --eval-every 100 --save-every 100 --log-every 10 --snap-every 0 \
      ${VAE_CFG:+--vae-cfg "$VAE_CFG"} 2>&1 | tee $OUT/vae.log
fi
if [ ! -f $OUT/base/final.pt ]; then
  $PY -m scmg.train --stage base --view $VIEW --stats $(L $STATS) --text-cache $TEXT --vae $OUT/vae/final.pt --out $OUT/base \
      --steps $STEPS --batch ${BASE_BATCH:-128} --lr 3e-4 --warmup ${BASE_WARMUP:-1000} --frames 256 --workers 6 --eval-every 100 \
      --save-every 100 --log-every 10 --snap-every 0 ${BASE_SC:+--sc "$BASE_SC"} 2>&1 | tee $OUT/base.log
fi
head -n $N $EG/official-meta/splits/val_common.txt > $OUT/stems_$N.txt
$PY scripts/echog_g1/infer_test.py --ckpt $OUT/base/best.pt --vae $OUT/vae/final.pt --stats $(L $STATS) --view $VIEW/test \
    --text-cache $TEXT_TEST --out $OUT/preds/seed_000 --stems $OUT/stems_$N.txt --seed 0 --steps 10 --save50 2>&1 | tee $OUT/infer.log
EV="--ref-dir $EG/ds/motion_39d_30fps --val-split $OUT/stems_$N.txt --wav-dir $EG/ds/audio
    --mmae-file $EG/official-meta/eval_assets/mmae/g1_mmae_30body_30fps.npy --g1-ae-ckpt $EG/weights-ro/evaluation/g1_fgd_encoder.pt
    --fps 30 --ba-direction audio_to_motion --enable-foot-metrics --require-all-stems"
PYTHONPATH=$OFF/src $PY $OFF/scripts/eval_g1_motion_cls.py --pred-dir $OUT/preds/seed_000 $EV --tag scmg_g1_smoke \
    --out $OUT/benchmark_seed0.json 2>&1 | tail -40 | tee $OUT/eval.log
$PY scripts/echog_g1/mask_dirs.py legs $OUT/preds/seed_000 $EG/ds/motion_39d_30fps $OUT/stems_$N.txt $OUT/masked_legs
PYTHONPATH=$OFF/src $PY $OFF/scripts/eval_g1_motion_cls.py --pred-dir $OUT/masked_legs/pred \
    $(echo $EV | sed "s#--ref-dir [^ ]*#--ref-dir $OUT/masked_legs/gt#") --tag scmg_g1_smoke_legsmasked \
    --out $OUT/benchmark_seed0_legsmasked.json 2>&1 | tail -20 | tee $OUT/eval_legsmasked.log
echo SMOKE_DONE $(date -Is)
