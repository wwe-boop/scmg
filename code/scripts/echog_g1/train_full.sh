#!/bin/bash
# ECHO-G G1 full training + official benchmark, ONE node with NGPU GPUs (proposal, not run by the smoke):
#   VAE (vae-r3 recipe, 1 GPU) -> base (r4 base recipe, DDP on all GPUs) -> val_common inference sharded over the GPUs
#   (seeds 0 and 1) -> official evaluator, all 39 channels (+ report-only legs-masked lane when LEGMASK_EVAL=1).
#   NGPU=8 R=/mnt/cpfs/zbl-cpfs-new/CKPT/zehan/gesture-scmg/echog-g1-r1 EG=<echo-g data root> OFF=<official code> \
#     bash scripts/echog_g1/train_full.sh
# Leg-masking TRAIN lane (default off): VAE_CFG='{"dim_mask": [10,11,12,13,14,15,16,17,18,19,20,21]}' BASE_SC='{"dim_mask": [...]}'
set -euo pipefail
NGPU=${NGPU:-8}; R=${R:?output dir}; PY=${PY:-python}
C=/mnt/cpfs/zbl-cpfs-new/dataset/zehan/gesture
VIEW=${VIEW:-$C/echog-g1-v1}; STATS=${STATS:-$C/scmg/STATS_echog_g1_v1.json}
TEXT=${TEXT:-$C/scmg/text-e5-echog-g1-v1}; TEXT_TEST=${TEXT_TEST:-$C/scmg/text-e5-echog-g1-v1-test}
SEEDS=${SEEDS:-0}; CFGW=${CFGW:-2.5}          # training seeds (shared VAE); inference CFG fixed a priori = r4 recipe (not tuned on test)
VAE_STEPS=${VAE_STEPS:-30000}; BASE_STEPS=${BASE_STEPS:-8000}
if [ "${TRAIN_ONLY:-0}" != 1 ]; then
  EG=${EG:?ECHO-G data root with motion_39d_30fps/ audio/ splits/ eval_assets/}; OFF=${OFF:?official ECHO-G code dir}
  FGD_CKPT=${FGD_CKPT:?weights/evaluation/g1_fgd_encoder.pt}
fi
export SCMG_J=39 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
mkdir -p $R
if [ ! -f $R/vae/DONE.json ]; then
  CUDA_VISIBLE_DEVICES=0 $PY -m scmg.train --stage vae --view $VIEW --stats $STATS --out $R/vae --steps $VAE_STEPS --batch 128 \
    --lr 1e-3 --warmup 500 --frames 256 --workers 6 --deriv-ramp 3000 --snap-every 0 --eval-every 2000 \
    ${VAE_CFG:+--vae-cfg "$VAE_CFG"} >> $R/vae.log 2>&1
fi
for ts in $SEEDS; do B=$R/base-s$ts; [ -f $B/DONE.json ] && continue
  $PY -m torch.distributed.run --standalone --nproc_per_node=$NGPU -m scmg.train --stage base --view $VIEW --stats $STATS \
    --text-cache $TEXT --vae $R/vae/final.pt --out $B --seed $ts --batch 128 --steps $BASE_STEPS --lr 3e-4 --warmup 1000 --frames 256 \
    --workers 6 --eval-every 500 --snap-every 2000 ${BASE_SC:+--sc "$BASE_SC"} >> $R/base-s$ts.log 2>&1
done
[ "${TRAIN_ONLY:-0}" = 1 ] && { echo TRAIN_DONE $(date -Is); exit 0; }
for ts in $SEEDS; do B=$R/base-s$ts
for seed in 0 1; do
  ss=$(printf %03d $seed); P=$R/preds/train$ts/seed_$ss
  for g in $(seq 0 $((NGPU - 1))); do
    CUDA_VISIBLE_DEVICES=$g $PY scripts/echog_g1/infer_test.py --ckpt $B/best.pt --vae $R/vae/final.pt --stats $STATS \
      --view $VIEW/test --text-cache $TEXT_TEST --out $P --seed $seed --steps 10 --cfg-w $CFGW --shard $g/$NGPU >> $R/infer_t${ts}_$ss.log 2>&1 &
  done; wait
  PYTHONPATH=$OFF/src $PY $OFF/scripts/eval_g1_motion_cls.py --pred-dir $P --ref-dir $EG/motion_39d_30fps \
    --val-split $EG/splits/val_common.txt --wav-dir $EG/audio --mmae-file $EG/eval_assets/mmae/g1_mmae_30body_30fps.npy \
    --g1-ae-ckpt $FGD_CKPT --fps 30 --ba-direction audio_to_motion --enable-foot-metrics --require-all-stems \
    --tag scmg_g1_train${ts}_seed$seed --out $R/benchmark_train${ts}_seed$seed.json >> $R/eval.log 2>&1
  if [ "${LEGMASK_EVAL:-0}" = 1 ]; then
    $PY scripts/echog_g1/mask_dirs.py legs $P $EG/motion_39d_30fps $EG/splits/val_common.txt $R/masked_legs_t${ts}_$ss
    PYTHONPATH=$OFF/src $PY $OFF/scripts/eval_g1_motion_cls.py --pred-dir $R/masked_legs_t${ts}_$ss/pred --ref-dir $R/masked_legs_t${ts}_$ss/gt \
      --val-split $EG/splits/val_common.txt --wav-dir $EG/audio --mmae-file $EG/eval_assets/mmae/g1_mmae_30body_30fps.npy \
      --g1-ae-ckpt $FGD_CKPT --fps 30 --ba-direction audio_to_motion --enable-foot-metrics --require-all-stems \
      --tag scmg_g1_train${ts}_seed${seed}_legsmasked --out $R/benchmark_train${ts}_seed${seed}_legsmasked.json >> $R/eval.log 2>&1
  fi
done; done
echo FULL_DONE $(date -Is)
