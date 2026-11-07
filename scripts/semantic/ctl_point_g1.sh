#!/bin/bash
# 11-07 pointing-only SCMG quick lane at the 2000-step protocol (criteria: docs/criteria/SEMLINE_7_55.md, "Pointing-only lane",
# written before this run). ctlpt = the ctlcf recipe (adsc + ctl + ctl_flow; ctl_w 4, ray loss w_ctl_aux 3, field drop 0.2,
# counterfactual targets 0.5, ctl lr x10, ctl_focus 0.5) on ctl-tokens-hos (no counts), hos-base warm start, 2000 steps, seed 0.
# Inference seeds 0 / 1 / 2, per_clip, offline, CFG 2.5, 10 steps: none / gt on the 400 dev clips (official benchmark), shuffle on
# the annotation clips; g1_ctl_eval.py (STG, Score_a / Score_b) per seed. 3 GPUs: train on GPU0, then one seed per GPU.
#   ctl_point_g1.sh <out dir>     env PY
set -uo pipefail
O=$1; CODE=$(cd "$(dirname "$0")/../.." && pwd); cd $CODE; mkdir -p $O
B=/mnt/cpfs/zbl-cpfs-new/CKPT/zehan/gesture-scmg; C=/mnt/cpfs/zbl-cpfs-new/dataset/zehan/gesture
DEC=$B/echog-g1-decft-1105/decft-w3d0m05/final.pt; A=$B/echog-g1-hos-assets; HV=$B/views/echog-g1-v1-hos
CTL0=$B/control-token-1105/ctl-tokens-hos; INIT=$B/echog-g1-arch-1104/hos-base/best.pt
PY=${PY:-python}; export SCMG_J=39 SCMG_NO_GLITCH=1 OMP_NUM_THREADS=${OMP_NUM_THREADS:-2}
W=/mnt/cpfs/zbl-cpfs-new/USERS/zehan/baselines/echog-official; D4=$B/baseline-repro-1104/echog/data_root
export EG=$D4 OFF=$W/official MMAE=$D4/eval_assets/mmae/g1_mmae_30body_30fps.npy FGD_CKPT=$W/weights/evaluation/g1_fgd_encoder.pt
export REF_DIR=$A/motion_39d_30fps WAV_DIR=$A/audio SPLIT=$A/dev_stems.txt VIEW_TEST=$HV VSPLIT=val TEXT_TEST=$C/scmg/text-e5-echog-g1-v1 \
  TEXT_CAUSAL_TEST=$B/models/text-e5-causal-echog-g1-v1 STATS=$C/scmg/STATS_echog_g1_v1.json CFGW=2.5 BENCH_STEPS=10
log() { echo "$(date -Is) $*" >> $O/lane.log; }
log "START commit $(cat GIT_COMMIT 2>/dev/null || git rev-parse HEAD 2>/dev/null)"
bench() {   # gpu ckpt outdir seed extra...
  local g=$1 ck=$2 P=$3 sd=$4; shift 4
  [ -f $(dirname $P)/benchmark_$(basename $P).json ] && return 0
  PY=$PY GPUS=$g INFER_EXTRA="--save50 --seed-mode per_clip $*" bash scripts/echog_g1/bench.sh scmg $ck $DEC $P offline $sd $A/dev_stems.txt \
    >> $O/bench.log 2>&1 || log "BENCH_FAILED $P"
}
infer() {   # gpu ckpt outdir stems seed extra...
  local g=$1 ck=$2 P=$3 st=$4 sd=$5; shift 5
  CUDA_VISIBLE_DEVICES=$g $PY scripts/echog_g1/infer_test.py --ckpt $ck --vae $DEC --stats $STATS --view $HV --text-cache $TEXT_TEST \
    --out $P --stems $st --seed $sd --seed-mode per_clip --steps 10 --cfg-w 2.5 --split val --save50 --protocol offline "$@" >> $O/infer.log 2>&1 \
    || log "INFER_FAILED $P"
}
mkdir -p $O/ctlpt
[ -f $O/ctlpt/train/DONE.json ] || { CUDA_VISIBLE_DEVICES=0 SCMG_TEXT_CAUSAL=$TEXT_CAUSAL_TEST $PY -m scmg.train --out $O/ctlpt/train --init $INIT \
    --view $HV --stats $STATS --text-cache $TEXT_TEST --vae $DEC --workers 8 --batch 64 --stage base --steps 2000 --lr 1e-4 --warmup 200 \
    --eval-every 1000 --snap-every 1000000 --seed 0 --ctl-tokens $CTL0 --ctl-focus 0.5 --cfg '{"adsc": true, "ctl": true, "ctl_flow": true}' \
    --sc '{"p_stream": 0.67, "p_async": 1.0, "p_async_seg": 0.0, "self_force_p": 0, "ctl_w": 4.0, "w_ctl_aux": 3.0, "ctl_field_drop": 0.2, "ctl_cf_p": 0.5, "ctl_lr_scale": 10.0}' \
    > $O/ctlpt/train.log 2>&1 || { log "TRAIN_FAILED"; exit 1; }; }
log "TRAINED"
CK=$O/ctlpt/train/final.pt
for sd in 0 1 2; do (
  for m in none gt; do bench $sd $CK $O/ctlpt/$m/seed_00$sd $sd --ctl-tokens $CTL0 --ctl-mode $m & done
  [ -f $O/ctlpt/shuffle/seed_00$sd/DONE ] || { infer $sd $CK $O/ctlpt/shuffle/seed_00$sd $CTL0/dev_ann_stems.txt $sd --ctl-tokens $CTL0 --ctl-mode shuffle \
      && touch $O/ctlpt/shuffle/seed_00$sd/DONE; } &
  wait
  $PY scripts/semantic/g1_ctl_eval.py --ctl $CTL0 --view $HV --out $O/ctlpt/CTL_EVAL_s$sd.json gt=- none=$O/ctlpt/none/seed_00$sd \
    gt_run=$O/ctlpt/gt/seed_00$sd shuffle=$O/ctlpt/shuffle/seed_00$sd >> $O/ctl_eval.log 2>&1 || log "CTL_EVAL_FAILED s$sd"
  log "SEED_DONE s$sd" ) &
done; wait
log "LANE_DONE"
