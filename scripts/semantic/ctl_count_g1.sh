#!/bin/bash
# 11-07 §7.55 item 3 (controllability) on SCMG, dev only (criteria: docs/criteria/SEMLINE_7_55.md, item 3 section, written before
# this run). Two models, inference seeds 0 / 1 / 2, --seed-mode per_clip, offline, CFG 2.5, 10 steps:
#   ctlcnt : NEW lane = the ctlcf recipe (adsc + ctl + ctl_flow, counterfactual pointing + ray loss) + count control (count slot of
#            ctl-tokens-hos-count, soft-count loss w_ctl_count 1, counterfactual counts 0.5), hos-base warm start, 2000 steps
#   ctlcf  : the existing 11-06 counterfactual-pointing model (4000 steps; its tokens have no counts), re-measured per_clip
# 4 GPUs: ctlcnt trains on GPU0 while ctlcf's seeds run on GPUs 1-3 (all modes of a seed concurrently), then ctlcnt's seeds on 1-3.
# Runs per model and seed: none / gt on the 400 dev clips (official benchmark: FGD / BA), shuffle (pointing targets deranged) on the
# annotation clips; ctlcnt also nocount / count_shuffle. Control metrics: g1_ctl_eval.py (STG, count hit, amplitude hit, action rate).
#   ctl_count_g1.sh <out dir>     env PY
set -uo pipefail
O=$1; CODE=$(cd "$(dirname "$0")/../.." && pwd); cd $CODE; mkdir -p $O
B=/mnt/cpfs/zbl-cpfs-new/CKPT/zehan/gesture-scmg; C=/mnt/cpfs/zbl-cpfs-new/dataset/zehan/gesture
DEC=$B/echog-g1-decft-1105/decft-w3d0m05/final.pt; A=$B/echog-g1-hos-assets; HV=$B/views/echog-g1-v1-hos
CTLC=$B/control-token-1105/ctl-tokens-hos-count; CTL0=$B/control-token-1105/ctl-tokens-hos
INIT=$B/echog-g1-arch-1104/hos-base/best.pt; CF=$B/control-token-1105/run1a/dev/ctlcf/final.pt
PY=${PY:-python}; export SCMG_J=39 SCMG_NO_GLITCH=1 OMP_NUM_THREADS=${OMP_NUM_THREADS:-2}
W=/mnt/cpfs/zbl-cpfs-new/USERS/zehan/baselines/echog-official; D4=$B/baseline-repro-1104/echog/data_root
export EG=$D4 OFF=$W/official MMAE=$D4/eval_assets/mmae/g1_mmae_30body_30fps.npy FGD_CKPT=$W/weights/evaluation/g1_fgd_encoder.pt
export REF_DIR=$A/motion_39d_30fps WAV_DIR=$A/audio SPLIT=$A/dev_stems.txt VIEW_TEST=$HV VSPLIT=val TEXT_TEST=$C/scmg/text-e5-echog-g1-v1 \
  TEXT_CAUSAL_TEST=$B/models/text-e5-causal-echog-g1-v1 STATS=$C/scmg/STATS_echog_g1_v1.json CFGW=2.5 BENCH_STEPS=10
log() { echo "$(date -Is) $*" >> $O/lane.log; }
THR=$($PY -c "import json; print(json.load(open('$CTLC/APEX.json'))['action_speed'])")
log "START commit $(cat GIT_COMMIT 2>/dev/null || git rev-parse HEAD 2>/dev/null) thr $THR"
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
train_cnt() {
  [ -f $O/ctlcnt/train/DONE.json ] && return 0
  CUDA_VISIBLE_DEVICES=0 SCMG_TEXT_CAUSAL=$B/models/text-e5-causal-echog-g1-v1 $PY -m scmg.train --out $O/ctlcnt/train --init $INIT --view $HV \
    --stats $STATS --text-cache $TEXT_TEST --vae $DEC --workers 8 --batch 64 --stage base --steps 2000 --lr 1e-4 --warmup 200 \
    --eval-every 1000 --snap-every 1000000 --seed 0 --ctl-tokens $CTLC --ctl-focus 0.5 --cfg '{"adsc": true, "ctl": true, "ctl_flow": true}' \
    --sc "{\"p_stream\": 0.67, \"p_async\": 1.0, \"p_async_seg\": 0.0, \"self_force_p\": 0, \"ctl_w\": 4.0, \"w_ctl_aux\": 3.0, \"ctl_field_drop\": 0.2, \"ctl_cf_p\": 0.5, \"ctl_lr_scale\": 10.0, \"w_ctl_count\": 1.0, \"ctl_cf_count_p\": 0.5, \"ctl_count_thr\": $THR}" \
    > $O/ctlcnt/train.log 2>&1 || { log "TRAIN_FAILED ctlcnt"; return 1; }
  log "TRAINED ctlcnt"
}
seed_runs() {    # gpu name ckpt ctl-dir seed extra-modes...   (all runs of one seed concurrently on one GPU)
  local g=$1 n=$2 ck=$3 cd=$4 sd=$5; shift 5
  for m in none gt; do bench $g $ck $O/$n/$m/seed_00$sd $sd --ctl-tokens $cd --ctl-mode $m & done
  for m in "$@"; do [ -f $O/$n/$m/seed_00$sd/DONE ] || { infer $g $ck $O/$n/$m/seed_00$sd $cd/dev_ann_stems.txt $sd --ctl-tokens $cd --ctl-mode $m \
      && touch $O/$n/$m/seed_00$sd/DONE; } & done
  wait; log "SEED_DONE $n s$sd"
}
mkdir -p $O/ctlcnt $O/ctlcf
for sd in 0 1 2; do seed_runs $((sd + 1)) ctlcf $CF $CTL0 $sd shuffle & done
train_cnt; wait
CK=$O/ctlcnt/train/final.pt
[ -f $CK ] && for sd in 0 1 2; do seed_runs $((sd + 1)) ctlcnt $CK $CTLC $sd shuffle nocount count_shuffle & done; wait
for n in ctlcf ctlcnt; do
  cd_=$CTL0; ms="none gt shuffle"; [ $n = ctlcnt ] && { cd_=$CTLC; ms="none gt shuffle nocount count_shuffle"; }
  for sd in 0 1 2; do sp=(gt=-); for m in $ms; do [ -d $O/$n/$m/seed_00$sd ] && sp+=($m=$O/$n/$m/seed_00$sd); done
    [ -f $O/$n/CTL_EVAL_s$sd.json ] || $PY scripts/semantic/g1_ctl_eval.py --ctl $cd_ --view $HV --out $O/$n/CTL_EVAL_s$sd.json "${sp[@]}" \
      >> $O/ctl_eval.log 2>&1 || log "CTL_EVAL_FAILED $n s$sd"
  done &
done; wait
log "LANE_DONE"
