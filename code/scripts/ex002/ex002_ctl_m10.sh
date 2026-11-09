#!/bin/bash
# 11-08 EX002 module 10 (criteria docs/criteria/EX002_CTL.md "Module 10", pre-registered 02045af): (a) override shapes from the train
# human library (rows tok10a), (b) ctl training 2000 -> 3000 quick check (raw STG gain, offline seed 0, dense), 4000 only if the gain
# rises >= 0.005, then regeneration and rows tok10b / tok10ab. Base enc-s0 + ctlenc, decoder r3, 5c, plans 1112. 4 GPUs, one pod.
#   ex002_ctl_m10.sh <out dir>      env PY
set -uo pipefail
O=$1; CODE=$(cd "$(dirname "$0")/../.." && pwd); cd $CODE; mkdir -p $O
B=/mnt/cpfs/zbl-cpfs-new/CKPT/zehan/gesture-scmg; C=/mnt/cpfs/zbl-cpfs-new/dataset/zehan/gesture; X=$B/control-token-1105/ex002
PY=${PY:-python}; export SCMG_J=34 SCMG_NO_GLITCH=1 OMP_NUM_THREADS=${OMP_NUM_THREADS:-1} SCMG_TEXT_CAUSAL=$B/models/text-e5-causal-v36_1
VIEW=$C/train-v1-20261101/views/data-v36_1; ST=$C/scmg/STATS_v36_1.json; TX=$C/scmg/text-e5-v36_1; TOK=$C/scmg/tok-qwen25-v36_1; T=$X/tokens
HS=$T/HAND_SHAPES.json; LIB=$T/HAND_LIBRARY_TRAIN.npz; VAE=$B/trained/vae-r3/final.pt; M=$X/m9b-enc; VA="--view $VIEW --stats $ST --text-cache $TX"
FD=$X/PLAN_dev_m9b_dense.json; FS=$X/PLAN_dev_m9b_sparse.json
BL="scripts/ex002/ex002_local_blend.py --pre 0.3 --post 0.9 --compose smooth"; FO="--finger-override $HS --fo-ramp 0.3 --fo-profile minjerk"
E="scripts/ex002/ex002_ctl_run.py eval $VA --ctl $T"; F=scripts/ex002/ex002_fgd_outside.py; XE=scripts/ex002/ex002_exec.py; BS=scripts/ex002/ex002_stg_boot.py
LV="scripts/video/static_share.py score --norm $T/NORM_DEV.json"; LA=scripts/ex002/lively_floor_ex002.py
log() { echo "$(date -Is) $*" >> $O/lane.log; }
log "START commit $(cat GIT_COMMIT 2>/dev/null || git rev-parse HEAD 2>/dev/null) library $(sha256sum $LIB | cut -c1-16)"
blend() {  # gendir_root(protocol gen dir) outroot seed  ; rows tok (template) and tokL (library) for the three plan runs
  local G=$1 R=$2 sd=$3
  for x in "$FD|given|d_given" "$FS|given|s_given" "$FD|shuffle|d_shuffle"; do IFS='|' read -r pl m nm <<< "$x"
    $PY $BL --none $M/${R##*/}/gen/none_s$sd --given $G/${nm}_s$sd --plan $pl --out $R/$4/${nm}_s$sd --mode $m $FO > /dev/null &
    [ -n "${5:-}" ] && $PY $BL --none $M/${R##*/}/gen/none_s$sd --given $G/${nm}_s$sd --plan $pl --out $R/$5/${nm}_s$sd --mode $m $FO --fo-library $LIB > /dev/null &
  done
}
evals() {  # protocol rows...
  local pr=$1; shift; local R=$O/$pr N=$M/$pr/gen; local tg=$(echo "$@" | tr " " "-")
  CUDA_VISIBLE_DEVICES=0 $PY $E --plan $FD --out $R/EVAL_given_$tg.json $(for s in 0 1 2; do echo none_s$s=$N/none_s$s tok5c_s$s=$X/m5c-enc/$pr/tok5c/d_given_s$s; for r in "$@"; do echo ${r}_s$s=$R/$r/d_given_s$s; done; done) >> $O/eval_$pr.log 2>&1 &
  CUDA_VISIBLE_DEVICES=1 $PY $E --plan $FD --shuffled-run --out $R/EVAL_shuffle_$tg.json $(for s in 0 1 2; do for r in "$@"; do echo ${r}_s$s=$R/$r/d_shuffle_s$s; done; done) >> $O/eval_$pr.log 2>&1 &
  CUDA_VISIBLE_DEVICES=2 $PY $F --plan $FS --radius 0,1.0 --out $R/FGDOUT_sparse_$tg.json $(for s in 0 1 2; do echo none_s$s=$N/none_s$s; for r in "$@"; do echo ${r}_s$s=$R/$r/s_given_s$s; done; done) >> $O/eval_$pr.log 2>&1 &
  for r in "$@"; do CUDA_VISIBLE_DEVICES=3 $PY $BS --view $VIEW --plan $FD --ctl $T --out $R/BOOT_$r.json $(for s in 0 1 2; do echo --pair s$s:$R/$r/d_given_s$s:$N/none_s$s; done) >> $O/eval_$pr.log 2>&1; done &
  $PY $XE --view $VIEW --plan $FS --ramp 0.3 --out $R/EXEC_sparse_$tg.json $(for s in 0 1 2; do for r in "$@"; do echo ${r}_s$s=$R/$r/s_given_s$s; done; done) >> $O/eval_$pr.log 2>&1 &
  $PY $LV --out $R/LIVE_mine_$tg.json $(for s in 0 1 2; do for r in "$@"; do echo ${r}_s$s=$R/$r/s_given_s$s; done; done) >> $O/eval_$pr.log 2>&1 &
  $PY $LA --out $R/LIVE_protocol_$tg.json GT=$M/offline/gen none=$N/none_s0,$N/none_s1,$N/none_s2 $(for r in "$@"; do echo $r=$R/$r/s_given_s0,$R/$r/s_given_s1,$R/$r/s_given_s2; done) >> $O/eval_$pr.log 2>&1 &
  wait $(jobs -p)
}
# (a) library override on the m9b-enc generations
$PY scripts/ex002/shape_check.py --plan $FD --library $LIB --shapes $HS --out $O/SHAPE_CHECK_dense.json >> $O/lane.log 2>&1
for pr in offline psync; do for sd in 0 1 2; do blend $M/$pr/gen $O/$pr $sd tok5c_ tok10a; done; done; wait $(jobs -p)
rm -rf $O/offline/tok5c_ $O/psync/tok5c_
log "A_BLENDED"
for pr in offline psync; do evals $pr tok10a; done
log "A_SCORED"
# (b) quick check: ctl training 2000 -> 3000, raw STG gain offline seed 0 dense
train() {  # init out steps
  [ -f $2/DONE.json ] && return 0
  CUDA_VISIBLE_DEVICES=0 $PY -m scmg.train --stage base --view $VIEW --stats $ST --text-cache $TX --vae $VAE \
    --init $1 --out $2 --steps $3 --batch 64 --lr 1e-4 --warmup 200 --workers 8 --seed 0 --eval-every 1000 --snap-every 1000000 \
    --semsynth-share 0.6 --tok-cache $TOK --tok-intent $B/models/intent-qwen25-v36_1 --intent-frames $B/models/intent-frames-v36_1 \
    --holdout $B/semantic-recognizer-v1/manifest-n200/MANIFEST.json --ctl-tokens $T --ctl-focus 0.5 --train-only ctl_ \
    --cfg '{"adsc": true, "intent_aux": true, "ctl": true, "ctl_flow": true}' \
    --sc '{"p_stream": 0.67, "p_async": 1.0, "p_async_seg": 0.0, "w_int": 0.3, "sem_w": 2.0, "self_force_p": 0, "self_force_scope": "encoders", "ctl_w": 1.0, "w_ctl_aux": 3.0, "ctl_field_drop": 0.2, "ctl_cf_p": 0.5, "ctl_lr_scale": 1.0, "w_ctl_finger": 1.0, "ctl_cf_finger_p": 0.5}' \
    > $2.log 2>&1
}
gen_run() {  # ckpt protocol plan mode seed out
  for k in 0 1 2 3; do CUDA_VISIBLE_DEVICES=$k $PY scripts/ex002/bench_ex002.py infer --ckpt $1 --protocol $2 --seed $5 --out $6 --plan $3 --mode $4 --shard $k/4 \
    $([ $2 = psync ] && echo --ctl-lookahead 1.0) >> $O/infer_$k.log 2>&1 & done; wait $(jobs -p)
}
train $M/ctlenc/train/final.pt $O/b3k 1000 && log "B_TRAINED_3000" || { log "B_TRAIN_FAILED"; log "LANE_DONE"; exit 0; }
gen_run $O/b3k/final.pt offline $FD given 0 $O/b3k_raw/d_given_s0; log "B_GEN_3000"
$PY $BS --view $VIEW --plan $FD --ctl $T --out $O/BOOT_raw_2000.json --pair s0:$M/offline/gen/d_given_s0:$M/offline/gen/none_s0 >> $O/eval_b.log 2>&1 &
$PY $BS --view $VIEW --plan $FD --ctl $T --out $O/BOOT_raw_3000.json --pair s0:$O/b3k_raw/d_given_s0:$M/offline/gen/none_s0 >> $O/eval_b.log 2>&1 &
wait $(jobs -p)
G2=$($PY -c "import json; print(json.load(open('$O/BOOT_raw_2000.json'))['gain_forearm'])"); G3=$($PY -c "import json; print(json.load(open('$O/BOOT_raw_3000.json'))['gain_forearm'])")
log "B_QUICK raw STG gain 2000 $G2 3000 $G3"
if ! $PY -c "import sys; sys.exit(0 if $G3 - $G2 >= 0.005 else 1)"; then log "B_STOPPED (gain rise < 0.005)"; log "LANE_DONE"; exit 0; fi
train $O/b3k/final.pt $O/b4k 1000 && log "B_TRAINED_4000" || { log "B_TRAIN_FAILED"; log "LANE_DONE"; exit 0; }
for pr in offline psync; do for sd in 0 1 2; do for x in "$FD|given|d_given" "$FS|given|s_given" "$FD|shuffle|d_shuffle"; do IFS='|' read -r pl m nm <<< "$x"
  gen_run $O/b4k/final.pt $pr $pl $m $sd $O/$pr/gen4k/${nm}_s$sd
done; done; done
log "B_GENERATED"
for pr in offline psync; do for sd in 0 1 2; do blend $O/$pr/gen4k $O/$pr $sd tok10b tok10ab; done; done; wait $(jobs -p)
for pr in offline psync; do evals $pr tok10b tok10ab; done
log "LANE_DONE"; exit 0
