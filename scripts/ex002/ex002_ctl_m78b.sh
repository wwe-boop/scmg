#!/bin/bash
# 11-08 EX002 modules 7b / 8b on sibag-s2 (criteria docs/criteria/EX002_CTL.md "Modules 7b / 8b", pre-registered cfc519d; base = sibag-s2
# after agent A's corrected liveliness selection). 4 GPUs, one pod. Fresh plans seed 1110; seeds 0-2; offline + psync.
#  7b: control modules retrained on sibag-s2 (lane-4 recipe, true freeze) + w_ctl_ray_az 6 / w_ctl_ray_el 6 / w_ctl_raise 2.5; none / given /
#      shuffle; 5b + minimum-jerk override; gates + aiming error + bootstrap + liveliness guard.
#  8b: lane-4 control model (ctl4, sibag-s2 true freeze) with ONE presentation token per count command + library wrist pose (PRESENT_WRIST)
#      + minimum-jerk override, blended with 5b; none = the 7b job's none runs (both are sibag-s2 bitwise without tokens).
#   ex002_ctl_m78b.sh <out dir>      env PY
set -uo pipefail
O=$1; CODE=$(cd "$(dirname "$0")/../.." && pwd); cd $CODE; mkdir -p $O/m7b $O/m8b
B=/mnt/cpfs/zbl-cpfs-new/CKPT/zehan/gesture-scmg; C=/mnt/cpfs/zbl-cpfs-new/dataset/zehan/gesture; X=$B/control-token-1105/ex002
PY=${PY:-python}; export SCMG_J=34 SCMG_NO_GLITCH=1 OMP_NUM_THREADS=${OMP_NUM_THREADS:-1} SCMG_TEXT_CAUSAL=$B/models/text-e5-causal-v36_1
VIEW=$C/train-v1-20261101/views/data-v36_1; ST=$C/scmg/STATS_v36_1.json; TX=$C/scmg/text-e5-v36_1; TOK=$C/scmg/tok-qwen25-v36_1
VAE=$B/trained/vae-r3/final.pt; CK0=$B/mainline-seeds-1104b/sibag-s2/final.pt; CK4=$X/lane4/ctl4/train/final.pt; T=$X/tokens; HS=$T/HAND_SHAPES.json
FD=$X/PLAN_dev_m78b_dense.json; FS=$X/PLAN_dev_m78b_sparse.json; VA="--view $VIEW --stats $ST --text-cache $TX"
BL="scripts/ex002/ex002_local_blend.py --pre 0.3 --post 0.9"; FO="--finger-override $HS --fo-ramp 0.3 --fo-profile minjerk"
E="scripts/ex002/ex002_ctl_run.py eval $VA --ctl $T"; F=scripts/ex002/ex002_fgd_outside.py; XE=scripts/ex002/ex002_exec.py
AE=scripts/ex002/ex002_aim_error.py; BS=scripts/ex002/ex002_stg_boot.py; PRS=scripts/ex002/ex002_present.py; LV="scripts/video/static_share.py score --norm $T/NORM_DEV.json"
log() { echo "$(date -Is) $*" >> $O/lane.log; }
log "START commit $(cat GIT_COMMIT 2>/dev/null || git rev-parse HEAD 2>/dev/null)"
$PY $PRS plan1 --plan $FD --present $T/PRESENT.json --out $O/m8b/PLAN_present1_dense.json >> $O/lane.log 2>&1
$PY $PRS plan1 --plan $FS --present $T/PRESENT.json --out $O/m8b/PLAN_present1_sparse.json >> $O/lane.log 2>&1
PDp=$O/m8b/PLAN_present1_dense.json; PSp=$O/m8b/PLAN_present1_sparse.json
( [ -f $O/m7b/train/DONE.json ] || CUDA_VISIBLE_DEVICES=0 $PY -m scmg.train --stage base --view $VIEW --stats $ST --text-cache $TX --vae $VAE \
    --init $CK0 --out $O/m7b/train --steps 2000 --batch 64 --lr 1e-4 --warmup 200 --workers 8 --seed 0 --eval-every 1000 --snap-every 1000000 \
    --semsynth-share 0.6 --tok-cache $TOK --tok-intent $B/models/intent-qwen25-v36_1 --intent-frames $B/models/intent-frames-v36_1 \
    --holdout $B/semantic-recognizer-v1/manifest-n200/MANIFEST.json --ctl-tokens $T --ctl-focus 0.5 --train-only ctl_ \
    --cfg '{"adsc": true, "intent_aux": true, "ctl": true, "ctl_flow": true}' \
    --sc '{"p_stream": 0.67, "p_async": 1.0, "p_async_seg": 0.0, "w_int": 0.3, "sem_w": 2.0, "self_force_p": 0, "ctl_w": 1.0, "w_ctl_aux": 3.0, "w_ctl_ray_az": 6.0, "w_ctl_ray_el": 6.0, "w_ctl_raise": 2.5, "ctl_field_drop": 0.2, "ctl_cf_p": 0.5, "ctl_lr_scale": 1.0, "w_ctl_finger": 1.0, "ctl_cf_finger_p": 0.5}' \
    > $O/m7b/train.log 2>&1 && log "M7B_TRAINED" || log "M7B_TRAIN_FAILED" ) &
run() {  # gpu ckpt protocol plan mode seed outdir
  [ -f $7/DONE ] && return 0
  if [ $3 = offline ]; then CUDA_VISIBLE_DEVICES=$1 $PY scripts/ex002/ex002_ctl_run.py infer $VA --tok-cache $TOK --vae $VAE --plan $4 --ckpt $2 --mode $5 --seed $6 --out $7 >> $O/infer_$1.log 2>&1
  else CUDA_VISIBLE_DEVICES=$1 $PY scripts/ex002/bench_ex002.py infer --ckpt $2 --protocol psync --seed $6 --out $7 --plan $4 --mode $5 --ctl-lookahead 1.0 >> $O/infer_$1.log 2>&1; fi \
    && touch $7/DONE || log "INFER_FAILED $7"
}
# 8b generation (ctl4) + the shared no-token runs (ctl4 none == sibag-s2) on GPUs 1-3 while 7b trains
i=0; J=()
for pr in offline psync; do for sd in 0 1 2; do J+=("$CK4|$pr|$FD|none|$sd|$O/$pr/none_s$sd" "$CK4|$pr|$PDp|given|$sd|$O/m8b/$pr/gen/d_given_s$sd" "$CK4|$pr|$PSp|given|$sd|$O/m8b/$pr/gen/s_given_s$sd"); done; done
for j in "${J[@]}"; do echo "$((i % 9))|$j"; i=$((i + 1)); done > $O/jobs_a.txt
for w in $(seq 0 8); do ( grep "^$w|" $O/jobs_a.txt | while IFS='|' read -r ww ck pr pl m sd od; do mkdir -p $(dirname $od); run $((1 + ww % 3)) $ck $pr $pl $m $sd $od; done ) & done
wait $(jobs -p)
log "A_GENERATED"
[ -f $O/m7b/train/DONE.json ] || { log "M7B_NO_CKPT"; exit 1; }
CK7=$O/m7b/train/final.pt; i=0; J=()
for pr in offline psync; do for sd in 0 1 2; do for m in given shuffle; do J+=("$pr|$FD|$m|$sd|$O/m7b/$pr/gen/d_${m}_s$sd" "$pr|$FS|$m|$sd|$O/m7b/$pr/gen/s_${m}_s$sd"); done; done; done
for j in "${J[@]}"; do echo "$((i % 12))|$j"; i=$((i + 1)); done > $O/jobs_b.txt
for w in $(seq 0 11); do ( grep "^$w|" $O/jobs_b.txt | while IFS='|' read -r ww pr pl m sd od; do mkdir -p $(dirname $od); run $((ww % 4)) $CK7 $pr $pl $m $sd $od; done ) & done
wait $(jobs -p)
log "B_GENERATED"
for pr in offline psync; do N=$O/$pr
  for s in 0 1 2; do
    for pl in d s; do PL=$FD; PP=$PDp; PN=dense; [ $pl = s ] && { PL=$FS; PP=$PSp; PN=sparse; }
      for m in given shuffle; do
        $PY $BL --none $N/none_s$s --given $O/m7b/$pr/gen/${pl}_${m}_s$s --plan $PL --out $O/m7b/$pr/$PN/${m}_s$s --mode $m > /dev/null &
        $PY $BL --none $N/none_s$s --given $O/m7b/$pr/gen/${pl}_${m}_s$s --plan $PL --out $O/m7b/$pr/$PN/fo_${m}_s$s --mode $m $FO > /dev/null &
      done
      $PY $BL --none $N/none_s$s --given $O/m8b/$pr/gen/${pl}_given_s$s --plan $PP --out $O/m8b/$pr/$PN/given_s$s --mode given $FO --wrist-pose $T/PRESENT_WRIST.json > /dev/null &
    done
  done; wait $(jobs -p)
done
log "BLENDED"
for pr in offline psync; do N=$O/$pr; G7=$O/m7b/$pr/gen; R7=$O/m7b/$pr; R8=$O/m8b/$pr
  CUDA_VISIBLE_DEVICES=0 $PY $E --plan $FD --out $R7/EVAL_given.json $(for s in 0 1 2; do echo none_s$s=$N/none_s$s raw_given_s$s=$G7/d_given_s$s m5b_given_s$s=$R7/dense/given_s$s m5bfo_given_s$s=$R7/dense/fo_given_s$s; done) >> $O/eval_$pr.log 2>&1 &
  CUDA_VISIBLE_DEVICES=1 $PY $E --plan $FD --shuffled-run --out $R7/EVAL_shuffle.json $(for s in 0 1 2; do echo m5b_shuffle_s$s=$R7/dense/shuffle_s$s m5bfo_shuffle_s$s=$R7/dense/fo_shuffle_s$s; done) >> $O/eval_$pr.log 2>&1 &
  CUDA_VISIBLE_DEVICES=2 $PY $F --plan $FS --radius 0,1.0 --out $R7/FGDOUT_sparse.json $(for s in 0 1 2; do echo none_s$s=$N/none_s$s m5bsp_given_s$s=$R7/sparse/given_s$s m5bspfo_given_s$s=$R7/sparse/fo_given_s$s; done) >> $O/eval_$pr.log 2>&1 &
  CUDA_VISIBLE_DEVICES=3 $PY $AE --plan $FD --ctl $T --out $R7/AIMERR.json $(for s in 0 1 2; do echo none_s$s=$N/none_s$s m5b_given_s$s=$R7/dense/given_s$s raw_given_s$s=$G7/d_given_s$s; done) >> $O/eval_$pr.log 2>&1 &
  CUDA_VISIBLE_DEVICES=0 $PY $BS --view $VIEW --plan $FD --ctl $T --out $R7/BOOT_m5b.json $(for s in 0 1 2; do echo --pair s$s:$R7/dense/given_s$s:$N/none_s$s; done) >> $O/eval_$pr.log 2>&1 &
  $PY $XE --view $VIEW --plan $FS --ramp 0.3 --out $R7/EXEC_sparse.json $(for s in 0 1 2; do echo m5bspfo_given_s$s=$R7/sparse/fo_given_s$s; done) >> $O/eval_$pr.log 2>&1 &
  $PY $LV --out $R7/LIVELINESS.json $(for s in 0 1 2; do echo none_s$s=$N/none_s$s m7b_sparse_fo_given_s$s=$R7/sparse/fo_given_s$s m7b_dense_fo_given_s$s=$R7/dense/fo_given_s$s; done) >> $O/eval_$pr.log 2>&1 &
  CUDA_VISIBLE_DEVICES=1 $PY $E --plan $FD --out $R8/EVAL_given.json $(for s in 0 1 2; do echo m8b_given_s$s=$R8/dense/given_s$s; done) >> $O/eval_$pr.log 2>&1 &
  $PY $PRS eval --plan $FD --out $R8/PRESENT.json $(for s in 0 1 2; do echo m8b_s$s=$R8/dense/given_s$s none_s$s=$N/none_s$s; done) >> $O/eval_$pr.log 2>&1 &
  CUDA_VISIBLE_DEVICES=2 $PY $F --plan $FS --radius 0,1.0 --out $R8/FGDOUT_sparse.json $(for s in 0 1 2; do echo none_s$s=$N/none_s$s m8bsp_given_s$s=$R8/sparse/given_s$s; done) >> $O/eval_$pr.log 2>&1 &
  $PY $XE --view $VIEW --plan $FS --ramp 0.3 --out $R8/EXEC_sparse.json $(for s in 0 1 2; do echo m8bsp_given_s$s=$R8/sparse/given_s$s; done) >> $O/eval_$pr.log 2>&1 &
  $PY $LV --out $R8/LIVELINESS.json $(for s in 0 1 2; do echo m8b_sparse_given_s$s=$R8/sparse/given_s$s m8b_dense_given_s$s=$R8/dense/given_s$s; done) >> $O/eval_$pr.log 2>&1 &
  wait $(jobs -p)
done
log "LANE_DONE"; exit 0
