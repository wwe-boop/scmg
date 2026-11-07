#!/bin/bash
# 11-08 EX002 modules 7 (aim direction) and 8 (present-count) on the causal-match port (criteria docs/criteria/EX002_CTL.md "Module 7",
# "Module 8", pre-registered before this job). 4 GPUs, one pod.
#  M7: control modules retrained on czm with w_ctl_ray_el 6 / w_ctl_raise 5 (true freeze, port recipe, 2000 steps); offline + psync
#      none / given / shuffle on the port fresh plans (seed 1109) x seeds 0-2; 5b; gates + aiming error.
#  M8: the PORT control model (port-czm/ctlz) with present plans (an AIM per hand toward tokens/PRESENT.json at every finger command);
#      offline + psync given on dense and sparse present plans x seeds 0-2; 5b + override ($OVR: fo = cos 0.15 s, mj = minjerk 0.3 s);
#      count hit, hand height, palm angle, quality gates (none runs = the port job's).
#   ex002_ctl_m78.sh <out dir> <fo|mj>      env PY
set -uo pipefail
O=$1; OVR=${2:-fo}; CODE=$(cd "$(dirname "$0")/../.." && pwd); cd $CODE; mkdir -p $O/m7 $O/m8
B=/mnt/cpfs/zbl-cpfs-new/CKPT/zehan/gesture-scmg; C=/mnt/cpfs/zbl-cpfs-new/dataset/zehan/gesture; X=$B/control-token-1105/ex002; PZ=$X/port-czm
PY=${PY:-python}; export SCMG_J=34 SCMG_NO_GLITCH=1 OMP_NUM_THREADS=${OMP_NUM_THREADS:-1} SCMG_TEXT_CAUSAL=$B/models/text-e5-causal-v36_1
VIEW=$C/train-v1-20261101/views/data-v36_1; ST=$C/scmg/STATS_v36_1.json; TX=$C/scmg/text-e5-v36_1; TOK=$C/scmg/tok-qwen25-v36_1
VAE=$B/trained/vae-r3/final.pt; Z=$B/ex002-sweep-1107/czm-s12000-lr1e-4-w0.1; HS=$X/tokens/HAND_SHAPES.json
FD=$X/PLAN_dev_port_dense.json; FS=$X/PLAN_dev_port_sparse.json; VA="--view $VIEW --stats $ST --text-cache $TX"
BL="scripts/ex002/ex002_local_blend.py --pre 0.3 --post 0.9"; FOARG="--finger-override $HS"; RAMP=0.15
[ $OVR = mj ] && { FOARG="--finger-override $HS --fo-ramp 0.3 --fo-profile minjerk"; RAMP=0.3; }
E="scripts/ex002/ex002_ctl_run.py eval $VA --ctl $X/tokens"; F=scripts/ex002/ex002_fgd_outside.py; XE=scripts/ex002/ex002_exec.py
AE=scripts/ex002/ex002_aim_error.py; BS=scripts/ex002/ex002_stg_boot.py; PRS=scripts/ex002/ex002_present.py
log() { echo "$(date -Is) $*" >> $O/lane.log; }
log "START commit $(cat GIT_COMMIT 2>/dev/null || git rev-parse HEAD 2>/dev/null) override $OVR"
$PY $PRS plan --plan $FD --present $X/tokens/PRESENT.json --out $O/m8/PLAN_present_dense.json >> $O/lane.log 2>&1
$PY $PRS plan --plan $FS --present $X/tokens/PRESENT.json --out $O/m8/PLAN_present_sparse.json >> $O/lane.log 2>&1
PDp=$O/m8/PLAN_present_dense.json; PSp=$O/m8/PLAN_present_sparse.json
[ -f $O/m7/train/DONE.json ] || CUDA_VISIBLE_DEVICES=0 $PY -m scmg.train --stage base --view $VIEW --stats $ST --text-cache $TX --vae $VAE \
    --init $Z/train/final.pt --out $O/m7/train --steps 2000 --batch 64 --lr 1e-4 --warmup 200 --workers 6 --seed 0 --eval-every 1000 --snap-every 1000000 \
    --semsynth-share 0.4 --tok-cache $TOK --tok-intent $B/models/intent-qwen25-v36_1 --intent-frames $B/models/intent-frames-v36_1 \
    --holdout $B/semantic-recognizer-v1/manifest-n200/MANIFEST.json --causal-dir $B/ex002-causal-1107/pre-train-slim2m --causal-spk-atol 0.001 \
    --causal-no-spk-check --ctl-tokens $X/tokens --ctl-focus 0.5 --train-only ctl_ \
    --cfg '{"adsc": true, "ctl": true, "ctl_flow": true}' \
    --sc '{"p_stream": 0.67, "p_async": 1.0, "p_async_seg": 0.0, "self_force_p": 0, "w_phys": 0.1, "phys_dyn_w": 0, "causal_front": true, "causal_spk": true, "causal_text": true, "ctl_w": 1.0, "w_ctl_aux": 3.0, "w_ctl_ray_el": 6.0, "w_ctl_raise": 5.0, "ctl_field_drop": 0.2, "ctl_cf_p": 0.5, "ctl_lr_scale": 1.0, "w_ctl_finger": 1.0, "ctl_cf_finger_p": 0.5}' \
    > $O/m7/train.log 2>&1 && log "M7_TRAINED" || { log "M7_TRAIN_FAILED"; exit 1; } &
run() {  # gpu ckpt protocol plan mode seed outdir
  [ -f $7/DONE ] && return 0
  if [ $3 = offline ]; then CUDA_VISIBLE_DEVICES=$1 $PY scripts/ex002/ex002_ctl_run.py infer $VA --tok-cache $TOK --vae $VAE --plan $4 --ckpt $2 --mode $5 --seed $6 --out $7 >> $O/infer_$1.log 2>&1
  else CUDA_VISIBLE_DEVICES=$1 $PY scripts/ex002/bench_ex002.py infer --ckpt $2 --protocol psync --seed $6 --out $7 --plan $4 --mode $5 --ctl-lookahead 1.0 >> $O/infer_$1.log 2>&1; fi \
    && touch $7/DONE || log "INFER_FAILED $7"
}
# M8 generation (port model) starts while M7 trains: GPUs 1-3
CKP=$PZ/ctlz/train/final.pt; i=0; J=()
for pr in offline psync; do for sd in 0 1 2; do J+=("$pr|$PDp|given|$sd|$O/m8/$pr/gen/d_given_s$sd" "$pr|$PSp|given|$sd|$O/m8/$pr/gen/s_given_s$sd"); done; done
for j in "${J[@]}"; do echo "$((i % 9))|$j"; i=$((i + 1)); done > $O/jobs_m8.txt
for w in $(seq 0 8); do ( grep "^$w|" $O/jobs_m8.txt | while IFS='|' read -r ww pr pl m sd od; do mkdir -p $(dirname $od); run $((1 + ww % 3)) $CKP $pr $pl $m $sd $od; done ) & done
wait $(jobs -p)
log "M8_GENERATED"
[ -f $O/m7/train/DONE.json ] || { log "M7_NO_CKPT"; exit 1; }
# M7 generation: all 4 GPUs
CK7=$O/m7/train/final.pt; i=0; J=()
for pr in offline psync; do for sd in 0 1 2; do J+=("$pr|$FD|none|$sd|$O/m7/$pr/gen/none_s$sd"); for m in given shuffle; do J+=("$pr|$FD|$m|$sd|$O/m7/$pr/gen/d_${m}_s$sd" "$pr|$FS|$m|$sd|$O/m7/$pr/gen/s_${m}_s$sd"); done; done; done
for j in "${J[@]}"; do echo "$((i % 12))|$j"; i=$((i + 1)); done > $O/jobs_m7.txt
for w in $(seq 0 11); do ( grep "^$w|" $O/jobs_m7.txt | while IFS='|' read -r ww pr pl m sd od; do mkdir -p $(dirname $od); run $((ww % 4)) $CK7 $pr $pl $m $sd $od; done ) & done
wait $(jobs -p)
log "M7_GENERATED"
for pr in offline psync; do
  # M8 blends: none from the port job; 5b on the present plan + override
  for s in 0 1 2; do for pl in d s; do PL=$PDp; PN=dense; [ $pl = s ] && { PL=$PSp; PN=sparse; }
    $PY $BL --none $PZ/$pr/gen/none_s$s --given $O/m8/$pr/gen/${pl}_given_s$s --plan $PL --out $O/m8/$pr/$PN/given_s$s --mode given $FOARG > /dev/null &
  done; done
  for s in 0 1 2; do for m in given shuffle; do for pl in d s; do PL=$FD; PN=dense; [ $pl = s ] && { PL=$FS; PN=sparse; }
    $PY $BL --none $O/m7/$pr/gen/none_s$s --given $O/m7/$pr/gen/${pl}_${m}_s$s --plan $PL --out $O/m7/$pr/$PN/${m}_s$s --mode $m > /dev/null &
    $PY $BL --none $O/m7/$pr/gen/none_s$s --given $O/m7/$pr/gen/${pl}_${m}_s$s --plan $PL --out $O/m7/$pr/$PN/fo_${m}_s$s --mode $m $FOARG > /dev/null &
  done; done; done
  wait $(jobs -p)
done
log "BLENDED"
for pr in offline psync; do G7=$O/m7/$pr/gen; R7=$O/m7/$pr; R8=$O/m8/$pr; GZ=$PZ/$pr/gen
  CUDA_VISIBLE_DEVICES=0 $PY $E --plan $FD --out $R7/EVAL_given.json $(for s in 0 1 2; do echo none_s$s=$G7/none_s$s raw_given_s$s=$G7/d_given_s$s m5b_given_s$s=$R7/dense/given_s$s m5bfo_given_s$s=$R7/dense/fo_given_s$s; done) >> $O/eval_$pr.log 2>&1 &
  CUDA_VISIBLE_DEVICES=1 $PY $E --plan $FD --shuffled-run --out $R7/EVAL_shuffle.json $(for s in 0 1 2; do echo m5b_shuffle_s$s=$R7/dense/shuffle_s$s m5bfo_shuffle_s$s=$R7/dense/fo_shuffle_s$s; done) >> $O/eval_$pr.log 2>&1 &
  CUDA_VISIBLE_DEVICES=2 $PY $F --plan $FS --radius 0,1.0 --out $R7/FGDOUT_sparse.json $(for s in 0 1 2; do echo none_s$s=$G7/none_s$s m5bsp_given_s$s=$R7/sparse/given_s$s m5bspfo_given_s$s=$R7/sparse/fo_given_s$s; done) >> $O/eval_$pr.log 2>&1 &
  CUDA_VISIBLE_DEVICES=3 $PY $AE --plan $FD --ctl $X/tokens --out $R7/AIMERR.json $(for s in 0 1 2; do echo none_s$s=$G7/none_s$s m5b_given_s$s=$R7/dense/given_s$s raw_given_s$s=$G7/d_given_s$s portm5b_given_s$s=$PZ/$pr/dense/given_s$s portnone_s$s=$GZ/none_s$s; done) >> $O/eval_$pr.log 2>&1 &
  CUDA_VISIBLE_DEVICES=0 $PY $BS --view $VIEW --plan $FD --ctl $X/tokens --out $R7/BOOT_m5b.json $(for s in 0 1 2; do echo --pair s$s:$R7/dense/given_s$s:$G7/none_s$s; done) >> $O/eval_$pr.log 2>&1 &
  $PY $XE --view $VIEW --plan $FS --out $R7/EXEC_sparse.json $(for s in 0 1 2; do echo m5bspfo_given_s$s=$R7/sparse/fo_given_s$s; done) >> $O/eval_$pr.log 2>&1 &
  # M8: finger hit on the ORIGINAL plans (evaluate ignores the extra aims only if scored with the original plan), presentation, quality
  CUDA_VISIBLE_DEVICES=1 $PY $E --plan $FD --out $R8/EVAL_given.json $(for s in 0 1 2; do echo m8_given_s$s=$R8/dense/given_s$s; done) >> $O/eval_$pr.log 2>&1 &
  $PY $PRS eval --plan $FD --out $R8/PRESENT.json $(for s in 0 1 2; do echo m8_s$s=$R8/dense/given_s$s portfo_s$s=$PZ/$pr/dense/fo_given_s$s portnone_s$s=$GZ/none_s$s; done) >> $O/eval_$pr.log 2>&1 &
  CUDA_VISIBLE_DEVICES=2 $PY $F --plan $FS --radius 0,1.0 --out $R8/FGDOUT_sparse.json $(for s in 0 1 2; do echo none_s$s=$GZ/none_s$s m8sp_given_s$s=$R8/sparse/given_s$s; done) >> $O/eval_$pr.log 2>&1 &
  $PY $XE --view $VIEW --plan $FS --ramp $RAMP --out $R8/EXEC_sparse.json $(for s in 0 1 2; do echo m8sp_given_s$s=$R8/sparse/given_s$s; done) >> $O/eval_$pr.log 2>&1 &
  wait $(jobs -p)
done
log "LANE_DONE"; exit 0
