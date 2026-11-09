#!/bin/bash
# 11-08 EX002 module 9 (criteria docs/criteria/EX002_CTL.md "Module 9", pre-registered 786b000): keyframe IK insertion + causal
# re-conditioning on sibag-s2 (tokens: lane-4 ctl4). Fresh plans seed 1111; seeds 0-2; offline + psync. 4 GPUs, one pod.
#   rows: tok (5b + minimum-jerk override), ik (5b + IK insertion, no re-conditioning), m9 (re-conditioned by latent inpainting + IK)
#   ex002_ctl_m9.sh <out dir>      env PY
set -uo pipefail
O=$1; CODE=$(cd "$(dirname "$0")/../.." && pwd); cd $CODE; mkdir -p $O
B=/mnt/cpfs/zbl-cpfs-new/CKPT/zehan/gesture-scmg; C=/mnt/cpfs/zbl-cpfs-new/dataset/zehan/gesture; X=$B/control-token-1105/ex002
PY=${PY:-python}; export SCMG_J=34 SCMG_NO_GLITCH=1 OMP_NUM_THREADS=${OMP_NUM_THREADS:-1} SCMG_TEXT_CAUSAL=$B/models/text-e5-causal-v36_1
VIEW=$C/train-v1-20261101/views/data-v36_1; ST=$C/scmg/STATS_v36_1.json; TX=$C/scmg/text-e5-v36_1; T=$X/tokens; HS=$T/HAND_SHAPES.json
CK4=$X/lane4/ctl4/train/final.pt; NONE=$X/m78b-sibag                      # no-token runs (= sibag-s2, plan independent)
FD=$X/PLAN_dev_m9_dense.json; FS=$X/PLAN_dev_m9_sparse.json; VA="--view $VIEW --stats $ST --text-cache $TX"
BL="scripts/ex002/ex002_local_blend.py --pre 0.3 --post 0.9"; FO="--finger-override $HS --fo-ramp 0.3 --fo-profile minjerk"; KF=scripts/ex002/ex002_keyframe.py
E="scripts/ex002/ex002_ctl_run.py eval $VA --ctl $T"; F=scripts/ex002/ex002_fgd_outside.py; XE=scripts/ex002/ex002_exec.py; AE=scripts/ex002/ex002_aim_error.py
BS=scripts/ex002/ex002_stg_boot.py; PRS=scripts/ex002/ex002_present.py; LV="scripts/video/static_share.py score --norm $T/NORM_DEV.json"; LA=scripts/ex002/lively_floor_ex002.py
log() { echo "$(date -Is) $*" >> $O/lane.log; }
log "START commit $(cat GIT_COMMIT 2>/dev/null || git rev-parse HEAD 2>/dev/null)"
gen() {  # gpu protocol plan mode seed outdir [inpaint dir]
  [ -f $6/DONE ] && return 0
  CUDA_VISIBLE_DEVICES=$1 $PY scripts/ex002/bench_ex002.py infer --ckpt $CK4 --protocol $2 --seed $5 --out $6 --plan $3 --mode $4 $([ $2 = psync ] && echo --ctl-lookahead 1.0) ${7:+--inpaint-dir $7} >> $O/infer_$1.log 2>&1 \
    && touch $6/DONE || log "INFER_FAILED $6"
}
pool() {  # jobs file, streams
  for w in $(seq 0 $(($2 - 1))); do ( grep "^$w|" $1 | while IFS='|' read -r ww pr pl m sd od ip; do mkdir -p $(dirname $od); gen $((ww % 4)) $pr $pl $m $sd $od $ip; done ) & done; wait $(jobs -p)
}
runs() { for pr in offline psync; do for sd in 0 1 2; do for x in "$FD|given|d_given" "$FS|given|s_given" "$FD|shuffle|d_shuffle"; do IFS='|' read -r pl m nm <<< "$x"; echo "$pr|$pl|$m|$sd|$O/$pr/$1/${nm}_s$sd|${2:+$O/$pr/ik/${nm}_s$sd}"; done; done; done; }
i=0; runs tok "" | while read -r l; do echo "$((i % 16))|$l"; i=$((i + 1)); done > $O/jobs_tok.txt; pool $O/jobs_tok.txt 16
log "TOK_GENERATED"
blend_ik() {  # protocol src(gen dir name) dst(ik|m9) ; 5b (tok/ik rows only) then IK insertion, sharded
  local pr=$1 src=$2 dst=$3
  for sd in 0 1 2; do for x in "$FD|given|d_given" "$FS|given|s_given" "$FD|shuffle|d_shuffle"; do IFS='|' read -r pl m nm <<< "$x"
    if [ $dst = ik ]; then
      $PY $BL --none $NONE/$pr/none_s$sd --given $O/$pr/$src/${nm}_s$sd --plan $pl --out $O/$pr/tok5b/${nm}_s$sd --mode $m $FO > /dev/null
      $PY $BL --none $NONE/$pr/none_s$sd --given $O/$pr/$src/${nm}_s$sd --plan $pl --out $O/$pr/tok5bplain/${nm}_s$sd --mode $m > /dev/null
      IN=$O/$pr/tok5bplain/${nm}_s$sd
    else IN=$O/$pr/$src/${nm}_s$sd; fi
    for k in 0 1 2 3; do $PY $KF insert --given $IN --plan $pl --out $O/$pr/$dst/${nm}_s$sd --mode $m --shard $k/4 > /dev/null 2>&1 & done
  done; done
}
for pr in offline psync; do blend_ik $pr tok ik; done; wait $(jobs -p)
log "IK_INSERTED"
i=0; runs rc 1 | while read -r l; do echo "$((i % 16))|$l"; i=$((i + 1)); done > $O/jobs_rc.txt; pool $O/jobs_rc.txt 16
log "RC_GENERATED"
for pr in offline psync; do blend_ik $pr rc m9; done; wait $(jobs -p)
log "M9_INSERTED"
for pr in offline psync; do R=$O/$pr; N=$NONE/$pr
  rows() { for s in 0 1 2; do for r in "$@"; do echo ${r}_s$s=$R/${r%%:*}; done; done; }
  CUDA_VISIBLE_DEVICES=0 $PY $E --plan $FD --out $R/EVAL_given.json $(for s in 0 1 2; do echo none_s$s=$N/none_s$s tok_s$s=$R/tok5b/d_given_s$s ik_s$s=$R/ik/d_given_s$s m9_s$s=$R/m9/d_given_s$s; done) >> $O/eval_$pr.log 2>&1 &
  CUDA_VISIBLE_DEVICES=1 $PY $E --plan $FD --shuffled-run --out $R/EVAL_shuffle.json $(for s in 0 1 2; do echo tok_s$s=$R/tok5b/d_shuffle_s$s ik_s$s=$R/ik/d_shuffle_s$s m9_s$s=$R/m9/d_shuffle_s$s; done) >> $O/eval_$pr.log 2>&1 &
  CUDA_VISIBLE_DEVICES=2 $PY $F --plan $FS --radius 0,1.0 --out $R/FGDOUT_sparse.json $(for s in 0 1 2; do echo none_s$s=$N/none_s$s tok_s$s=$R/tok5b/s_given_s$s ik_s$s=$R/ik/s_given_s$s m9_s$s=$R/m9/s_given_s$s; done) >> $O/eval_$pr.log 2>&1 &
  CUDA_VISIBLE_DEVICES=3 $PY $AE --plan $FD --ctl $T --out $R/AIMERR.json $(for s in 0 1 2; do echo none_s$s=$N/none_s$s tok_s$s=$R/tok5b/d_given_s$s ik_s$s=$R/ik/d_given_s$s m9_s$s=$R/m9/d_given_s$s; done) >> $O/eval_$pr.log 2>&1 &
  CUDA_VISIBLE_DEVICES=0 $PY $BS --view $VIEW --plan $FD --ctl $T --out $R/BOOT_m9.json $(for s in 0 1 2; do echo --pair s$s:$R/m9/d_given_s$s:$N/none_s$s; done) >> $O/eval_$pr.log 2>&1 &
  CUDA_VISIBLE_DEVICES=1 $PY $BS --view $VIEW --plan $FD --ctl $T --out $R/BOOT_tok.json $(for s in 0 1 2; do echo --pair s$s:$R/tok5b/d_given_s$s:$N/none_s$s; done) >> $O/eval_$pr.log 2>&1 &
  $PY $PRS eval --plan $FD --out $R/PRESENT.json $(for s in 0 1 2; do echo none_s$s=$N/none_s$s tok_s$s=$R/tok5b/d_given_s$s ik_s$s=$R/ik/d_given_s$s m9_s$s=$R/m9/d_given_s$s; done) >> $O/eval_$pr.log 2>&1 &
  $PY $XE --view $VIEW --plan $FS --ramp 0.3 --out $R/EXEC_sparse.json $(for s in 0 1 2; do echo tok_s$s=$R/tok5b/s_given_s$s ik_s$s=$R/ik/s_given_s$s m9_s$s=$R/m9/s_given_s$s; done) >> $O/eval_$pr.log 2>&1 &
  $PY $KF release --plan $FD --gt-dir $NONE/offline --out $R/RELEASE_dense.json $(for s in 0 1 2; do echo tok_s$s=$R/tok5b/d_given_s$s ik_s$s=$R/ik/d_given_s$s m9_s$s=$R/m9/d_given_s$s; done) >> $O/eval_$pr.log 2>&1 &
  $PY $KF release --plan $FS --gt-dir $NONE/offline --out $R/RELEASE_sparse.json $(for s in 0 1 2; do echo tok_s$s=$R/tok5b/s_given_s$s ik_s$s=$R/ik/s_given_s$s m9_s$s=$R/m9/s_given_s$s; done) >> $O/eval_$pr.log 2>&1 &
  $PY $LV --out $R/LIVE_mine.json $(for s in 0 1 2; do echo tok_s$s=$R/tok5b/s_given_s$s m9_s$s=$R/m9/s_given_s$s m9dense_s$s=$R/m9/d_given_s$s; done) >> $O/eval_$pr.log 2>&1 &
  $PY $LA --out $R/LIVE_protocol.json GT=$NONE/offline tok=$R/tok5b/s_given_s0,$R/tok5b/s_given_s1,$R/tok5b/s_given_s2 m9=$R/m9/s_given_s0,$R/m9/s_given_s1,$R/m9/s_given_s2 \
      m9dense=$R/m9/d_given_s0,$R/m9/d_given_s1,$R/m9/d_given_s2 ik=$R/ik/s_given_s0,$R/ik/s_given_s1,$R/ik/s_given_s2 >> $O/eval_$pr.log 2>&1 &
  wait $(jobs -p)
done
log "LANE_DONE"; exit 0
