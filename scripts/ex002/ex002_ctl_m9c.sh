#!/bin/bash
# 11-08 EX002 module 9c (criteria docs/criteria/EX002_CTL.md "Module 9c"): post-processing of the m9b-enc generations (enc-s0 + ctlenc,
# decoder r3, plans 1112, seeds 0-2, offline + psync) on the 5c input: rows tok5c (5c + override, reference), m9c (9c insertion, point
# pool = replayed train pointing-hold wrist motion) and m9cctl (control: train non-pointing gesture windows). Gates as 9b. 4 GPUs, one pod.
#   9c' (11-08): POOL9C=<pool npz>, ROWS9C='m9c|point' (insert only the point row), LINK_CTL9C=<m9c-enc> (reuse its control row)
#   ex002_ctl_m9c.sh <out dir> [insert|eval|all]      env PY, NPROC (parallel insertion processes, default 38), NSH / ISH (insert
#   stage split over NSH CPU pods: this pod runs command lines i with i % NSH == ISH)
# insert = 5c blends + 9c insertion (CPU only: run on a CPU pod); eval = the gate evaluations (4 GPUs)
set -uo pipefail
O=$1; STAGE=${2:-all}; NP=${NPROC:-38}; NSH=${NSH:-1}; ISH=${ISH:-0}; CODE=$(cd "$(dirname "$0")/../.." && pwd); cd $CODE; mkdir -p $O
B=/mnt/cpfs/zbl-cpfs-new/CKPT/zehan/gesture-scmg; C=/mnt/cpfs/zbl-cpfs-new/dataset/zehan/gesture; X=$B/control-token-1105/ex002
PY=${PY:-python}; export SCMG_J=34 SCMG_NO_GLITCH=1 OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
VIEW=$C/train-v1-20261101/views/data-v36_1; ST=$C/scmg/STATS_v36_1.json; TX=$C/scmg/text-e5-v36_1; T=$X/tokens; HS=$T/HAND_SHAPES.json; POOL=${POOL9C:-$T/HOLD_POOL_TRAIN.npz}
M=$X/m9b-enc; VA="--view $VIEW --stats $ST --text-cache $TX"; FD=$X/PLAN_dev_m9b_dense.json; FS=$X/PLAN_dev_m9b_sparse.json
BL="scripts/ex002/ex002_local_blend.py --pre 0.3 --post 0.9 --compose smooth"; FO="--finger-override $HS --fo-ramp 0.3 --fo-profile minjerk"; KF=scripts/ex002/ex002_keyframe.py
E="scripts/ex002/ex002_ctl_run.py eval $VA --ctl $T"; F=scripts/ex002/ex002_fgd_outside.py; XE=scripts/ex002/ex002_exec.py; AE=scripts/ex002/ex002_aim_error.py
BS=scripts/ex002/ex002_stg_boot.py; PRS=scripts/ex002/ex002_present.py; LV="scripts/video/static_share.py score --norm $T/NORM_DEV.json"; LA=scripts/ex002/lively_floor_ex002.py
HR=scripts/ex002/ex002_hold_report.py
log() { echo "$(date -Is) $*" >> $O/lane.log; }
log "START $STAGE commit $(cat GIT_COMMIT 2>/dev/null || git rev-parse HEAD 2>/dev/null) pool $(sha256sum $POOL | cut -c1-16)"
$PY scripts/ex002/test_blend5c.py > $O/TEST_5c.txt 2>&1; grep -q ALL_OK $O/TEST_5c.txt || { log "TEST_5c_FAILED"; exit 1; }
grep -q LANE_DONE $M/lane.log || { log "M9B_NOT_DONE"; exit 1; }
if [ $STAGE != eval ]; then
if [ $ISH = 0 ] && [ ! -f $O/BLENDED ]; then for pr in offline psync; do G=$M/$pr/gen
  for sd in 0 1 2; do for x in "$FD|given|d_given" "$FS|given|s_given" "$FD|shuffle|d_shuffle"; do IFS='|' read -r pl m nm <<< "$x"
    $PY $BL --none $G/none_s$sd --given $G/${nm}_s$sd --plan $pl --out $O/$pr/tok5c/${nm}_s$sd --mode $m $FO > /dev/null &
    $PY $BL --none $G/none_s$sd --given $G/${nm}_s$sd --plan $pl --out $O/$pr/tok5cplain/${nm}_s$sd --mode $m > /dev/null &
  done; done
done; wait $(jobs -p)
touch $O/BLENDED; log "BLENDED"
fi
until [ -f $O/BLENDED ]; do sleep 30; done                      # pods ISH > 0 wait for pod 0
for pr in offline psync; do for sd in 0 1 2; do for x in "$FD|given|d_given" "$FS|given|s_given" "$FD|shuffle|d_shuffle"; do IFS='|' read -r pl m nm <<< "$x"
  for row in ${ROWS9C:-m9c|point m9cctl|control}; do IFS='|' read -r rn pool <<< "$row"; for k in 0 1 2 3; do
    echo "$PY $KF insert --version 9c --hold-pool $POOL --pool $pool --given $O/$pr/tok5cplain/${nm}_s$sd --plan $pl --out $O/$pr/$rn/${nm}_s$sd --mode $m --shard $k/4"
  done; done
done; done; done | awk -v n=$NSH -v i=$ISH '(NR - 1) % n == i' > $O/insert_cmds.$ISH.txt
xargs -P $NP -I{} bash -c '{} > /dev/null 2>&1 || echo "FAILED {}"' < $O/insert_cmds.$ISH.txt >> $O/lane.log
log "INSERTED $(ls $O/*/m9c*/*/KEYFRAMES.*.json 2>/dev/null | wc -l) keyframe files"
fi
[ $STAGE = insert ] && { log "STAGE_DONE insert $ISH/$NSH"; exit 0; }
if [ -n "${LINK_CTL9C:-}" ]; then for pr in offline psync; do ln -sfn $LINK_CTL9C/$pr/m9cctl $O/$pr/m9cctl; done; log "LINKED m9cctl from $LINK_CTL9C"; fi   # 9c': control row reused
for pr in offline psync; do R=$O/$pr; N=$M/$pr/gen
  CUDA_VISIBLE_DEVICES=0 $PY $E --plan $FD --out $R/EVAL_given.json $(for s in 0 1 2; do echo none_s$s=$N/none_s$s tok5c_s$s=$R/tok5c/d_given_s$s m9c_s$s=$R/m9c/d_given_s$s m9cctl_s$s=$R/m9cctl/d_given_s$s; done) >> $O/eval_$pr.log 2>&1 &
  CUDA_VISIBLE_DEVICES=1 $PY $E --plan $FD --shuffled-run --out $R/EVAL_shuffle.json $(for s in 0 1 2; do echo tok5c_s$s=$R/tok5c/d_shuffle_s$s m9c_s$s=$R/m9c/d_shuffle_s$s m9cctl_s$s=$R/m9cctl/d_shuffle_s$s; done) >> $O/eval_$pr.log 2>&1 &
  CUDA_VISIBLE_DEVICES=2 $PY $F --plan $FS --radius 0,1.0 --out $R/FGDOUT_sparse.json $(for s in 0 1 2; do echo none_s$s=$N/none_s$s tok5c_s$s=$R/tok5c/s_given_s$s m9c_s$s=$R/m9c/s_given_s$s m9cctl_s$s=$R/m9cctl/s_given_s$s; done) >> $O/eval_$pr.log 2>&1 &
  CUDA_VISIBLE_DEVICES=3 $PY $AE --plan $FD --ctl $T --out $R/AIMERR.json $(for s in 0 1 2; do echo none_s$s=$N/none_s$s tok5c_s$s=$R/tok5c/d_given_s$s m9c_s$s=$R/m9c/d_given_s$s m9cctl_s$s=$R/m9cctl/d_given_s$s; done) >> $O/eval_$pr.log 2>&1 &
  CUDA_VISIBLE_DEVICES=0 $PY $BS --view $VIEW --plan $FD --ctl $T --out $R/BOOT_tok5c.json $(for s in 0 1 2; do echo --pair s$s:$R/tok5c/d_given_s$s:$N/none_s$s; done) >> $O/eval_$pr.log 2>&1 &
  CUDA_VISIBLE_DEVICES=1 $PY $BS --view $VIEW --plan $FD --ctl $T --out $R/BOOT_m9c.json $(for s in 0 1 2; do echo --pair s$s:$R/m9c/d_given_s$s:$N/none_s$s; done) >> $O/eval_$pr.log 2>&1 &
  CUDA_VISIBLE_DEVICES=2 $PY $BS --view $VIEW --plan $FD --ctl $T --out $R/BOOT_m9cctl.json $(for s in 0 1 2; do echo --pair s$s:$R/m9cctl/d_given_s$s:$N/none_s$s; done) >> $O/eval_$pr.log 2>&1 &
  $PY $PRS eval --plan $FD --out $R/PRESENT.json $(for s in 0 1 2; do echo tok5c_s$s=$R/tok5c/d_given_s$s m9c_s$s=$R/m9c/d_given_s$s m9cctl_s$s=$R/m9cctl/d_given_s$s; done) >> $O/eval_$pr.log 2>&1 &
  $PY $XE --view $VIEW --plan $FS --ramp 0.3 --out $R/EXEC_sparse.json $(for s in 0 1 2; do echo tok5c_s$s=$R/tok5c/s_given_s$s m9c_s$s=$R/m9c/s_given_s$s m9cctl_s$s=$R/m9cctl/s_given_s$s; done) >> $O/eval_$pr.log 2>&1 &
  $PY $KF release --plan $FD --gt-dir $N --out $R/RELEASE_dense.json $(for s in 0 1 2; do echo tok5c_s$s=$R/tok5c/d_given_s$s m9c_s$s=$R/m9c/d_given_s$s m9cctl_s$s=$R/m9cctl/d_given_s$s; done) >> $O/eval_$pr.log 2>&1 &
  $PY $KF release --plan $FS --gt-dir $N --out $R/RELEASE_sparse.json $(for s in 0 1 2; do echo tok5c_s$s=$R/tok5c/s_given_s$s m9c_s$s=$R/m9c/s_given_s$s m9cctl_s$s=$R/m9cctl/s_given_s$s; done) >> $O/eval_$pr.log 2>&1 &
  $PY $LV --out $R/LIVE_mine.json $(for s in 0 1 2; do echo tok5c_s$s=$R/tok5c/s_given_s$s m9c_s$s=$R/m9c/s_given_s$s m9cctl_s$s=$R/m9cctl/s_given_s$s m9cdense_s$s=$R/m9c/d_given_s$s; done) >> $O/eval_$pr.log 2>&1 &
  $PY $LA --out $R/LIVE_protocol.json GT=$M/offline/gen none=$N/none_s0,$N/none_s1,$N/none_s2 tok5c=$R/tok5c/s_given_s0,$R/tok5c/s_given_s1,$R/tok5c/s_given_s2 \
      m9c=$R/m9c/s_given_s0,$R/m9c/s_given_s1,$R/m9c/s_given_s2 m9cctl=$R/m9cctl/s_given_s0,$R/m9cctl/s_given_s1,$R/m9cctl/s_given_s2 \
      m9cdense=$R/m9c/d_given_s0,$R/m9c/d_given_s1,$R/m9c/d_given_s2 >> $O/eval_$pr.log 2>&1 &
  $PY $HR --plan $FD --ctl $T --pool-json ${POOL%.npz}.json --out $R/HOLD_REPORT.json $(for s in 0 1 2; do echo m9c_s$s=$R/m9c/d_given_s$s m9cctl_s$s=$R/m9cctl/d_given_s$s; done) >> $O/eval_$pr.log 2>&1 &
  wait $(jobs -p)
done
log "LANE_DONE"; exit 0
