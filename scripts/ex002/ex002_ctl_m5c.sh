#!/bin/bash
# 11-08 EX002 module 5c (criteria docs/criteria/EX002_CTL.md "Module 5c", pre-registered 9e65fc6) + 9b on the 5c input: post-processing of
# the m9b-enc generations (enc-s0 + ctlenc, plans 1112, seeds 0-2, offline + psync) with the C1 window composition; rows tok5c (5c +
# override) and m9b5c (9b insertion on 5c plain); 5b rows of m9b-enc re-scored where needed (release max, accel, dense FGD).
# sibag-s2 check: m9-sibag tok with 5c (psync (ii) hypothesis) and release max for the m9-sibag rows. 4 GPUs, one pod.
#   ex002_ctl_m5c.sh <out dir>      env PY
set -uo pipefail
O=$1; CODE=$(cd "$(dirname "$0")/../.." && pwd); cd $CODE; mkdir -p $O
B=/mnt/cpfs/zbl-cpfs-new/CKPT/zehan/gesture-scmg; C=/mnt/cpfs/zbl-cpfs-new/dataset/zehan/gesture; X=$B/control-token-1105/ex002
PY=${PY:-python}; export SCMG_J=34 SCMG_NO_GLITCH=1 OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
VIEW=$C/train-v1-20261101/views/data-v36_1; ST=$C/scmg/STATS_v36_1.json; TX=$C/scmg/text-e5-v36_1; T=$X/tokens; HS=$T/HAND_SHAPES.json
M=$X/m9b-enc; S9=$X/m9-sibag; SN=$X/m78b-sibag; VA="--view $VIEW --stats $ST --text-cache $TX"
FD=$X/PLAN_dev_m9b_dense.json; FS=$X/PLAN_dev_m9b_sparse.json; GD=$X/PLAN_dev_m9_dense.json; GS=$X/PLAN_dev_m9_sparse.json
BL="scripts/ex002/ex002_local_blend.py --pre 0.3 --post 0.9 --compose smooth"; FO="--finger-override $HS --fo-ramp 0.3 --fo-profile minjerk"; KF=scripts/ex002/ex002_keyframe.py
E="scripts/ex002/ex002_ctl_run.py eval $VA --ctl $T"; F=scripts/ex002/ex002_fgd_outside.py; XE=scripts/ex002/ex002_exec.py; AE=scripts/ex002/ex002_aim_error.py
BS=scripts/ex002/ex002_stg_boot.py; PRS=scripts/ex002/ex002_present.py; LV="scripts/video/static_share.py score --norm $T/NORM_DEV.json"; LA=scripts/ex002/lively_floor_ex002.py
AC=scripts/ex002/diag/accel_stats.py
log() { echo "$(date -Is) $*" >> $O/lane.log; }
log "START commit $(cat GIT_COMMIT 2>/dev/null || git rev-parse HEAD 2>/dev/null)"
$PY scripts/ex002/test_blend5c.py > $O/TEST_5c.txt 2>&1; grep -q ALL_OK $O/TEST_5c.txt || { log "TEST_5c_FAILED"; exit 1; }
grep -q LANE_DONE $M/lane.log || { log "M9B_NOT_DONE"; exit 1; }
# 1. 5c blends (+ 9b insertion on 5c plain), enc-s0
for pr in offline psync; do G=$M/$pr/gen
  for sd in 0 1 2; do for x in "$FD|given|d_given" "$FS|given|s_given" "$FD|shuffle|d_shuffle"; do IFS='|' read -r pl m nm <<< "$x"
    $PY $BL --none $G/none_s$sd --given $G/${nm}_s$sd --plan $pl --out $O/$pr/tok5c/${nm}_s$sd --mode $m $FO > /dev/null
    $PY $BL --none $G/none_s$sd --given $G/${nm}_s$sd --plan $pl --out $O/$pr/tok5cplain/${nm}_s$sd --mode $m > /dev/null
    for k in 0 1 2 3; do $PY $KF insert --version 9b --given $O/$pr/tok5cplain/${nm}_s$sd --plan $pl --out $O/$pr/m9b5c/${nm}_s$sd --mode $m --shard $k/4 > /dev/null 2>&1 & done
  done; done
done
# sibag-s2 tok with 5c (both plans), for the (ii) hypothesis and the dense-plan FGD
for pr in offline psync; do for sd in 0 1 2; do for x in "$GD|given|d_given" "$GS|given|s_given"; do IFS='|' read -r pl m nm <<< "$x"
  $PY $BL --none $SN/$pr/none_s$sd --given $S9/$pr/tok/${nm}_s$sd --plan $pl --out $O/sibag/$pr/tok5c/${nm}_s$sd --mode $m $FO > /dev/null
done; done; done
wait $(jobs -p)
log "BLENDED"
$PY - $O $M $S9 > $O/IDENTITY.txt 2>&1 <<'PYEOF'
import sys, numpy as np
from pathlib import Path
O, M, S9 = (Path(x) for x in sys.argv[1:4])
def cmp(a, b):
    fs = sorted(a.glob('[0-9]*.npy')); eq = [np.array_equal(np.load(f), np.load(b / f.name)) for f in fs if (b / f.name).exists()]
    d = [float(np.abs(np.load(f) - np.load(b / f.name)).max()) for f in fs if (b / f.name).exists()]
    return f'{sum(eq)}/{len(eq)} bitwise, max|diff| {max(d) if d else float("nan"):.3e}'
for pr in ('offline', 'psync'):
    for s in range(3):
        print(f'{pr} s{s} sparse: tok5c vs tok5b {cmp(O/pr/"tok5c"/f"s_given_s{s}", M/pr/"tok5b"/f"s_given_s{s}")}; tok5cplain vs tok5bplain {cmp(O/pr/"tok5cplain"/f"s_given_s{s}", M/pr/"tok5bplain"/f"s_given_s{s}")}; m9b5c vs m9b {cmp(O/pr/"m9b5c"/f"s_given_s{s}", M/pr/"m9b"/f"s_given_s{s}")}')
        print(f'{pr} s{s} dense:  tok5c vs tok5b {cmp(O/pr/"tok5c"/f"d_given_s{s}", M/pr/"tok5b"/f"d_given_s{s}")}')
        print(f'sibag {pr} s{s} sparse: tok5c vs tok5b {cmp(O/"sibag"/pr/"tok5c"/f"s_given_s{s}", S9/pr/"tok5b"/f"s_given_s{s}")}')
PYEOF
# 2. gates and reports, enc-s0
for pr in offline psync; do R=$O/$pr; N=$M/$pr/gen; Q=$M/$pr
  CUDA_VISIBLE_DEVICES=0 $PY $E --plan $FD --out $R/EVAL_given.json $(for s in 0 1 2; do echo none_s$s=$N/none_s$s tok5b_s$s=$Q/tok5b/d_given_s$s tok5c_s$s=$R/tok5c/d_given_s$s m9b5c_s$s=$R/m9b5c/d_given_s$s; done) >> $O/eval_$pr.log 2>&1 &
  CUDA_VISIBLE_DEVICES=1 $PY $E --plan $FD --shuffled-run --out $R/EVAL_shuffle.json $(for s in 0 1 2; do echo tok5c_s$s=$R/tok5c/d_shuffle_s$s m9b5c_s$s=$R/m9b5c/d_shuffle_s$s; done) >> $O/eval_$pr.log 2>&1 &
  CUDA_VISIBLE_DEVICES=2 $PY $F --plan $FS --radius 0,1.0 --out $R/FGDOUT_sparse.json $(for s in 0 1 2; do echo none_s$s=$N/none_s$s tok5c_s$s=$R/tok5c/s_given_s$s tok5cplain_s$s=$R/tok5cplain/s_given_s$s m9b5c_s$s=$R/m9b5c/s_given_s$s; done) >> $O/eval_$pr.log 2>&1 &
  CUDA_VISIBLE_DEVICES=3 $PY $F --plan $FD --radius 0 --out $R/FGDOUT_dense.json $(for s in 0 1 2; do echo none_s$s=$N/none_s$s tok5b_s$s=$Q/tok5b/d_given_s$s tok5c_s$s=$R/tok5c/d_given_s$s m9b_s$s=$Q/m9b/d_given_s$s m9b5c_s$s=$R/m9b5c/d_given_s$s; done) >> $O/eval_$pr.log 2>&1 &
  CUDA_VISIBLE_DEVICES=0 $PY $AE --plan $FD --ctl $T --out $R/AIMERR.json $(for s in 0 1 2; do echo tok5c_s$s=$R/tok5c/d_given_s$s m9b5c_s$s=$R/m9b5c/d_given_s$s; done) >> $O/eval_$pr.log 2>&1 &
  CUDA_VISIBLE_DEVICES=1 $PY $BS --view $VIEW --plan $FD --ctl $T --out $R/BOOT_tok5c.json $(for s in 0 1 2; do echo --pair s$s:$R/tok5c/d_given_s$s:$N/none_s$s; done) >> $O/eval_$pr.log 2>&1 &
  CUDA_VISIBLE_DEVICES=2 $PY $BS --view $VIEW --plan $FD --ctl $T --out $R/BOOT_m9b5c.json $(for s in 0 1 2; do echo --pair s$s:$R/m9b5c/d_given_s$s:$N/none_s$s; done) >> $O/eval_$pr.log 2>&1 &
  $PY $PRS eval --plan $FD --out $R/PRESENT.json $(for s in 0 1 2; do echo tok5c_s$s=$R/tok5c/d_given_s$s m9b5c_s$s=$R/m9b5c/d_given_s$s; done) >> $O/eval_$pr.log 2>&1 &
  $PY $XE --view $VIEW --plan $FS --ramp 0.3 --out $R/EXEC_sparse.json $(for s in 0 1 2; do echo tok5c_s$s=$R/tok5c/s_given_s$s m9b5c_s$s=$R/m9b5c/s_given_s$s; done) >> $O/eval_$pr.log 2>&1 &
  $PY $XE --view $VIEW --plan $FD --ramp 0.3 --out $R/EXEC_dense.json $(for s in 0 1 2; do echo tok5b_s$s=$Q/tok5b/d_given_s$s tok5c_s$s=$R/tok5c/d_given_s$s m9b_s$s=$Q/m9b/d_given_s$s m9b5c_s$s=$R/m9b5c/d_given_s$s; done) >> $O/eval_$pr.log 2>&1 &
  $PY $KF release --plan $FD --gt-dir $N --out $R/RELEASE_dense.json $(for s in 0 1 2; do echo tok5b_s$s=$Q/tok5b/d_given_s$s m9b_s$s=$Q/m9b/d_given_s$s tok5c_s$s=$R/tok5c/d_given_s$s m9b5c_s$s=$R/m9b5c/d_given_s$s; done) >> $O/eval_$pr.log 2>&1 &
  $PY $KF release --plan $FS --gt-dir $N --out $R/RELEASE_sparse.json $(for s in 0 1 2; do echo tok5b_s$s=$Q/tok5b/s_given_s$s m9b_s$s=$Q/m9b/s_given_s$s tok5c_s$s=$R/tok5c/s_given_s$s m9b5c_s$s=$R/m9b5c/s_given_s$s; done) >> $O/eval_$pr.log 2>&1 &
  $PY $LV --out $R/LIVE_mine.json $(for s in 0 1 2; do echo tok5c_s$s=$R/tok5c/s_given_s$s m9b5c_s$s=$R/m9b5c/s_given_s$s m9b5cdense_s$s=$R/m9b5c/d_given_s$s; done) >> $O/eval_$pr.log 2>&1 &
  $PY $LA --out $R/LIVE_protocol.json GT=$M/offline/gen none=$N/none_s0,$N/none_s1,$N/none_s2 tok5c=$R/tok5c/s_given_s0,$R/tok5c/s_given_s1,$R/tok5c/s_given_s2 \
      m9b5c=$R/m9b5c/s_given_s0,$R/m9b5c/s_given_s1,$R/m9b5c/s_given_s2 m9b5cdense=$R/m9b5c/d_given_s0,$R/m9b5c/d_given_s1,$R/m9b5c/d_given_s2 >> $O/eval_$pr.log 2>&1 &
  for pn in d_given s_given; do
    $PY $AC --out $R/ACCEL_$pn.json GT=$M/offline/gen none=$N/none_s0,$N/none_s1,$N/none_s2 $(for r in tok5b m9b; do echo $r=$Q/$r/${pn}_s0,$Q/$r/${pn}_s1,$Q/$r/${pn}_s2; done) \
        $(for r in tok5c m9b5c; do echo $r=$R/$r/${pn}_s0,$R/$r/${pn}_s1,$R/$r/${pn}_s2; done) >> $O/eval_$pr.log 2>&1 &
  done
  wait $(jobs -p)
done
log "ENC_SCORED"
# 3. sibag-s2: (ii) with 5c (sparse; expected bitwise 5b), dense whole-clip FGD 5b vs 5c, release with max for the m9-sibag rows
for pr in offline psync; do R=$O/sibag/$pr; N=$SN/$pr; Q=$S9/$pr
  CUDA_VISIBLE_DEVICES=0 $PY $F --plan $GS --radius 0,1.0 --out $R/FGDOUT_sparse.json $(for s in 0 1 2; do echo none_s$s=$N/none_s$s tok5b_s$s=$Q/tok5b/s_given_s$s tok5c_s$s=$R/tok5c/s_given_s$s; done) >> $O/eval_sib_$pr.log 2>&1 &
  CUDA_VISIBLE_DEVICES=1 $PY $F --plan $GD --radius 0 --out $R/FGDOUT_dense.json $(for s in 0 1 2; do echo none_s$s=$N/none_s$s tok5b_s$s=$Q/tok5b/d_given_s$s tok5c_s$s=$R/tok5c/d_given_s$s; done) >> $O/eval_sib_$pr.log 2>&1 &
  $PY $KF release --plan $GD --gt-dir $N --out $R/RELEASE_dense.json $(for s in 0 1 2; do echo tok5b_s$s=$Q/tok5b/d_given_s$s ik_s$s=$Q/ik/d_given_s$s m9_s$s=$Q/m9/d_given_s$s tok5c_s$s=$R/tok5c/d_given_s$s; done) >> $O/eval_sib_$pr.log 2>&1 &
  $PY $KF release --plan $GS --gt-dir $N --out $R/RELEASE_sparse.json $(for s in 0 1 2; do echo tok5b_s$s=$Q/tok5b/s_given_s$s ik_s$s=$Q/ik/s_given_s$s m9_s$s=$Q/m9/s_given_s$s tok5c_s$s=$R/tok5c/s_given_s$s; done) >> $O/eval_sib_$pr.log 2>&1 &
  wait $(jobs -p)
done
log "LANE_DONE"; exit 0
