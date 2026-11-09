#!/bin/bash
# 11-08 EX002 module 9b (criteria docs/criteria/EX002_CTL.md "Module 9b", pre-registered 12f8351): base enc-s0; control modules trained on enc-s0
# (lane-4 recipe, true freeze); fresh plans seed 1112; seeds 0-2; offline + psync; rows tok (5b + override) and m9b (IK on the generated stream,
# adaptive quintic ramps, same-side present target, collision-constrained IK with fallback grid). 4 GPUs, one pod.
#   ex002_ctl_m9b.sh <out dir>      env PY
set -uo pipefail
O=$1; CODE=$(cd "$(dirname "$0")/../.." && pwd); cd $CODE; mkdir -p $O/ctlenc
B=/mnt/cpfs/zbl-cpfs-new/CKPT/zehan/gesture-scmg; C=/mnt/cpfs/zbl-cpfs-new/dataset/zehan/gesture; X=$B/control-token-1105/ex002
PY=${PY:-python}; export SCMG_J=34 SCMG_NO_GLITCH=1 OMP_NUM_THREADS=${OMP_NUM_THREADS:-1} SCMG_TEXT_CAUSAL=$B/models/text-e5-causal-v36_1
VIEW=$C/train-v1-20261101/views/data-v36_1; ST=$C/scmg/STATS_v36_1.json; TX=$C/scmg/text-e5-v36_1; TOK=$C/scmg/tok-qwen25-v36_1; T=$X/tokens; HS=$T/HAND_SHAPES.json
VAE=$B/trained/vae-r3/final.pt; ENC=$B/ex002-sfscope-1108/enc-s0; FD=$X/PLAN_dev_m9b_dense.json; FS=$X/PLAN_dev_m9b_sparse.json; VA="--view $VIEW --stats $ST --text-cache $TX"
BL="scripts/ex002/ex002_local_blend.py --pre 0.3 --post 0.9"; FO="--finger-override $HS --fo-ramp 0.3 --fo-profile minjerk"; KF=scripts/ex002/ex002_keyframe.py
E="scripts/ex002/ex002_ctl_run.py eval $VA --ctl $T"; F=scripts/ex002/ex002_fgd_outside.py; XE=scripts/ex002/ex002_exec.py; AE=scripts/ex002/ex002_aim_error.py
BS=scripts/ex002/ex002_stg_boot.py; PRS=scripts/ex002/ex002_present.py; LV="scripts/video/static_share.py score --norm $T/NORM_DEV.json"; LA=scripts/ex002/lively_floor_ex002.py
log() { echo "$(date -Is) $*" >> $O/lane.log; }
log "START commit $(cat GIT_COMMIT 2>/dev/null || git rev-parse HEAD 2>/dev/null)"
[ -f $O/ctlenc/train/DONE.json ] || CUDA_VISIBLE_DEVICES=0 $PY -m scmg.train --stage base --view $VIEW --stats $ST --text-cache $TX --vae $VAE \
    --init $ENC/train/final.pt --out $O/ctlenc/train --steps 2000 --batch 64 --lr 1e-4 --warmup 200 --workers 8 --seed 0 --eval-every 1000 --snap-every 1000000 \
    --semsynth-share 0.6 --tok-cache $TOK --tok-intent $B/models/intent-qwen25-v36_1 --intent-frames $B/models/intent-frames-v36_1 \
    --holdout $B/semantic-recognizer-v1/manifest-n200/MANIFEST.json --ctl-tokens $T --ctl-focus 0.5 --train-only ctl_ \
    --cfg '{"adsc": true, "intent_aux": true, "ctl": true, "ctl_flow": true}' \
    --sc '{"p_stream": 0.67, "p_async": 1.0, "p_async_seg": 0.0, "w_int": 0.3, "sem_w": 2.0, "self_force_p": 0, "self_force_scope": "encoders", "ctl_w": 1.0, "w_ctl_aux": 3.0, "ctl_field_drop": 0.2, "ctl_cf_p": 0.5, "ctl_lr_scale": 1.0, "w_ctl_finger": 1.0, "ctl_cf_finger_p": 0.5}' \
    > $O/ctlenc/train.log 2>&1 && log "TRAINED" || { log "TRAIN_FAILED"; exit 1; }
CK=$O/ctlenc/train/final.pt
gen() {  # gpu protocol plan mode seed outdir
  [ -f $6/DONE ] && return 0
  CUDA_VISIBLE_DEVICES=$1 $PY scripts/ex002/bench_ex002.py infer --ckpt $CK --protocol $2 --seed $5 --out $6 --plan $3 --mode $4 $([ $2 = psync ] && echo --ctl-lookahead 1.0) >> $O/infer_$1.log 2>&1 \
    && touch $6/DONE || log "INFER_FAILED $6"
}
i=0; for pr in offline psync; do for sd in 0 1 2; do for x in "$FD|none|none" "$FD|given|d_given" "$FS|given|s_given" "$FD|shuffle|d_shuffle"; do IFS='|' read -r pl m nm <<< "$x"; echo "$((i % 16))|$pr|$pl|$m|$sd|$O/$pr/gen/${nm}_s$sd"; i=$((i + 1)); done; done; done > $O/jobs.txt
for w in $(seq 0 15); do ( grep "^$w|" $O/jobs.txt | while IFS='|' read -r ww pr pl m sd od; do mkdir -p $(dirname $od); gen $((ww % 4)) $pr $pl $m $sd $od; done ) & done; wait $(jobs -p)
log "GENERATED"
for pr in offline psync; do G=$O/$pr/gen
  for sd in 0 1 2; do for x in "$FD|given|d_given" "$FS|given|s_given" "$FD|shuffle|d_shuffle"; do IFS='|' read -r pl m nm <<< "$x"
    $PY $BL --none $G/none_s$sd --given $G/${nm}_s$sd --plan $pl --out $O/$pr/tok5b/${nm}_s$sd --mode $m $FO > /dev/null
    $PY $BL --none $G/none_s$sd --given $G/${nm}_s$sd --plan $pl --out $O/$pr/tok5bplain/${nm}_s$sd --mode $m > /dev/null
    for k in 0 1 2 3; do $PY $KF insert --version 9b --given $O/$pr/tok5bplain/${nm}_s$sd --plan $pl --out $O/$pr/m9b/${nm}_s$sd --mode $m --shard $k/4 > /dev/null 2>&1 & done
  done; done
done; wait $(jobs -p)
log "INSERTED"
$PY - $O $ENC > $O/IDENTITY.txt 2>&1 <<'PYEOF'
import sys, numpy as np
from pathlib import Path
O, Z = Path(sys.argv[1]), Path(sys.argv[2])
for pr in ('offline', 'psync'):
    for s in range(3):
        a, b = O / pr / 'gen' / f'none_s{s}', Z / pr / f's{s}'
        d = [float(np.abs(np.load(f) - np.load(b / f.name)).max()) for f in sorted(a.glob('[0-9]*.npy')) if (b / f.name).exists()]
        print(f'{pr} seed {s}: control-model none vs enc-s0 ({b}) max|diff| over {len(d)} clips = {max(d) if d else float("nan"):.3e}')
PYEOF
for pr in offline psync; do R=$O/$pr; N=$R/gen
  CUDA_VISIBLE_DEVICES=0 $PY $E --plan $FD --out $R/EVAL_given.json $(for s in 0 1 2; do echo none_s$s=$N/none_s$s tok_s$s=$R/tok5b/d_given_s$s m9b_s$s=$R/m9b/d_given_s$s; done) >> $O/eval_$pr.log 2>&1 &
  CUDA_VISIBLE_DEVICES=1 $PY $E --plan $FD --shuffled-run --out $R/EVAL_shuffle.json $(for s in 0 1 2; do echo tok_s$s=$R/tok5b/d_shuffle_s$s m9b_s$s=$R/m9b/d_shuffle_s$s; done) >> $O/eval_$pr.log 2>&1 &
  CUDA_VISIBLE_DEVICES=2 $PY $F --plan $FS --radius 0,1.0 --out $R/FGDOUT_sparse.json $(for s in 0 1 2; do echo none_s$s=$N/none_s$s tok_s$s=$R/tok5b/s_given_s$s m9b_s$s=$R/m9b/s_given_s$s; done) >> $O/eval_$pr.log 2>&1 &
  CUDA_VISIBLE_DEVICES=3 $PY $AE --plan $FD --ctl $T --out $R/AIMERR.json $(for s in 0 1 2; do echo none_s$s=$N/none_s$s tok_s$s=$R/tok5b/d_given_s$s m9b_s$s=$R/m9b/d_given_s$s; done) >> $O/eval_$pr.log 2>&1 &
  CUDA_VISIBLE_DEVICES=0 $PY $BS --view $VIEW --plan $FD --ctl $T --out $R/BOOT_m9b.json $(for s in 0 1 2; do echo --pair s$s:$R/m9b/d_given_s$s:$N/none_s$s; done) >> $O/eval_$pr.log 2>&1 &
  CUDA_VISIBLE_DEVICES=1 $PY $BS --view $VIEW --plan $FD --ctl $T --out $R/BOOT_tok.json $(for s in 0 1 2; do echo --pair s$s:$R/tok5b/d_given_s$s:$N/none_s$s; done) >> $O/eval_$pr.log 2>&1 &
  $PY $PRS eval --plan $FD --out $R/PRESENT.json $(for s in 0 1 2; do echo none_s$s=$N/none_s$s tok_s$s=$R/tok5b/d_given_s$s m9b_s$s=$R/m9b/d_given_s$s; done) >> $O/eval_$pr.log 2>&1 &
  $PY $XE --view $VIEW --plan $FS --ramp 0.3 --out $R/EXEC_sparse.json $(for s in 0 1 2; do echo tok_s$s=$R/tok5b/s_given_s$s m9b_s$s=$R/m9b/s_given_s$s; done) >> $O/eval_$pr.log 2>&1 &
  $PY $KF release --plan $FD --gt-dir $N --out $R/RELEASE_dense.json $(for s in 0 1 2; do echo tok_s$s=$R/tok5b/d_given_s$s m9b_s$s=$R/m9b/d_given_s$s; done) >> $O/eval_$pr.log 2>&1 &
  $PY $KF release --plan $FS --gt-dir $N --out $R/RELEASE_sparse.json $(for s in 0 1 2; do echo tok_s$s=$R/tok5b/s_given_s$s m9b_s$s=$R/m9b/s_given_s$s; done) >> $O/eval_$pr.log 2>&1 &
  $PY $LV --out $R/LIVE_mine.json $(for s in 0 1 2; do echo none_s$s=$N/none_s$s tok_s$s=$R/tok5b/s_given_s$s m9b_s$s=$R/m9b/s_given_s$s m9bdense_s$s=$R/m9b/d_given_s$s; done) >> $O/eval_$pr.log 2>&1 &
  $PY $LA --out $R/LIVE_protocol.json GT=$O/offline/gen none=$N/none_s0,$N/none_s1,$N/none_s2 tok=$R/tok5b/s_given_s0,$R/tok5b/s_given_s1,$R/tok5b/s_given_s2 \
      m9b=$R/m9b/s_given_s0,$R/m9b/s_given_s1,$R/m9b/s_given_s2 m9bdense=$R/m9b/d_given_s0,$R/m9b/d_given_s1,$R/m9b/d_given_s2 >> $O/eval_$pr.log 2>&1 &
  wait $(jobs -p)
done
log "LANE_DONE"; exit 0
