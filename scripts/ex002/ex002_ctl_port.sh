#!/bin/bash
# 11-08 EX002 final control configuration ported to the causal-match base czm-s12000-lr1e-4-w0.1 (criteria docs/criteria/EX002_CTL.md
# "Port to the causal-match base", frozen before this job). 4 GPUs, one pod: control modules trained with the true freeze on the czm
# recipe (2000 steps); fresh plans (seed 1109); offline AND psync generations none / given / shuffle x seeds 0-2 (30 runs, 3 streams
# per GPU); 5b blend + finger override (frozen cos 0.15 s, and the minimum-jerk 0.3 s comparison row); gates per protocol; bootstrap.
#   ex002_ctl_port.sh <out dir>      env PY
set -uo pipefail
O=$1; CODE=$(cd "$(dirname "$0")/../.." && pwd); cd $CODE; mkdir -p $O/ctlz
B=/mnt/cpfs/zbl-cpfs-new/CKPT/zehan/gesture-scmg; C=/mnt/cpfs/zbl-cpfs-new/dataset/zehan/gesture; X=$B/control-token-1105/ex002
PY=${PY:-python}; export SCMG_J=34 SCMG_NO_GLITCH=1 OMP_NUM_THREADS=${OMP_NUM_THREADS:-2} SCMG_TEXT_CAUSAL=$B/models/text-e5-causal-v36_1
VIEW=$C/train-v1-20261101/views/data-v36_1; ST=$C/scmg/STATS_v36_1.json; TX=$C/scmg/text-e5-v36_1; TOK=$C/scmg/tok-qwen25-v36_1
VAE=$B/trained/vae-r3/final.pt; Z=$B/ex002-sweep-1107/czm-s12000-lr1e-4-w0.1; CK0=$Z/train/final.pt; HS=$X/tokens/HAND_SHAPES.json
FD=$X/PLAN_dev_port_dense.json; FS=$X/PLAN_dev_port_sparse.json
VA="--view $VIEW --stats $ST --text-cache $TX"; BL="scripts/ex002/ex002_local_blend.py --pre 0.3 --post 0.9"
E="scripts/ex002/ex002_ctl_run.py eval $VA --ctl $X/tokens"; F=scripts/ex002/ex002_fgd_outside.py; XE=scripts/ex002/ex002_exec.py; BS=scripts/ex002/ex002_stg_boot.py
log() { echo "$(date -Is) $*" >> $O/lane.log; }
log "START commit $(cat GIT_COMMIT 2>/dev/null || git rev-parse HEAD 2>/dev/null)"
[ -f $O/ctlz/train/DONE.json ] || CUDA_VISIBLE_DEVICES=0 $PY -m scmg.train --stage base --view $VIEW --stats $ST --text-cache $TX --vae $VAE \
    --init $CK0 --out $O/ctlz/train --steps 2000 --batch 64 --lr 1e-4 --warmup 200 --workers 6 --seed 0 --eval-every 1000 --snap-every 1000000 \
    --semsynth-share 0.4 --tok-cache $TOK --tok-intent $B/models/intent-qwen25-v36_1 --intent-frames $B/models/intent-frames-v36_1 \
    --holdout $B/semantic-recognizer-v1/manifest-n200/MANIFEST.json --causal-dir $B/ex002-causal-1107/pre-train-slim2m --causal-spk-atol 0.001 \
    --causal-no-spk-check --ctl-tokens $X/tokens --ctl-focus 0.5 --train-only ctl_ \
    --cfg '{"adsc": true, "ctl": true, "ctl_flow": true}' \
    --sc '{"p_stream": 0.67, "p_async": 1.0, "p_async_seg": 0.0, "self_force_p": 0, "w_phys": 0.1, "phys_dyn_w": 0, "causal_front": true, "causal_spk": true, "causal_text": true, "ctl_w": 1.0, "w_ctl_aux": 3.0, "ctl_field_drop": 0.2, "ctl_cf_p": 0.5, "ctl_lr_scale": 1.0, "w_ctl_finger": 1.0, "ctl_cf_finger_p": 0.5}' \
    > $O/ctlz/train.log 2>&1 && log "TRAINED" || { log "TRAIN_FAILED"; exit 1; }
CK=$O/ctlz/train/final.pt
run() {  # gpu protocol plan mode seed outdir
  [ -f $6/DONE ] && return 0
  if [ $2 = offline ]; then
    CUDA_VISIBLE_DEVICES=$1 $PY scripts/ex002/ex002_ctl_run.py infer $VA --tok-cache $TOK --vae $VAE --plan $3 --ckpt $CK --mode $4 --seed $5 --out $6 >> $O/infer_$1.log 2>&1
  else
    CUDA_VISIBLE_DEVICES=$1 $PY scripts/ex002/bench_ex002.py infer --ckpt $CK --protocol psync --seed $5 --out $6 --plan $3 --mode $4 --ctl-lookahead 1.0 >> $O/infer_$1.log 2>&1
  fi && touch $6/DONE || log "INFER_FAILED $6"
}
i=0; J=()
for pr in offline psync; do for sd in 0 1 2; do
  J+=("$pr|$FD|none|$sd|$O/$pr/gen/none_s$sd"); for m in given shuffle; do J+=("$pr|$FD|$m|$sd|$O/$pr/gen/d_${m}_s$sd" "$pr|$FS|$m|$sd|$O/$pr/gen/s_${m}_s$sd"); done
done; done
for j in "${J[@]}"; do echo "$((i % 12))|$j"; i=$((i + 1)); done > $O/jobs.txt
for w in $(seq 0 11); do ( grep "^$w|" $O/jobs.txt | while IFS='|' read -r ww pr pl m sd od; do mkdir -p $(dirname $od); run $((ww % 4)) $pr $pl $m $sd $od; done ) & done
wait $(jobs -p)
log "GENERATED"
for pr in offline psync; do G=$O/$pr/gen; R=$O/$pr
  for s in 0 1 2; do for m in given shuffle; do for pl in d s; do PL=$FD; PN=dense; [ $pl = s ] && { PL=$FS; PN=sparse; }
    ( $PY $BL --none $G/none_s$s --given $G/${pl}_${m}_s$s --plan $PL --out $R/$PN/${m}_s$s --mode $m > /dev/null
      $PY $BL --none $G/none_s$s --given $G/${pl}_${m}_s$s --plan $PL --out $R/$PN/fo_${m}_s$s --mode $m --finger-override $HS > /dev/null
      $PY $BL --none $G/none_s$s --given $G/${pl}_${m}_s$s --plan $PL --out $R/$PN/mj_${m}_s$s --mode $m --finger-override $HS --fo-ramp 0.3 --fo-profile minjerk > /dev/null ) &
  done; done; done; wait $(jobs -p)
done
log "BLENDED"
for pr in offline psync; do G=$O/$pr/gen; R=$O/$pr
  CUDA_VISIBLE_DEVICES=0 $PY $E --plan $FD --out $R/EVAL_given.json $(for s in 0 1 2; do echo none_s$s=$G/none_s$s raw_given_s$s=$G/d_given_s$s m5b_given_s$s=$R/dense/given_s$s m5bfo_given_s$s=$R/dense/fo_given_s$s m5bmj_given_s$s=$R/dense/mj_given_s$s; done) >> $O/eval_$pr.log 2>&1 &
  CUDA_VISIBLE_DEVICES=1 $PY $E --plan $FD --shuffled-run --out $R/EVAL_shuffle.json $(for s in 0 1 2; do echo raw_shuffle_s$s=$G/d_shuffle_s$s m5b_shuffle_s$s=$R/dense/shuffle_s$s m5bfo_shuffle_s$s=$R/dense/fo_shuffle_s$s m5bmj_shuffle_s$s=$R/dense/mj_shuffle_s$s; done) >> $O/eval_$pr.log 2>&1 &
  CUDA_VISIBLE_DEVICES=2 $PY $E --plan $FS --out $R/EVAL_sp_given.json $(for s in 0 1 2; do echo none_s$s=$G/none_s$s m5bsp_given_s$s=$R/sparse/given_s$s m5bspfo_given_s$s=$R/sparse/fo_given_s$s m5bspmj_given_s$s=$R/sparse/mj_given_s$s; done) >> $O/eval_$pr.log 2>&1 &
  CUDA_VISIBLE_DEVICES=3 $PY $F --plan $FS --radius 0,1.0 --out $R/FGDOUT_sparse.json $(for s in 0 1 2; do echo none_s$s=$G/none_s$s raw_sp_given_s$s=$G/s_given_s$s m5bsp_given_s$s=$R/sparse/given_s$s m5bspfo_given_s$s=$R/sparse/fo_given_s$s m5bspmj_given_s$s=$R/sparse/mj_given_s$s; done) >> $O/eval_$pr.log 2>&1 &
  $PY $XE --view $VIEW --plan $FS --out $R/EXEC_sparse_fo.json $(for s in 0 1 2; do echo none_s$s=$G/none_s$s m5bsp_given_s$s=$R/sparse/given_s$s m5bspfo_given_s$s=$R/sparse/fo_given_s$s; done) >> $O/eval_$pr.log 2>&1 &
  $PY $XE --view $VIEW --plan $FS --ramp 0.3 --out $R/EXEC_sparse_mj.json $(for s in 0 1 2; do echo none_s$s=$G/none_s$s m5bsp_given_s$s=$R/sparse/given_s$s m5bspmj_given_s$s=$R/sparse/mj_given_s$s; done) >> $O/eval_$pr.log 2>&1 &
  CUDA_VISIBLE_DEVICES=0 $PY $BS --view $VIEW --plan $FD --ctl $X/tokens --out $R/BOOT_m5b.json $(for s in 0 1 2; do echo --pair s$s:$R/dense/given_s$s:$G/none_s$s; done) >> $O/eval_$pr.log 2>&1 &
  wait $(jobs -p)
done
$PY - $O $Z > $O/IDENTITY.txt 2>&1 <<'PYEOF'
import sys, numpy as np
from pathlib import Path
O, Z = Path(sys.argv[1]), Path(sys.argv[2])
for pr in ('offline', 'psync'):
    for s in range(3):
        a, b = O / pr / 'gen' / f'none_s{s}', Z / pr / f's{s}'
        d = [float(np.abs(np.load(f) - np.load(b / f.name)).max()) for f in sorted(a.glob('*.npy')) if (b / f.name).exists()]
        print(f'{pr} seed {s}: ctl-port none vs czm ({b}) max|diff| over {len(d)} clips = {max(d) if d else float("nan"):.3e}')
PYEOF
log "LANE_DONE"; exit 0
