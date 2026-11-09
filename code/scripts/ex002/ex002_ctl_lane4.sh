#!/bin/bash
# 11-07 EX002 controllability lane 4 (criteria: docs/criteria/EX002_CTL.md, "lane 4", gates committed before this run). 4 GPUs, one pod:
# ctl4 = sibag-s2 + control modules, TRUE freeze (--train-only ctl_, control modules lr 1e-4), 2000 steps; then dense plan
# none / given / shuffle + finger-override given / shuffle and sparse plan given / shuffle + finger-override given / shuffle,
# seeds 0-2 (27 runs, 2 streams per GPU); evaluation: ex002_ctl_run.py eval (dense + sparse), ex002_fgd_outside.py (gates i / ii and
# report rows, control re-scored), ex002_exec.py (executability after CBF, override ramp jerk), none-vs-control identity check.
#   ex002_ctl_lane4.sh <out dir>      env PY
set -uo pipefail
O=$1; CODE=$(cd "$(dirname "$0")/../.." && pwd); cd $CODE; mkdir -p $O/ctl4 $O/ctl4sp
B=/mnt/cpfs/zbl-cpfs-new/CKPT/zehan/gesture-scmg; C=/mnt/cpfs/zbl-cpfs-new/dataset/zehan/gesture; X=$B/control-token-1105/ex002
PY=${PY:-python}; export SCMG_J=34 SCMG_NO_GLITCH=1 OMP_NUM_THREADS=${OMP_NUM_THREADS:-2} SCMG_TEXT_CAUSAL=$B/models/text-e5-causal-v36_1
VIEW=$C/train-v1-20261101/views/data-v36_1; ST=$C/scmg/STATS_v36_1.json; TX=$C/scmg/text-e5-v36_1; TOK=$C/scmg/tok-qwen25-v36_1
VAE=$B/trained/vae-r3/final.pt; CK0=$B/mainline-seeds-1104b/sibag-s2/final.pt; HS=$X/tokens/HAND_SHAPES.json
PD=$X/PLAN_dev.json; PS=$X/PLAN_dev_sparse.json; CTRL=$X/lane1/ctrl
VA="--view $VIEW --stats $ST --text-cache $TX"; IA="--tok-cache $TOK --vae $VAE"
log() { echo "$(date -Is) $*" >> $O/lane.log; }
log "START commit $(cat GIT_COMMIT 2>/dev/null || git rev-parse HEAD 2>/dev/null)"
run() {  # gpu ckpt plan mode seed outdir [extra]
  [ -f $6/DONE ] && return 0
  CUDA_VISIBLE_DEVICES=$1 $PY scripts/ex002/ex002_ctl_run.py infer $VA $IA --plan $3 --ckpt $2 --mode $4 --seed $5 --out $6 ${7:-} >> $O/infer_$1.log 2>&1 \
    && touch $6/DONE || log "INFER_FAILED $6"
}
[ -f $O/ctl4/train/DONE.json ] || CUDA_VISIBLE_DEVICES=0 $PY -m scmg.train --stage base --view $VIEW --stats $ST --text-cache $TX --vae $VAE \
    --init $CK0 --out $O/ctl4/train --steps 2000 --batch 64 --lr 1e-4 --warmup 200 --workers 8 --seed 0 --eval-every 1000 --snap-every 1000000 \
    --semsynth-share 0.6 --tok-cache $TOK --tok-intent $B/models/intent-qwen25-v36_1 --intent-frames $B/models/intent-frames-v36_1 \
    --holdout $B/semantic-recognizer-v1/manifest-n200/MANIFEST.json --ctl-tokens $X/tokens --ctl-focus 0.5 --train-only ctl_ \
    --cfg '{"adsc": true, "intent_aux": true, "ctl": true, "ctl_flow": true}' \
    --sc '{"p_stream": 0.67, "p_async": 1.0, "p_async_seg": 0.0, "w_int": 0.3, "sem_w": 2.0, "self_force_p": 0, "ctl_w": 1.0, "w_ctl_aux": 3.0, "ctl_field_drop": 0.2, "ctl_cf_p": 0.5, "ctl_lr_scale": 1.0, "w_ctl_finger": 1.0, "ctl_cf_finger_p": 0.5}' \
    > $O/ctl4/train.log 2>&1 && log "TRAINED ctl4" || { log "TRAIN_FAILED ctl4"; exit 1; }
CK=$O/ctl4/train/final.pt
i=0; J=()
for sd in 0 1 2; do
  for m in none given shuffle; do J+=("$PD|$m|$sd|$O/ctl4/${m}_s$sd|"); done
  for m in given shuffle; do J+=("$PD|$m|$sd|$O/ctl4/fo_${m}_s$sd|--finger-override $HS"); done
  for m in given shuffle; do J+=("$PS|$m|$sd|$O/ctl4sp/${m}_s$sd|" "$PS|$m|$sd|$O/ctl4sp/fo_${m}_s$sd|--finger-override $HS"); done
done
for j in "${J[@]}"; do echo "$((i % 8))|$j"; i=$((i + 1)); done > $O/jobs.txt
for w in 0 1 2 3 4 5 6 7; do ( grep "^$w|" $O/jobs.txt | while IFS='|' read -r ww pl m sd od ex; do run $((ww % 4)) $CK $pl $m $sd $od "$ex"; done ) & done
wait $(jobs -p)
log "INFER_DONE"
E="scripts/ex002/ex002_ctl_run.py eval $VA --ctl $X/tokens"
CUDA_VISIBLE_DEVICES=0 $PY $E --plan $PD --out $O/EVAL_given.json $(for s in 0 1 2; do echo ctl4_none_s$s=$O/ctl4/none_s$s ctl4_given_s$s=$O/ctl4/given_s$s ctl4fo_given_s$s=$O/ctl4/fo_given_s$s; done) >> $O/eval.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 $PY $E --plan $PD --shuffled-run --out $O/EVAL_shuffle.json $(for s in 0 1 2; do echo ctl4_shuffle_s$s=$O/ctl4/shuffle_s$s ctl4fo_shuffle_s$s=$O/ctl4/fo_shuffle_s$s; done) >> $O/eval.log 2>&1 &
CUDA_VISIBLE_DEVICES=2 $PY $E --plan $PS --out $O/EVAL_sp_given.json $(for s in 0 1 2; do echo ctl4_none_s$s=$O/ctl4/none_s$s ctl4sp_given_s$s=$O/ctl4sp/given_s$s ctl4spfo_given_s$s=$O/ctl4sp/fo_given_s$s; done) >> $O/eval.log 2>&1 &
CUDA_VISIBLE_DEVICES=3 $PY $E --plan $PS --shuffled-run --out $O/EVAL_sp_shuffle.json $(for s in 0 1 2; do echo ctl4sp_shuffle_s$s=$O/ctl4sp/shuffle_s$s ctl4spfo_shuffle_s$s=$O/ctl4sp/fo_shuffle_s$s; done) >> $O/eval.log 2>&1 &
F=scripts/ex002/ex002_fgd_outside.py
CUDA_VISIBLE_DEVICES=0 $PY $F --plan $PS --radius 0,1.0 --out $O/FGDOUT_sparse.json $(for s in 0 1 2; do echo ctrl_s$s=$CTRL/none_s$s ctl4_none_s$s=$O/ctl4/none_s$s ctl4sp_given_s$s=$O/ctl4sp/given_s$s ctl4spfo_given_s$s=$O/ctl4sp/fo_given_s$s; done) >> $O/eval.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 $PY $F --plan $PD --radius 0,0.5,1.0 --out $O/FGDOUT_dense.json $(for s in 0 1 2; do echo ctrl_s$s=$CTRL/none_s$s ctl4_none_s$s=$O/ctl4/none_s$s ctl4_given_s$s=$O/ctl4/given_s$s ctl4fo_given_s$s=$O/ctl4/fo_given_s$s; done) >> $O/eval.log 2>&1 &
$PY scripts/ex002/ex002_exec.py --view $VIEW --plan $PD --out $O/EXEC_dense.json ctrl_s0=$CTRL/none_s0 $(for s in 0 1 2; do echo ctl4_none_s$s=$O/ctl4/none_s$s ctl4_given_s$s=$O/ctl4/given_s$s ctl4fo_given_s$s=$O/ctl4/fo_given_s$s; done) >> $O/eval.log 2>&1 &
$PY scripts/ex002/ex002_exec.py --view $VIEW --plan $PS --out $O/EXEC_sparse.json $(for s in 0 1 2; do echo ctl4sp_given_s$s=$O/ctl4sp/given_s$s ctl4spfo_given_s$s=$O/ctl4sp/fo_given_s$s; done) >> $O/eval.log 2>&1 &
$PY - $CTRL $O/ctl4 > $O/IDENTITY.txt 2>&1 <<'PYEOF' &
import sys, numpy as np
from pathlib import Path
a, b = Path(sys.argv[1]), Path(sys.argv[2])
for s in range(3):
    d = [float(np.abs(np.load(f) - np.load(b / f'none_s{s}' / f.name)).max()) for f in sorted((a / f'none_s{s}').glob('*.npy'))]
    print(f'seed {s}: ctl4 none vs control max|diff| over {len(d)} clips = {max(d):.3e} rad (median clip {np.median(d):.3e})')
PYEOF
wait $(jobs -p); log "LANE_DONE"; exit 0
