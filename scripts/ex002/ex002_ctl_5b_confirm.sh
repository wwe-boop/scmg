#!/bin/bash
# 11-08 EX002 module 5b confirmation (criteria docs/criteria/EX002_CTL.md "Module 5b confirmation on fresh plans", frozen before
# this job). 4 GPUs, one pod. A: old-plan A800 re-score of 5b (re-blend the lane-4 arrays here). B: fresh plans, NEW lane-4
# generations (seeds 3-5, none / given / shuffle, 15 runs, 2 streams per GPU), 5b as frozen, all gates + bootstrap.
#   ex002_ctl_5b_confirm.sh <out dir>      env PY
set -uo pipefail
O=$1; CODE=$(cd "$(dirname "$0")/../.." && pwd); cd $CODE; mkdir -p $O/old $O/fresh
B=/mnt/cpfs/zbl-cpfs-new/CKPT/zehan/gesture-scmg; C=/mnt/cpfs/zbl-cpfs-new/dataset/zehan/gesture; X=$B/control-token-1105/ex002; L=$X/lane4
PY=${PY:-python}; export SCMG_J=34 SCMG_NO_GLITCH=1 OMP_NUM_THREADS=${OMP_NUM_THREADS:-2} SCMG_TEXT_CAUSAL=$B/models/text-e5-causal-v36_1
VIEW=$C/train-v1-20261101/views/data-v36_1; ST=$C/scmg/STATS_v36_1.json; TX=$C/scmg/text-e5-v36_1; TOK=$C/scmg/tok-qwen25-v36_1
VAE=$B/trained/vae-r3/final.pt; CK=$L/ctl4/train/final.pt; HS=$X/tokens/HAND_SHAPES.json
PD=$X/PLAN_dev.json; PS=$X/PLAN_dev_sparse.json; FD=$X/PLAN_dev_fresh_dense.json; FS=$X/PLAN_dev_fresh_sparse.json
VA="--view $VIEW --stats $ST --text-cache $TX"; IA="--tok-cache $TOK --vae $VAE"; BL="scripts/ex002/ex002_local_blend.py --pre 0.3 --post 0.9"
E="scripts/ex002/ex002_ctl_run.py eval $VA --ctl $X/tokens"; F=scripts/ex002/ex002_fgd_outside.py; XE=scripts/ex002/ex002_exec.py; BS=scripts/ex002/ex002_stg_boot.py
log() { echo "$(date -Is) $*" >> $O/lane.log; }
log "START commit $(cat GIT_COMMIT 2>/dev/null || git rev-parse HEAD 2>/dev/null)"
blend() {  # none_dir given_dir plan out mode
  $PY $BL --none $1 --given $2 --plan $3 --out $4 --mode $5 > /dev/null && $PY $BL --none $1 --given $2 --plan $3 --out $(dirname $4)/fo_$(basename $4) --mode $5 --finger-override $HS > /dev/null
}
run() {  # gpu plan mode seed outdir
  [ -f $5/DONE ] && return 0
  CUDA_VISIBLE_DEVICES=$1 $PY scripts/ex002/ex002_ctl_run.py infer $VA $IA --plan $2 --ckpt $CK --mode $3 --seed $4 --out $5 >> $O/infer_$1.log 2>&1 \
    && touch $5/DONE || log "INFER_FAILED $5"
}
# ---- B (generation) first so the GPUs start at once
i=0; J=()
for sd in 3 4 5; do J+=("$FD|none|$sd|$O/fresh/gen/none_s$sd"); for m in given shuffle; do J+=("$FD|$m|$sd|$O/fresh/gen/d_${m}_s$sd" "$FS|$m|$sd|$O/fresh/gen/s_${m}_s$sd"); done; done
for j in "${J[@]}"; do echo "$((i % 8))|$j"; i=$((i + 1)); done > $O/jobs.txt
for w in 0 1 2 3 4 5 6 7; do ( grep "^$w|" $O/jobs.txt | while IFS='|' read -r ww pl m sd od; do run $((ww % 4)) $pl $m $sd $od; done ) & done
# ---- A: old-plan re-score (CPU blends + GPU0 scoring, alongside the generation)
(
for s in 0 1 2; do for m in given shuffle; do blend $L/ctl4/none_s$s $L/ctl4/${m}_s$s $PD $O/old/dense/${m}_s$s $m; blend $L/ctl4/none_s$s $L/ctl4sp/${m}_s$s $PS $O/old/sparse/${m}_s$s $m; done; done
log "OLD_BLENDED"
export CUDA_VISIBLE_DEVICES=0
$PY $E --plan $PD --out $O/old/EVAL_given.json $(for s in 0 1 2; do echo none_s$s=$L/ctl4/none_s$s m5b_given_s$s=$O/old/dense/given_s$s m5bfo_given_s$s=$O/old/dense/fo_given_s$s; done) >> $O/eval_old.log 2>&1
$PY $E --plan $PD --shuffled-run --out $O/old/EVAL_shuffle.json $(for s in 0 1 2; do echo m5b_shuffle_s$s=$O/old/dense/shuffle_s$s m5bfo_shuffle_s$s=$O/old/dense/fo_shuffle_s$s; done) >> $O/eval_old.log 2>&1
$PY $E --plan $PS --out $O/old/EVAL_sp_given.json $(for s in 0 1 2; do echo none_s$s=$L/ctl4/none_s$s m5bsp_given_s$s=$O/old/sparse/given_s$s m5bspfo_given_s$s=$O/old/sparse/fo_given_s$s; done) >> $O/eval_old.log 2>&1
$PY $E --plan $PS --shuffled-run --out $O/old/EVAL_sp_shuffle.json $(for s in 0 1 2; do echo m5bsp_shuffle_s$s=$O/old/sparse/shuffle_s$s m5bspfo_shuffle_s$s=$O/old/sparse/fo_shuffle_s$s; done) >> $O/eval_old.log 2>&1
$PY $F --plan $PS --radius 0,1.0 --out $O/old/FGDOUT_sparse.json $(for s in 0 1 2; do echo none_s$s=$L/ctl4/none_s$s m5bsp_given_s$s=$O/old/sparse/given_s$s m5bspfo_given_s$s=$O/old/sparse/fo_given_s$s; done) >> $O/eval_old.log 2>&1
$PY $XE --view $VIEW --plan $PS --out $O/old/EXEC_sparse.json $(for s in 0 1 2; do echo m5bsp_given_s$s=$O/old/sparse/given_s$s m5bspfo_given_s$s=$O/old/sparse/fo_given_s$s; done) >> $O/eval_old.log 2>&1
$PY $BS --view $VIEW --plan $PD --ctl $X/tokens --out $O/old/BOOT_m5b.json $(for s in 0 1 2; do echo --pair s$s:$O/old/dense/given_s$s:$L/ctl4/none_s$s; done) >> $O/eval_old.log 2>&1
$PY $BS --view $VIEW --plan $PD --ctl $X/tokens --out $O/old/BOOT_lane4raw.json $(for s in 0 1 2; do echo --pair s$s:$L/ctl4/given_s$s:$L/ctl4/none_s$s; done) >> $O/eval_old.log 2>&1
log "OLD_SCORED"
) &
wait $(jobs -p)
log "FRESH_GENERATED"
G=$O/fresh/gen; R=$O/fresh
for s in 3 4 5; do for m in given shuffle; do blend $G/none_s$s $G/d_${m}_s$s $FD $R/dense/${m}_s$s $m & blend $G/none_s$s $G/s_${m}_s$s $FS $R/sparse/${m}_s$s $m & done; done; wait $(jobs -p)
log "FRESH_BLENDED"
CUDA_VISIBLE_DEVICES=0 $PY $E --plan $FD --out $R/EVAL_given.json $(for s in 3 4 5; do echo none_s$s=$G/none_s$s raw_given_s$s=$G/d_given_s$s m5b_given_s$s=$R/dense/given_s$s m5bfo_given_s$s=$R/dense/fo_given_s$s; done) >> $O/eval_fresh.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 $PY $E --plan $FD --shuffled-run --out $R/EVAL_shuffle.json $(for s in 3 4 5; do echo raw_shuffle_s$s=$G/d_shuffle_s$s m5b_shuffle_s$s=$R/dense/shuffle_s$s m5bfo_shuffle_s$s=$R/dense/fo_shuffle_s$s; done) >> $O/eval_fresh.log 2>&1 &
CUDA_VISIBLE_DEVICES=2 $PY $E --plan $FS --out $R/EVAL_sp_given.json $(for s in 3 4 5; do echo none_s$s=$G/none_s$s raw_sp_given_s$s=$G/s_given_s$s m5bsp_given_s$s=$R/sparse/given_s$s m5bspfo_given_s$s=$R/sparse/fo_given_s$s; done) >> $O/eval_fresh.log 2>&1 &
CUDA_VISIBLE_DEVICES=3 $PY $E --plan $FS --shuffled-run --out $R/EVAL_sp_shuffle.json $(for s in 3 4 5; do echo m5bsp_shuffle_s$s=$R/sparse/shuffle_s$s m5bspfo_shuffle_s$s=$R/sparse/fo_shuffle_s$s; done) >> $O/eval_fresh.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 $PY $F --plan $FS --radius 0,1.0 --out $R/FGDOUT_sparse.json $(for s in 3 4 5; do echo none_s$s=$G/none_s$s raw_sp_given_s$s=$G/s_given_s$s m5bsp_given_s$s=$R/sparse/given_s$s m5bspfo_given_s$s=$R/sparse/fo_given_s$s; done) >> $O/eval_fresh.log 2>&1 &
CUDA_VISIBLE_DEVICES=2 $PY $F --plan $FD --radius 0,0.5,1.0 --out $R/FGDOUT_dense.json $(for s in 3 4 5; do echo none_s$s=$G/none_s$s m5b_given_s$s=$R/dense/given_s$s; done) >> $O/eval_fresh.log 2>&1 &
$PY $XE --view $VIEW --plan $FS --out $R/EXEC_sparse.json $(for s in 3 4 5; do echo m5bsp_given_s$s=$R/sparse/given_s$s m5bspfo_given_s$s=$R/sparse/fo_given_s$s; done) >> $O/eval_fresh.log 2>&1 &
$PY $XE --view $VIEW --plan $FD --out $R/EXEC_dense.json $(for s in 3 4 5; do echo none_s$s=$G/none_s$s m5b_given_s$s=$R/dense/given_s$s m5bfo_given_s$s=$R/dense/fo_given_s$s; done) >> $O/eval_fresh.log 2>&1 &
CUDA_VISIBLE_DEVICES=3 $PY $BS --view $VIEW --plan $FD --ctl $X/tokens --out $R/BOOT_m5b.json $(for s in 3 4 5; do echo --pair s$s:$R/dense/given_s$s:$G/none_s$s; done) >> $O/eval_fresh.log 2>&1 &
CUDA_VISIBLE_DEVICES=0 $PY $BS --view $VIEW --plan $FD --ctl $X/tokens --out $R/BOOT_lane4raw.json $(for s in 3 4 5; do echo --pair s$s:$G/d_given_s$s:$G/none_s$s; done) >> $O/eval_fresh.log 2>&1 &
wait $(jobs -p)
$PY - $L/ctl4 $G > $O/IDENTITY_none.txt 2>&1 <<'PYEOF'
import sys, numpy as np
from pathlib import Path
print('fresh none runs use seeds 3-5 (no bitwise counterpart in lane 4); clips', len(list((Path(sys.argv[2]) / 'none_s3').glob('*.npy'))))
PYEOF
log "LANE_DONE"; exit 0
