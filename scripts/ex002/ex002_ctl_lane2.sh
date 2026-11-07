#!/bin/bash
# 11-07 EX002 controllability lane 2 (criteria: docs/criteria/EX002_CTL.md, "lane 2", written before this run). 4 GPUs, one pod:
#   GPU0 train rcp  (sibag-s2 recipe: self-forcing 0.5, BAgger 0.33, no control modules, 2000 steps)
#   GPU1 train ctl2 (sibag-s2 recipe + control modules, light weights)
#   GPU2/3 meanwhile: module 2 (finger override) on the lane-1 model, given / shuffle x seeds 0-2
#   then rcp none x 3; ctl2 none / given / shuffle x 3 and fo_given / fo_shuffle x 3; evaluation (ex002_ctl_run.py eval).
#   ex002_ctl_lane2.sh <out dir> <lane-1 dir>      env PY
set -uo pipefail
O=$1; L1=$2; CODE=$(cd "$(dirname "$0")/../.." && pwd); cd $CODE; mkdir -p $O/rcp $O/ctl2 $O/lane1fo
B=/mnt/cpfs/zbl-cpfs-new/CKPT/zehan/gesture-scmg; C=/mnt/cpfs/zbl-cpfs-new/dataset/zehan/gesture; X=$B/control-token-1105/ex002
PY=${PY:-python}; export SCMG_J=34 SCMG_NO_GLITCH=1 OMP_NUM_THREADS=${OMP_NUM_THREADS:-2} SCMG_TEXT_CAUSAL=$B/models/text-e5-causal-v36_1
VIEW=$C/train-v1-20261101/views/data-v36_1; ST=$C/scmg/STATS_v36_1.json; TX=$C/scmg/text-e5-v36_1; TOK=$C/scmg/tok-qwen25-v36_1
VAE=$B/trained/vae-r3/final.pt; CK0=$B/mainline-seeds-1104b/sibag-s2/final.pt; HS=$X/tokens/HAND_SHAPES.json
VA="--view $VIEW --stats $ST --text-cache $TX"; IA="--tok-cache $TOK --vae $VAE --plan $X/PLAN_dev.json"
log() { echo "$(date -Is) $*" >> $O/lane.log; }
log "START commit $(cat GIT_COMMIT 2>/dev/null || git rev-parse HEAD 2>/dev/null)"
run() {  # gpu ckpt mode seed outdir [extra]
  [ -f $5/DONE ] && return 0
  CUDA_VISIBLE_DEVICES=$1 $PY scripts/ex002/ex002_ctl_run.py infer $VA $IA --ckpt $2 --mode $3 --seed $4 --out $5 ${6:-} >> $O/infer.log 2>&1 \
    && touch $5/DONE || log "INFER_FAILED $5"
}
REC="--semsynth-share 0.6 --tok-cache $TOK --tok-intent $B/models/intent-qwen25-v36_1 --intent-frames $B/models/intent-frames-v36_1 \
  --holdout $B/semantic-recognizer-v1/manifest-n200/MANIFEST.json --self-force-p 0.5 --self-force-steps 4 --self-force-rounds 1 \
  --bagger-dir $B/r4-1104/bagger-pai/bagger-data --bagger-share 0.33"
TR="--stage base --view $VIEW --stats $ST --text-cache $TX --vae $VAE --init $CK0 --steps 2000 --batch 64 --lr 1e-4 --warmup 200 --workers 8 --seed 0 --eval-every 1000 --snap-every 1000000"
S0='"p_stream": 0.67, "p_async": 1.0, "p_async_seg": 0.0, "w_int": 0.3, "sem_w": 2.0'
( [ -f $O/rcp/train/DONE.json ] || CUDA_VISIBLE_DEVICES=0 $PY -m scmg.train $TR $REC --out $O/rcp/train --cfg '{"adsc": true, "intent_aux": true}' \
    --sc "{$S0}" > $O/rcp/train.log 2>&1 && log "TRAINED rcp" || log "TRAIN_FAILED rcp" ) &
( [ -f $O/ctl2/train/DONE.json ] || CUDA_VISIBLE_DEVICES=1 $PY -m scmg.train $TR $REC --out $O/ctl2/train --ctl-tokens $X/tokens --ctl-focus 0.3 \
    --cfg '{"adsc": true, "intent_aux": true, "ctl": true, "ctl_flow": true}' \
    --sc "{$S0, \"ctl_w\": 1.0, \"w_ctl_aux\": 1.0, \"ctl_field_drop\": 0.2, \"ctl_cf_p\": 0.5, \"ctl_lr_scale\": 10.0, \"w_ctl_finger\": 1.0, \"ctl_cf_finger_p\": 0.5}" \
    > $O/ctl2/train.log 2>&1 && log "TRAINED ctl2" || log "TRAIN_FAILED ctl2" ) &
CK1=$L1/ex002ctl/train/final.pt
( for sd in 0 1; do for m in given shuffle; do run 2 $CK1 $m $sd $O/lane1fo/fo_${m}_s$sd "--finger-override $HS"; done; done ) &
( for m in given shuffle; do run 3 $CK1 $m 2 $O/lane1fo/fo_${m}_s2 "--finger-override $HS"; done ) &
wait
for n in rcp ctl2; do [ -f $O/$n/train/final.pt ] || { log "NO_CKPT $n"; exit 1; }; done
J=(); for sd in 0 1 2; do J+=("rcp none $sd") ; for m in none given shuffle; do J+=("ctl2 $m $sd"); done; for m in given shuffle; do J+=("ctl2fo $m $sd"); done; done
i=0; for j in "${J[@]}"; do set -- $j; g=$((i % 4)); i=$((i + 1))
  if [ $1 = ctl2fo ]; then echo "$g|$O/ctl2/train/final.pt|$2|$3|$O/ctl2/fo_${2}_s$3|--finger-override $HS"
  else echo "$g|$O/$1/train/final.pt|$2|$3|$O/$1/${2}_s$3|"; fi
done > $O/jobs.txt
for g in 0 1 2 3; do ( grep "^$g|" $O/jobs.txt | while IFS='|' read -r gg ck m sd od ex; do run $gg $ck $m $sd $od "$ex"; done ) & done; wait
log "INFER_DONE"
E="scripts/ex002/ex002_ctl_run.py eval $VA --plan $X/PLAN_dev.json --ctl $X/tokens"
ls1() { for x in "$@"; do echo $x; done; }
CUDA_VISIBLE_DEVICES=0 $PY $E --out $O/EVAL_given.json $(for s in 0 1 2; do echo rcp_none_s$s=$O/rcp/none_s$s ctl2_none_s$s=$O/ctl2/none_s$s ctl2_given_s$s=$O/ctl2/given_s$s \
  ctl2fo_given_s$s=$O/ctl2/fo_given_s$s lane1fo_given_s$s=$O/lane1fo/fo_given_s$s; done) >> $O/eval.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 $PY $E --shuffled-run --out $O/EVAL_shuffle.json $(for s in 0 1 2; do echo ctl2_shuffle_s$s=$O/ctl2/shuffle_s$s ctl2fo_shuffle_s$s=$O/ctl2/fo_shuffle_s$s \
  lane1fo_shuffle_s$s=$O/lane1fo/fo_shuffle_s$s; done) >> $O/eval.log 2>&1 &
wait; log "LANE_DONE"
