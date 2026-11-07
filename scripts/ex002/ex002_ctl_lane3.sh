#!/bin/bash
# 11-07 EX002 controllability lane 3 (criteria: docs/criteria/EX002_CTL.md, "lane 3", written before this run). 3 GPUs, one pod:
# ctl3 = sibag-s2 + control modules, base lr 1e-6 / control modules 1e-4 (frozen base), 2000 steps; then none / given / shuffle and
# finger-override given / shuffle x seeds 0-2 (15 runs over 3 GPUs); evaluation (ex002_ctl_run.py eval).
#   ex002_ctl_lane3.sh <out dir>      env PY
set -uo pipefail
O=$1; CODE=$(cd "$(dirname "$0")/../.." && pwd); cd $CODE; mkdir -p $O/ctl3
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
[ -f $O/ctl3/train/DONE.json ] || CUDA_VISIBLE_DEVICES=0 $PY -m scmg.train --stage base --view $VIEW --stats $ST --text-cache $TX --vae $VAE \
    --init $CK0 --out $O/ctl3/train --steps 2000 --batch 64 --lr 1e-6 --warmup 200 --workers 8 --seed 0 --eval-every 1000 --snap-every 1000000 \
    --semsynth-share 0.6 --tok-cache $TOK --tok-intent $B/models/intent-qwen25-v36_1 --intent-frames $B/models/intent-frames-v36_1 \
    --holdout $B/semantic-recognizer-v1/manifest-n200/MANIFEST.json --ctl-tokens $X/tokens --ctl-focus 0.5 \
    --cfg '{"adsc": true, "intent_aux": true, "ctl": true, "ctl_flow": true}' \
    --sc '{"p_stream": 0.67, "p_async": 1.0, "p_async_seg": 0.0, "w_int": 0.3, "sem_w": 2.0, "self_force_p": 0, "ctl_w": 1.0, "w_ctl_aux": 3.0, "ctl_field_drop": 0.2, "ctl_cf_p": 0.5, "ctl_lr_scale": 100.0, "w_ctl_finger": 1.0, "ctl_cf_finger_p": 0.5}' \
    > $O/ctl3/train.log 2>&1 && log "TRAINED ctl3" || { log "TRAIN_FAILED ctl3"; exit 1; }
CK=$O/ctl3/train/final.pt
i=0; J=()
for sd in 0 1 2; do for m in none given shuffle; do J+=("$m|$sd|$O/ctl3/${m}_s$sd|"); done; for m in given shuffle; do J+=("$m|$sd|$O/ctl3/fo_${m}_s$sd|--finger-override $HS"); done; done
for j in "${J[@]}"; do echo "$((i % 3))|$j"; i=$((i + 1)); done > $O/jobs.txt
for g in 0 1 2; do ( grep "^$g|" $O/jobs.txt | while IFS='|' read -r gg m sd od ex; do run $gg $CK $m $sd $od "$ex"; done ) & done; wait $(jobs -p)
log "INFER_DONE"
E="scripts/ex002/ex002_ctl_run.py eval $VA --plan $X/PLAN_dev.json --ctl $X/tokens"
CUDA_VISIBLE_DEVICES=0 $PY $E --out $O/EVAL_given.json $(for s in 0 1 2; do echo ctl3_none_s$s=$O/ctl3/none_s$s ctl3_given_s$s=$O/ctl3/given_s$s ctl3fo_given_s$s=$O/ctl3/fo_given_s$s; done) >> $O/eval.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 $PY $E --shuffled-run --out $O/EVAL_shuffle.json $(for s in 0 1 2; do echo ctl3_shuffle_s$s=$O/ctl3/shuffle_s$s ctl3fo_shuffle_s$s=$O/ctl3/fo_shuffle_s$s; done) >> $O/eval.log 2>&1 &
wait $(jobs -p); log "LANE_DONE"; exit 0
