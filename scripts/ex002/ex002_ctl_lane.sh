#!/bin/bash
# 11-07 EX002 controllability quick lane (criteria: docs/criteria/EX002_CTL.md, written before this run). 3 GPUs, one pod:
#   GPU0 : train ex002ctl = sibag-s2 + ctl / ctl_flow (aim tokens: counterfactual targets + forearm-ray loss; finger tokens: soft
#          extended-finger count loss + counterfactual counts), EX002 tokens of ex002_ctl_tokens.py, ctl_focus 0.5, 2000 steps,
#          self-forcing off, no BAgger (otherwise the sibag-s2 data arguments)
#   GPU1/2: control runs while it trains: sibag-s2, mode none, seeds 0 / 1 / 2 (delta = seed spread of these FGD v2 values)
#   then : ex002ctl none / given / shuffle x seeds 0 / 1 / 2 on GPUs 0-2; evaluation of every run (ex002_ctl_run.py eval).
#   ex002_ctl_lane.sh <out dir>        env PY
set -uo pipefail
O=$1; CODE=$(cd "$(dirname "$0")/../.." && pwd); cd $CODE; mkdir -p $O
B=/mnt/cpfs/zbl-cpfs-new/CKPT/zehan/gesture-scmg; C=/mnt/cpfs/zbl-cpfs-new/dataset/zehan/gesture; X=$B/control-token-1105/ex002
PY=${PY:-python}; export SCMG_J=34 SCMG_NO_GLITCH=1 OMP_NUM_THREADS=${OMP_NUM_THREADS:-2} SCMG_TEXT_CAUSAL=$B/models/text-e5-causal-v36_1
VIEW=$C/train-v1-20261101/views/data-v36_1; ST=$C/scmg/STATS_v36_1.json; TX=$C/scmg/text-e5-v36_1; TOK=$C/scmg/tok-qwen25-v36_1
VAE=$B/trained/vae-r3/final.pt; CK0=$B/mainline-seeds-1104b/sibag-s2/final.pt
VA="--view $VIEW --stats $ST --text-cache $TX"; IA="--tok-cache $TOK --vae $VAE --plan $X/PLAN_dev.json"
log() { echo "$(date -Is) $*" >> $O/lane.log; }
log "START commit $(cat GIT_COMMIT 2>/dev/null || git rev-parse HEAD 2>/dev/null)"
run() {  # gpu ckpt mode seed outdir
  [ -f $5/DONE ] && return 0
  CUDA_VISIBLE_DEVICES=$1 $PY scripts/ex002/ex002_ctl_run.py infer $VA $IA --ckpt $2 --mode $3 --seed $4 --out $5 >> $O/infer.log 2>&1 \
    && touch $5/DONE || log "INFER_FAILED $5"
}
( [ -f $O/ex002ctl/train/DONE.json ] || CUDA_VISIBLE_DEVICES=0 $PY -m scmg.train --stage base --view $VIEW --stats $ST --text-cache $TX --vae $VAE \
    --init $CK0 --out $O/ex002ctl/train --steps 2000 --batch 64 --lr 1e-4 --warmup 200 --workers 8 --seed 0 --eval-every 1000 --snap-every 1000000 \
    --semsynth-share 0.6 --tok-cache $TOK --tok-intent $B/models/intent-qwen25-v36_1 --intent-frames $B/models/intent-frames-v36_1 \
    --holdout $B/semantic-recognizer-v1/manifest-n200/MANIFEST.json --ctl-tokens $X/tokens --ctl-focus 0.5 \
    --cfg '{"adsc": true, "intent_aux": true, "ctl": true, "ctl_flow": true}' \
    --sc '{"p_stream": 0.67, "p_async": 1.0, "p_async_seg": 0.0, "w_int": 0.3, "sem_w": 2.0, "self_force_p": 0, "ctl_w": 4.0, "w_ctl_aux": 3.0, "ctl_field_drop": 0.2, "ctl_cf_p": 0.5, "ctl_lr_scale": 10.0, "w_ctl_finger": 1.0, "ctl_cf_finger_p": 0.5}' \
    > $O/ex002ctl/train.log 2>&1 || log "TRAIN_FAILED"; log "TRAINED" ) &
mkdir -p $O/ex002ctl $O/ctrl
( run 1 $CK0 none 0 $O/ctrl/none_s0; run 1 $CK0 none 2 $O/ctrl/none_s2 ) &
( run 2 $CK0 none 1 $O/ctrl/none_s1 ) &
wait
CK=$O/ex002ctl/train/final.pt
[ -f $CK ] || { log "NO_CKPT"; exit 1; }
for sd in 0 1 2; do ( for m in none given shuffle; do run $sd $CK $m $sd $O/ex002ctl/${m}_s$sd; done ) & done; wait
log "INFER_DONE"
E="scripts/ex002/ex002_ctl_run.py eval $VA --plan $X/PLAN_dev.json --ctl $X/tokens"
CUDA_VISIBLE_DEVICES=0 $PY $E --out $O/EVAL_ctrl.json $(for s in 0 1 2; do echo ctrl_none_s$s=$O/ctrl/none_s$s; done) >> $O/eval.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 $PY $E --out $O/EVAL_ex002ctl.json $(for s in 0 1 2; do for m in none given; do echo ex002ctl_${m}_s$s=$O/ex002ctl/${m}_s$s; done; done) >> $O/eval.log 2>&1 &
CUDA_VISIBLE_DEVICES=2 $PY $E --shuffled-run --out $O/EVAL_ex002ctl_shuffle.json $(for s in 0 1 2; do echo ex002ctl_shuffle_s$s=$O/ex002ctl/shuffle_s$s; done) >> $O/eval.log 2>&1 &
wait; log "LANE_DONE"
